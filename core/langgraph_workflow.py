from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, TypedDict
from pathlib import Path
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph

from core.llm_client import LLMClient
from core.schemas import (
    DraftReview,
    EvidenceItem,
    FinalReport,
    ResearchSection,
    ReviewFeedback,
    ReviewScore,
    SubQuestion,
    WorkflowResult,
)
from core.settings import Settings
from core.tracer import TraceCollector
from agents.reviewer import ReviewerAgent
from rag.text_utils import tokenize_text
from tools.citation_checker import CitationChecker
from tools.knowledge_base import KnowledgeBaseTool
from tools.web_search import WebSearchTool


class WorkflowState(TypedDict, total=False):
    topic: str
    use_web: bool
    use_kb: bool
    report_language: str
    max_subquestions: int
    max_search_rounds: int
    enable_reviewer: bool
    allow_query_rewrite: bool
    replan_count: int
    sub_questions: list[SubQuestion]
    sections: list[ResearchSection]
    final_report: FinalReport
    review: DraftReview
    review_feedback: list[str]
    review_feedback_structured: ReviewFeedback
    reviewer_replan_count: int
    repair_targets: list[str]
    repair_sub_questions: list[SubQuestion]


class LangGraphResearchWorkflow:
    def __init__(
        self, settings: Settings, tracer: TraceCollector | None = None
    ) -> None:
        self.settings = settings
        self.tracer = tracer or TraceCollector()
        self.llm_client = (
            LLMClient(settings, self.tracer) if settings.has_llm_credentials else None
        )
        self.web_search_tool = WebSearchTool(settings, self.tracer)
        self.knowledge_base_tool = KnowledgeBaseTool(settings, self.tracer)
        self.reviewer = ReviewerAgent(self.tracer, self.llm_client)
        self.citation_checker = CitationChecker()
        self.graph = self._build_graph()

    def _debug(self, message: str) -> None:
        if self.settings.log_level.upper() != "DEBUG":
            return
        timestamp = time.strftime("%H:%M:%S")
        print(f"[DEBUG {timestamp}] {message}", file=sys.stderr, flush=True)

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("reviewer", self._reviewer_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "writer")
        graph.add_edge("writer", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            self._review_router,
            {
                "planner": "planner",
                "end": END,
            },
        )
        return graph.compile()

    def run(
        self,
        *,
        topic: str,
        use_web: bool = True,
        use_kb: bool = True,
        max_subquestions: int | None = None,
        max_search_rounds: int | None = None,
        report_language: str | None = None,
        enable_reviewer: bool = True,
        allow_query_rewrite: bool = True,
    ) -> WorkflowResult:
        state: WorkflowState = {
            "topic": topic,
            "use_web": use_web,
            "use_kb": use_kb,
            "report_language": report_language or self.settings.report_language,
            "max_subquestions": (
                self.settings.max_subquestions
                if max_subquestions is None
                else max_subquestions
            ),
            "max_search_rounds": max_search_rounds or self.settings.max_search_rounds,
            "enable_reviewer": enable_reviewer,
            "allow_query_rewrite": allow_query_rewrite,
            "replan_count": 0,
            "reviewer_replan_count": 0,
            "review_feedback": [],
            "review_feedback_structured": ReviewFeedback(),
            "repair_targets": [],
            "repair_sub_questions": [],
        }
        result_state = self.graph.invoke(state)
        final_report = result_state["final_report"]
        final_report.content = self._append_references(final_report)
        result = WorkflowResult(
            topic=topic,
            sub_questions=result_state.get("sub_questions", []),
            sections=result_state.get("sections", []),
            final_report=final_report,
            trace=self.tracer.events,
        )
        self._persist_run_artifact(result, result_state)
        return result

    def _planner_node(self, state: WorkflowState) -> WorkflowState:
        self._debug(
            f"planner.start topic={state['topic']} max_subquestions={state['max_subquestions']} "
            f"replan_count={state.get('replan_count', 0)}"
        )
        repair_targets = state.get("repair_targets", [])
        if repair_targets and state.get("sub_questions"):
            repair_sub_questions, merged_sub_questions = self._plan_repair_subquestions(
                topic=state["topic"],
                repair_targets=repair_targets,
                previous_subquestions=state.get("sub_questions", []),
            )
            self._debug(
                f"planner.done repair_targets={len(repair_targets)} "
                f"repair_sub_questions={len(repair_sub_questions)}"
            )
            return {
                **state,
                "sub_questions": merged_sub_questions,
                "repair_sub_questions": repair_sub_questions,
            }

        questions = self._plan_subquestions(
            topic=state["topic"],
            max_subquestions=state["max_subquestions"],
            review_feedback=state.get("review_feedback"),
            review_feedback_structured=state.get("review_feedback_structured"),
            previous_subquestions=state.get("sub_questions"),
        )
        self._debug(f"planner.done sub_questions={len(questions)}")
        return {
            **state,
            "sub_questions": questions,
            "repair_sub_questions": [],
        }

    def _researcher_node(self, state: WorkflowState) -> WorkflowState:
        self._debug(
            f"researcher.start sub_questions={len(state.get('sub_questions', []))} "
            f"use_web={state['use_web']} use_kb={state['use_kb']} max_rounds={state['max_search_rounds']}"
        )
        active_sub_questions = state.get("repair_sub_questions") or state.get("sub_questions", [])
        existing_sections = {
            section.sub_question: section
            for section in state.get("sections", [])
        }
        updated_sections: dict[str, ResearchSection] = {}
        for sub_question in active_sub_questions:
            self._debug(f"researcher.sub_question.start question={sub_question.question}")
            section = self._research_subquestion(
                topic=state["topic"],
                sub_question=sub_question,
                max_rounds=state["max_search_rounds"],
                use_web=state["use_web"],
                use_kb=state["use_kb"],
                allow_query_rewrite=state["allow_query_rewrite"],
                review_feedback=state.get("review_feedback"),
            )
            # keep previous draft until writer repairs this section
            if sub_question.question in existing_sections:
                section.draft_section = existing_sections[sub_question.question].draft_section
            updated_sections[sub_question.question] = section
            self._debug(
                f"researcher.sub_question.done question={sub_question.question} "
                f"evidence_count={len(section.evidence_items)} rounds={len(section.metadata.get('round_details', []))}"
            )

        merged_sections: list[ResearchSection] = []
        for sub_question in state.get("sub_questions", []):
            if sub_question.question in updated_sections:
                merged_sections.append(updated_sections[sub_question.question])
            elif sub_question.question in existing_sections:
                merged_sections.append(existing_sections[sub_question.question])
        for question, section in updated_sections.items():
            if question not in {item.question for item in state.get("sub_questions", [])}:
                merged_sections.append(section)

        self._debug(
            f"researcher.done sections={len(merged_sections)} "
            f"active_sub_questions={len(active_sub_questions)}"
        )
        return {
            **state,
            "sections": merged_sections,
        }

    def _writer_node(self, state: WorkflowState) -> WorkflowState:
        self._debug(
            f"writer.start sections={len(state.get('sections', []))} "
            f"references={sum(len(s.evidence_items) for s in state.get('sections', []))}"
        )
        self._reset_intermediate_artifacts(state["topic"])
        sections = [section.model_copy(deep=True) for section in state.get("sections", [])]
        repair_targets = set(state.get("repair_targets", []))
        references = [
            item
            for section in sections
            for item in section.evidence_items
        ]
        invoke_started_at = time.perf_counter()
        self._debug("writer.section_loop.start")
        sections, section_writer_trace = self._write_sections(
            state=state,
            sections=sections,
            repair_targets=repair_targets,
        )
        self._debug(f"writer.section_loop.done section_count={len(sections)}")
        content = self._synthesize_report(
            state=state,
            sections=sections,
        )
        invoke_duration_ms = int((time.perf_counter() - invoke_started_at) * 1000)
        self._debug(
            f"writer.synthesize.done duration_ms={invoke_duration_ms} "
            f"writer_trace={len(section_writer_trace)} content_len={len(content)}"
        )

        final_report = FinalReport(
            topic=state["topic"],
            outline=[item.question for item in state.get("sub_questions", [])],
            content=content,
            references=references,
            writer_trace=section_writer_trace,
        )
        return {
            **state,
            "sections": sections,
            "final_report": final_report,
        }

    def _write_sections(
        self,
        *,
        state: WorkflowState,
        sections: list[ResearchSection],
        repair_targets: set[str],
    ) -> tuple[list[ResearchSection], list[dict[str, Any]]]:
        writer_trace: list[dict[str, Any]] = []
        if self.llm_client is None:
            for index, section in enumerate(sections, start=1):
                if repair_targets and section.sub_question not in repair_targets and section.draft_section:
                    continue
                section.draft_section = section.summary or "证据不足。"
                self._persist_section_progress(
                    state=state,
                    sections=sections,
                    writer_trace=writer_trace,
                    completed_index=index,
                )
            return sections, writer_trace

        for index, section in enumerate(sections, start=1):
            if repair_targets and section.sub_question not in repair_targets and section.draft_section:
                self._debug(
                    f"writer.section.skip index={index} sub_question={section.sub_question}"
                )
                continue
            self._debug(
                f"writer.section.start index={index} sub_question={section.sub_question} "
                f"evidence_count={len(section.evidence_items)}"
            )
            web_items = [item for item in section.evidence_items if item.source_type == "web"]
            kb_items = [item for item in section.evidence_items if item.source_type == "kb"]
            draft, section_trace = self._run_section_writer_loop(
                state=state,
                section=section,
                web_items=web_items,
                kb_items=kb_items,
            )
            section.draft_section = draft
            writer_trace.extend(section_trace)
            self._persist_section_progress(
                state=state,
                sections=sections,
                writer_trace=writer_trace,
                completed_index=index,
            )
            self._debug(
                f"writer.section.done index={index} draft_len={len(draft)} "
                f"web={len(web_items)} kb={len(kb_items)}"
            )
        return sections, writer_trace

    def _run_section_writer_loop(
        self,
        *,
        state: WorkflowState,
        section: ResearchSection,
        web_items: list[EvidenceItem],
        kb_items: list[EvidenceItem],
    ) -> tuple[str, list[dict[str, Any]]]:
        assert self.llm_client is not None

        writer_trace: list[dict[str, Any]] = []
        loaded_details: list[dict[str, Any]] = []
        loaded_ids: set[str] = set()
        max_steps = 3

        for step in range(1, max_steps + 1):
            self._debug(f"writer.section.step.start sub_question={section.sub_question} step={step}")
            prompt = self._build_section_selection_prompt(
                state=state,
                section=section,
                web_items=web_items,
                kb_items=kb_items,
                loaded_details=loaded_details,
                step=step,
                max_steps=max_steps,
            )
            payload = self.llm_client.generate_json(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1400,
            )
            thought = str(payload.get("thought", "")).strip()
            enough = bool(payload.get("enough", False))
            selected_ids_raw = payload.get("selected_evidence_ids", [])
            selected_ids = [
                str(value).strip()
                for value in selected_ids_raw
                if str(value).strip()
            ][:3]

            trace_item: dict[str, Any] = {
                "stage": "section_writer",
                "sub_question": section.sub_question,
                "step": step,
                "thought": thought,
                "enough": enough,
                "selected_evidence_ids": selected_ids,
            }

            if enough:
                self._debug(
                    f"writer.section.step.enough sub_question={section.sub_question} "
                    f"step={step} loaded_detail_count={len(loaded_details)}"
                )
                section_draft = self._generate_section_draft_from_loaded_details(
                    state=state,
                    section=section,
                    web_items=web_items,
                    kb_items=kb_items,
                    loaded_details=loaded_details,
                )
                trace_item["draft_len"] = len(section_draft)
                writer_trace.append(trace_item)
                self._debug(
                    f"writer.section.step.final sub_question={section.sub_question} "
                    f"step={step} draft_len={len(section_draft)}"
                )
                return section_draft, writer_trace

            new_details = self._fetch_section_evidence_details(
                section=section,
                selected_ids=selected_ids,
                loaded_ids=loaded_ids,
            )
            loaded_details.extend(new_details)
            loaded_ids.update(detail["evidence_id"] for detail in new_details)
            trace_item["loaded_detail_count"] = len(new_details)
            trace_item["total_loaded_detail_count"] = len(loaded_details)
            writer_trace.append(trace_item)
            self._debug(
                f"writer.section.step.done sub_question={section.sub_question} "
                f"step={step} selected={len(selected_ids)} loaded_new={len(new_details)} "
                f"loaded_total={len(loaded_details)}"
            )

        draft = self._generate_section_draft_from_loaded_details(
            state=state,
            section=section,
            web_items=web_items,
            kb_items=kb_items,
            loaded_details=loaded_details,
        )
        writer_trace.append(
            {
                "stage": "section_writer",
                "sub_question": section.sub_question,
                "step": max_steps + 1,
                "thought": "达到最大 section writer 步数，使用已加载证据生成 section draft。",
                "enough": True,
                "selected_evidence_ids": [],
                "loaded_detail_count": len(loaded_details),
                "draft_len": len(draft),
                "action": "fallback_section_draft",
            }
        )
        self._debug(
            f"writer.section.fallback sub_question={section.sub_question} "
            f"draft_len={len(draft)} loaded_total={len(loaded_details)}"
        )
        return draft, writer_trace

    def _build_section_selection_prompt(
        self,
        *,
        state: WorkflowState,
        section: ResearchSection,
        web_items: list[EvidenceItem],
        kb_items: list[EvidenceItem],
        loaded_details: list[dict[str, Any]],
        step: int,
        max_steps: int,
    ) -> str:
        def render_catalog(items: list[EvidenceItem]) -> str:
            if not items:
                return "无"
            blocks: list[str] = []
            for item in items:
                blocks.append(
                    "\n".join(
                        [
                            f"evidence_id: {item.evidence_id}",
                            f"title: {item.title}",
                            f"source_url: {item.source_url}",
                            f"snippet: {item.snippet}",
                        ]
                    )
                )
            return "\n\n".join(blocks)

        loaded_block = (
            json.dumps(loaded_details, ensure_ascii=False, indent=2)
            if loaded_details
            else "[]"
        )

        return f"""
你是研究报告的 Section Writer。你这一次只需要回答一个子问题，不要处理其他子问题。
你的任务分两步：
1. 先阅读当前子问题在两个来源上的全部检索结果目录（title/snippet/url）。
2. 再判断是否已经足够回答；如果不够，就只挑出少数关键 evidence_id 让我继续加载它们的详细内容。

研究主题：{state['topic']}
报告语言：{state['report_language']}
当前子问题：{section.sub_question}
当前步数：{step}/{max_steps}

Web 检索结果目录：
{render_catalog(web_items)}

本地知识库结果目录：
{render_catalog(kb_items)}

已加载的关键证据详情：
{loaded_block}

决策要求：
1. 只围绕当前子问题作答。
2. 只有当“结果目录 + 已加载详情”已经足够覆盖核心组件、关键机制、性能瓶颈与实现细节时，才返回 enough=true，并直接给出 section_draft。
3. 如果还不够，就返回 enough=false，并在 selected_evidence_ids 中列出你还需要查看详细内容的 evidence_id，最多 3 个。
4. 优先选择最关键、最有代表性的证据，不要把所有 evidence_id 都选出来。
5. 所有关键结论都必须带 evidence_id 引用，如 [WEB-1]、[KB-01-001]。
6. 如果某些方面证据不足，要明确写“证据不足”。
7. 对系统/架构类问题，不要停留在科普层面；如果还缺少“组件职责、执行流程、关键数据结构、调度/缓存/并行机制、性能收益来源”这些信息，就不要过早 enough=true。
7. 不要输出空行，不要使用 **。

严格返回 JSON：
{{
  "thought": "...",
  "enough": false,
  "selected_evidence_ids": ["WEB-1", "KB-01-001"]
}}

如果 enough=true，则 selected_evidence_ids 为空列表。不要在 JSON 里输出 section draft。
""".strip()

    def _fetch_section_evidence_details(
        self,
        *,
        section: ResearchSection,
        selected_ids: list[str],
        loaded_ids: set[str],
    ) -> list[dict[str, Any]]:
        detail_map = {item.evidence_id: item for item in section.evidence_items}
        loaded: list[dict[str, Any]] = []
        for evidence_id in selected_ids:
            if evidence_id in loaded_ids:
                continue
            item = detail_map.get(evidence_id)
            if item is None:
                continue
            loaded.append(
                {
                    "evidence_id": item.evidence_id,
                    "title": item.title,
                    "source_type": item.source_type,
                    "source_url": item.source_url,
                    "content": item.content or item.snippet,
                }
            )
        return loaded

    def _generate_section_draft_from_loaded_details(
        self,
        *,
        state: WorkflowState,
        section: ResearchSection,
        web_items: list[EvidenceItem],
        kb_items: list[EvidenceItem],
        loaded_details: list[dict[str, Any]],
    ) -> str:
        assert self.llm_client is not None
        loaded_chars = sum(len(str(detail.get("content", ""))) for detail in loaded_details)
        prompt = f"""
你是研究报告的 Section Writer。请基于当前子问题的检索目录与已加载关键证据详情，直接输出这一节的 Markdown 内容。

研究主题：{state['topic']}
报告语言：{state['report_language']}
当前子问题：{section.sub_question}

Web 结果条数：{len(web_items)}
KB 结果条数：{len(kb_items)}

已加载关键证据详情：
{json.dumps(loaded_details, ensure_ascii=False, indent=2) if loaded_details else '[]'}

要求：
1. 第一行使用二级标题。
2. 只回答当前子问题。
3. 所有关键结论都带 evidence_id 引用。
4. 证据不足时明确写“证据不足”。
5. 输出要偏技术分析，不要只写科普式概述。
6. 尽量覆盖：组件职责、执行流程、关键机制、性能瓶颈、这些机制为什么有效、可能的实现约束。
7. 如果材料支持，适当写出更细的工程细节，例如 prefill/decode、KV Cache、调度粒度、内存块/分页、并行方式、吞吐/时延/显存利用率之间的关系。
8. 在不虚构的前提下，尽量写得完整一些，不要只有泛泛概括。
9. 不要输出空行，不要使用 **。
""".strip()
        self._debug(
            f"writer.section.generate_draft.start sub_question={section.sub_question} "
            f"loaded_detail_count={len(loaded_details)} loaded_chars={loaded_chars} "
            f"prompt_len={len(prompt)}"
        )
        draft = self.llm_client.generate_text(
            prompt=prompt,
            temperature=0.1,
            max_tokens=5000,
        ).strip()
        self._debug(
            f"writer.section.generate_draft.done sub_question={section.sub_question} "
            f"draft_len={len(draft)}"
        )
        return draft

    def _synthesize_report(
        self,
        *,
        state: WorkflowState,
        sections: list[ResearchSection],
    ) -> str:
        self._persist_synthesis_input(state=state, sections=sections)
        if self.llm_client is None:
            return "\n".join(
                [section.draft_section or section.summary for section in sections]
            ).strip()

        section_block = "\n\n".join(
            [
                "\n".join(
                    [
                        f"子问题：{section.sub_question}",
                        "section_draft:",
                        section.draft_section or section.summary,
                    ]
                )
                for section in sections
            ]
        )
        prompt = f"""
你是研究报告的 Synthesizer。请把多个子问题的 section drafts 组织成一篇完整、连贯的 Markdown 研究报告。

研究主题：{state['topic']}
报告语言：{state['report_language']}

Section drafts:
{section_block}

要求：
1. 保留原有 evidence_id 引用，不要伪造新引用。
2. 把各 section draft 组织成一篇完整报告，补足必要过渡，但不要引入材料外信息。
3. 去掉重复表述，让整篇报告逻辑连贯。
4. 如果某个 section draft 已写“证据不足”，保留这种结论。
5. 如果材料已经足够支撑对系统/架构的整体理解，可以补一版简洁的 ASCII 图帮助读者理解组件关系；如果证据不足则不要强行画。
6. ASCII 图只用纯文本字符，例如 + - | >，不要使用 mermaid。
7. 输出要偏技术综述，不要停留在面向初学者的科普层。
8. 优先保留并强化技术细节：关键模块职责、执行路径、数据流、缓存/调度/并行策略、这些策略解决的瓶颈以及性能收益来源。
9. 如果 section drafts 已经提供足够证据，尽量把报告写得更完整一些，而不是只保留结论摘要。
10. 输出最终 Markdown 报告，不要输出空行，不要使用 **。
""".strip()
        return self.llm_client.generate_text(
            prompt=prompt,
            temperature=0.1,
            max_tokens=5000,
        ).strip()

    def _reset_intermediate_artifacts(self, topic: str) -> None:
        target_dir = self._intermediate_dir(topic)
        target_dir.mkdir(parents=True, exist_ok=True)
        for pattern in (
            "section_*.md",
            "section_writer_progress.json",
            "synthesis_input.json",
            "synthesis_input.md",
        ):
            for file_path in target_dir.glob(pattern):
                file_path.unlink(missing_ok=True)

    def _persist_section_progress(
        self,
        *,
        state: WorkflowState,
        sections: list[ResearchSection],
        writer_trace: list[dict[str, Any]],
        completed_index: int,
    ) -> None:
        target_dir = self._intermediate_dir(state["topic"])
        target_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "topic": state["topic"],
            "sub_questions": [
                item.model_dump(mode="json") for item in state.get("sub_questions", [])
            ],
            "completed_sections": completed_index,
            "sections": [
                {
                    "index": index,
                    "sub_question": section.sub_question,
                    "query_history": section.query_history,
                    "evidence_count": len(section.evidence_items),
                    "evidence_ids": [item.evidence_id for item in section.evidence_items],
                    "draft_section": section.draft_section,
                }
                for index, section in enumerate(sections, start=1)
            ],
            "writer_trace": writer_trace,
        }

        progress_path = target_dir / "section_writer_progress.json"
        with progress_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        section = sections[completed_index - 1]
        markdown_path = target_dir / f"section_{completed_index:02d}.md"
        markdown_body = "\n".join(
            [
                f"# Section {completed_index}",
                f"## 子问题",
                section.sub_question,
                "## 回答",
                section.draft_section or "尚未生成",
            ]
        ).strip()
        markdown_path.write_text(markdown_body + "\n", encoding="utf-8")

    def _persist_synthesis_input(
        self,
        *,
        state: WorkflowState,
        sections: list[ResearchSection],
    ) -> None:
        target_dir = self._intermediate_dir(state["topic"])
        target_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "topic": state["topic"],
            "sub_questions": [
                item.model_dump(mode="json") for item in state.get("sub_questions", [])
            ],
            "sections": [
                {
                    "index": index,
                    "sub_question": section.sub_question,
                    "draft_section": section.draft_section,
                    "evidence_count": len(section.evidence_items),
                    "evidence_ids": [item.evidence_id for item in section.evidence_items],
                }
                for index, section in enumerate(sections, start=1)
            ],
        }

        json_path = target_dir / "synthesis_input.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        blocks = [f"# {state['topic']}", "## 子问题列表"]
        for index, item in enumerate(state.get("sub_questions", []), start=1):
            blocks.append(f"{index}. {item.question}")
        blocks.append("## Section Drafts")
        for index, section in enumerate(sections, start=1):
            blocks.append(f"### Section {index}: {section.sub_question}")
            blocks.append(section.draft_section or "尚未生成")
        (target_dir / "synthesis_input.md").write_text(
            "\n".join(blocks).strip() + "\n",
            encoding="utf-8",
        )

    def _intermediate_dir(self, topic: str) -> Path:
        return self.settings.trace_path / "intermediate" / self._topic_slug(topic)

    @staticmethod
    def _topic_slug(topic: str) -> str:
        return (
            "".join(char if char.isalnum() else "_" for char in topic)[:40].strip("_")
            or "run"
        )

    def _run_writer_react_loop(
        self,
        *,
        state: WorkflowState,
        sections: list[ResearchSection],
        references: list[EvidenceItem],
        reference_map: dict[str, EvidenceItem],
    ) -> tuple[str, list[dict[str, Any]]]:
        if self.llm_client is None:
            return self._generate_writer_final_answer(
                state=state,
                sections=sections,
                references=references,
                observations=[],
            ), []

        writer_trace: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        max_steps = 8

        for step in range(1, max_steps + 1):
            prompt = self._build_writer_react_prompt(
                state=state,
                sections=sections,
                observations=observations,
                step=step,
                max_steps=max_steps,
            )
            self._debug(f"writer.react.step.start step={step}")
            payload = self.llm_client.generate_json(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1200,
            )
            thought = str(payload.get("thought", "")).strip()
            action = str(payload.get("action", "")).strip()
            action_input = payload.get("action_input", {})
            final_answer = str(payload.get("final_answer", "")).strip()

            trace_item: dict[str, Any] = {
                "step": step,
                "thought": thought,
                "action": action,
                "action_input": action_input,
            }

            if action == "final_answer":
                content = final_answer or self._generate_writer_final_answer(
                    state=state,
                    sections=sections,
                    references=references,
                    observations=observations,
                )
                trace_item["final_answer_len"] = len(content)
                writer_trace.append(trace_item)
                self._debug(f"writer.react.step.final step={step} content_len={len(content)}")
                return content, writer_trace

            observation = self._execute_writer_tool(
                action=action,
                action_input=action_input,
                state=state,
                sections=sections,
                references=references,
                reference_map=reference_map,
            )
            observation_preview = observation[:400]
            observations.append(
                {
                    "step": step,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation_preview,
                }
            )
            trace_item["observation_preview"] = observation_preview
            writer_trace.append(trace_item)
            self._debug(f"writer.react.step.done step={step} action={action}")

        content = self._generate_writer_final_answer(
            state=state,
            sections=sections,
            references=references,
            observations=observations,
        )
        writer_trace.append(
            {
                "step": max_steps + 1,
                "thought": "达到最大 ReAct 步数，转入最终报告生成。",
                "action": "final_answer_fallback",
                "final_answer_len": len(content),
            }
        )
        self._debug(f"writer.react.fallback_final content_len={len(content)}")
        return content, writer_trace

    def _build_writer_react_prompt(
        self,
        *,
        state: WorkflowState,
        sections: list[ResearchSection],
        observations: list[dict[str, Any]],
        step: int,
        max_steps: int,
    ) -> str:
        sub_questions_block = "\n".join(
            f"{idx}. {item.question}"
            for idx, item in enumerate(state.get("sub_questions", []), start=1)
        ) or "无"
        section_stats_block = "\n".join(
            f"{idx}. 子问题={section.sub_question} | evidence_count={len(section.evidence_items)}"
            for idx, section in enumerate(sections, start=1)
        ) or "无"
        observation_block = (
            json.dumps(observations[-6:], ensure_ascii=False, indent=2)
            if observations
            else "[]"
        )
        return f"""
你是一个研究报告 Writer ReAct Agent。你需要在有限步数内决定是否继续调用工具，或者直接完成最终研究报告。

研究主题：{state['topic']}
报告语言：{state['report_language']}
当前步数：{step}/{max_steps}

子问题列表：
{sub_questions_block}

当前 section 概览：
{section_stats_block}

已完成的工具调用观察：
{observation_block}

可用 action：
1. list_subquestions
2. get_section_evidence
3. get_evidence_by_source
4. get_evidence_detail
5. final_answer

决策规则：
- 如果还没看过全部子问题，先用 list_subquestions。
- 如果还没读某个 section 的证据，先用 get_section_evidence。
- 如果来源质量不明，才用 get_evidence_by_source。
- 如果想引用某条具体证据，才用 get_evidence_detail。
- 不要重复调用已经拿到足够信息的工具。
- 如果现有观察已经足够支撑写作，优先选择 final_answer。
- 最终报告必须是结构化 Markdown，使用二级标题，所有结论都要带证据编号。
- 如果证据不足，要明确写“证据不足”。

严格返回 JSON，不要输出额外解释：
{{
  "thought": "...",
  "action": "list_subquestions|get_section_evidence|get_evidence_by_source|get_evidence_detail|final_answer",
  "action_input": {{}},
  "final_answer": ""
}}

如果 action 是：
- list_subquestions: action_input={{}}
- get_section_evidence: action_input={{"section_index": 1}}
- get_evidence_by_source: action_input={{"source_type": "web"}}
- get_evidence_detail: action_input={{"evidence_id": "WEB-1"}}
- final_answer: 把完整报告放进 final_answer
""".strip()

    def _execute_writer_tool(
        self,
        *,
        action: str,
        action_input: Any,
        state: WorkflowState,
        sections: list[ResearchSection],
        references: list[EvidenceItem],
        reference_map: dict[str, EvidenceItem],
    ) -> str:
        if action == "list_subquestions":
            return self._writer_tool_list_subquestions(state)
        if action == "get_section_evidence":
            section_index = self._safe_int_from_mapping(action_input, "section_index", 1)
            return self._writer_tool_get_section_evidence(sections, section_index)
        if action == "get_evidence_by_source":
            source_type = self._safe_str_from_mapping(action_input, "source_type", "web")
            return self._writer_tool_get_evidence_by_source(references, source_type)
        if action == "get_evidence_detail":
            evidence_id = self._safe_str_from_mapping(action_input, "evidence_id", "")
            return self._writer_tool_get_evidence_detail(reference_map, evidence_id)
        return f"unsupported action: {action}"

    def _writer_tool_list_subquestions(self, state: WorkflowState) -> str:
        self._debug("writer.tool.list_subquestions.start")
        output = "\n".join(
            f"{idx}. {item.question}"
            for idx, item in enumerate(state.get("sub_questions", []), start=1)
        )
        self._debug(
            f"writer.tool.list_subquestions.done count={len(state.get('sub_questions', []))}"
        )
        return output

    def _writer_tool_get_section_evidence(
        self, sections: list[ResearchSection], section_index: int
    ) -> str:
        self._debug(
            f"writer.tool.get_section_evidence.start section_index={section_index}"
        )
        if section_index < 1 or section_index > len(sections):
            self._debug(
                f"writer.tool.get_section_evidence.invalid section_index={section_index}"
            )
            return "invalid section index"
        section = sections[section_index - 1]
        lines = [f"子问题：{section.sub_question}"]
        for item in section.evidence_items:
            excerpt = item.content[:220] if item.content else item.snippet[:220]
            lines.append(
                f"[{item.evidence_id}] {item.title}\n来源：{item.source_url}\n内容：{excerpt}"
            )
        output = "\n\n".join(lines)
        self._debug(
            f"writer.tool.get_section_evidence.done section_index={section_index} "
            f"evidence_count={len(section.evidence_items)}"
        )
        return output

    def _writer_tool_get_evidence_by_source(
        self, references: list[EvidenceItem], source_type: str
    ) -> str:
        self._debug(
            f"writer.tool.get_evidence_by_source.start source_type={source_type}"
        )
        normalized = source_type.strip().lower()
        if normalized not in {"web", "kb"}:
            self._debug(
                f"writer.tool.get_evidence_by_source.invalid source_type={source_type}"
            )
            return "source_type must be one of: web, kb"
        matched = [item for item in references if item.source_type == normalized]
        if not matched:
            self._debug(
                f"writer.tool.get_evidence_by_source.empty source_type={normalized}"
            )
            return f"no evidence found for source_type={normalized}"
        lines = [f"source_type={normalized}, total={len(matched)}"]
        for item in matched:
            excerpt = item.content[:220] if item.content else item.snippet[:220]
            lines.append(
                f"[{item.evidence_id}] {item.title}\n来源：{item.source_url}\n内容：{excerpt}"
            )
        output = "\n\n".join(lines)
        self._debug(
            f"writer.tool.get_evidence_by_source.done source_type={normalized} count={len(matched)}"
        )
        return output

    def _writer_tool_get_evidence_detail(
        self, reference_map: dict[str, EvidenceItem], evidence_id: str
    ) -> str:
        self._debug(
            f"writer.tool.get_evidence_detail.start evidence_id={evidence_id}"
        )
        key = evidence_id.strip()
        item = reference_map.get(key)
        if item is None:
            self._debug(
                f"writer.tool.get_evidence_detail.miss evidence_id={key}"
            )
            return f"evidence_id not found: {key}"
        content = item.content or item.snippet
        output = "\n".join(
            [
                f"evidence_id: {item.evidence_id}",
                f"title: {item.title}",
                f"source_type: {item.source_type}",
                f"source_url: {item.source_url}",
                f"metadata: {item.metadata}",
                "content:",
                content,
            ]
        )
        self._debug(
            f"writer.tool.get_evidence_detail.done evidence_id={key} source_type={item.source_type}"
        )
        return output

    def _generate_writer_final_answer(
        self,
        *,
        state: WorkflowState,
        sections: list[ResearchSection],
        references: list[EvidenceItem],
        observations: list[dict[str, Any]],
    ) -> str:
        if self.llm_client is None:
            lines = [f"# {state['topic']}"]
            for section in sections:
                lines.append(f"## {section.sub_question}")
                lines.append(section.summary or "证据不足。")
            return "\n".join(lines)

        section_block = "\n\n".join(
            [
                "\n".join(
                    [
                        f"子问题：{section.sub_question}",
                        f"证据条数：{len(section.evidence_items)}",
                        f"摘要：{section.summary}",
                    ]
                )
                for section in sections
            ]
        )
        observation_block = json.dumps(observations[-8:], ensure_ascii=False, indent=2)
        prompt = f"""
你是研究报告 Writer。请基于已收集到的 section 摘要和工具观察，直接输出最终 Markdown 报告。

研究主题：{state['topic']}
报告语言：{state['report_language']}

Section 信息：
{section_block}

工具观察：
{observation_block}

要求：
1. 使用二级标题组织报告。
2. 对每个子问题给出明确回答。
3. 只使用证据中已经出现的 evidence_id。
4. 所有关键结论都要带引用编号。
5. 证据不足时明确写“证据不足”。
6. 不要输出空行，不要使用 **。
""".strip()
        return self.llm_client.generate_text(
            prompt=prompt,
            temperature=0.1,
            max_tokens=3000,
        )

    @staticmethod
    def _safe_int_from_mapping(value: Any, key: str, default: int) -> int:
        if not isinstance(value, dict):
            return default
        raw = value.get(key, default)
        try:
            return int(raw)
        except Exception:
            return default

    @staticmethod
    def _safe_str_from_mapping(value: Any, key: str, default: str) -> str:
        if not isinstance(value, dict):
            return default
        raw = value.get(key, default)
        return str(raw).strip()

    def _reviewer_node(self, state: WorkflowState) -> WorkflowState:
        self._debug(
            f"reviewer.start content_len={len(state['final_report'].content)} "
            f"references={len(state['final_report'].references)}"
        )
        final_report = state["final_report"]
        if not state["enable_reviewer"]:
            review = DraftReview(accepted=True)
        else:
            review = self.reviewer.review(
                final_report.content,
                {item.evidence_id for item in final_report.references},
                sub_questions=state.get("sub_questions"),
                sections=state.get("sections"),
                references=final_report.references,
            )
        final_report.review = review
        repair_targets = self._build_repair_targets(review.feedback)
        return {
            **state,
            "final_report": final_report,
            "review": review,
            "review_feedback": list(review.suggestions),
            "review_feedback_structured": review.feedback,
            "repair_targets": repair_targets,
        }

    def _review_router(self, state: WorkflowState) -> str:
        review = state.get("review")
        if (
            review
            and not review.accepted
            and state.get("enable_reviewer", True)
            and state.get("replan_count", 0) < self.settings.max_replan_cycles
        ):
            state["replan_count"] = state.get("replan_count", 0) + 1
            state["reviewer_replan_count"] = state.get("reviewer_replan_count", 0) + 1
            return "planner"
        return "end"

    def _plan_repair_subquestions(
        self,
        *,
        topic: str,
        repair_targets: list[str],
        previous_subquestions: list[SubQuestion],
    ) -> tuple[list[SubQuestion], list[SubQuestion]]:
        previous_map = {item.question: item for item in previous_subquestions}
        repair_sub_questions: list[SubQuestion] = []
        merged_sub_questions = list(previous_subquestions)

        for target in repair_targets:
            matched = previous_map.get(target)
            if matched is not None:
                repair_sub_questions.append(matched)
                continue

            new_item = SubQuestion(
                index=len(merged_sub_questions) + 1,
                question=target,
                rationale="Reviewer 定向补差规划",
            )
            merged_sub_questions.append(new_item)
            repair_sub_questions.append(new_item)

        return repair_sub_questions, merged_sub_questions

    @staticmethod
    def _build_repair_targets(feedback: ReviewFeedback) -> list[str]:
        targets: list[str] = []
        seen: set[str] = set()
        for question in [*feedback.missing_subquestions, *feedback.weak_sections]:
            normalized = question.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            targets.append(normalized)
        return targets

    def _append_references(self, final_report: FinalReport) -> str:
        references_markdown = self.citation_checker.build_reference_markdown(
            final_report.references
        )
        return f"{final_report.content}\n\n{references_markdown}".strip()

    def _persist_run_artifact(
        self, result: WorkflowResult, state: WorkflowState | None = None
    ) -> None:
        self.settings.trace_path.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        web_evidence = [
            item
            for section in result.sections
            for item in section.evidence_items
            if item.source_type == "web"
        ]
        web_search_round_count = 0
        for section in result.sections:
            for round_detail in section.metadata.get("round_details", []):
                if round_detail.get("web_searches"):
                    web_search_round_count += 1
        payload["run_summary"] = {
            "reference_count": len(result.final_report.references),
            "section_count": len(result.sections),
            "trace_count": len(result.trace),
            "web_evidence_count": len(web_evidence),
            "web_search_round_count": web_search_round_count,
            "writer_react_turn_count": len(result.final_report.writer_trace),
            "reviewer_replan_count": (state or {}).get("reviewer_replan_count", 0),
            "accepted": result.final_report.review.accepted
            if result.final_report.review
            else False,
        }
        latest_path = self.settings.trace_path / "latest_run.json"
        slug = (
            "".join(char if char.isalnum() else "_" for char in result.topic)[
                :40
            ].strip("_")
            or "run"
        )
        topic_path = self.settings.trace_path / f"{slug}_run.json"
        for target in (latest_path, topic_path):
            with Path(target).open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _serialize_writer_trace(messages: list[Any]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for message in messages:
            if hasattr(message, "model_dump"):
                try:
                    serialized.append(message.model_dump(mode="json"))
                    continue
                except Exception:
                    pass
            serialized.append(
                {
                    "type": type(message).__name__,
                    "content": getattr(message, "content", ""),
                    "name": getattr(message, "name", None),
                    "tool_calls": getattr(message, "tool_calls", None),
                }
            )
        return serialized

    def _plan_subquestions(
        self,
        *,
        topic: str,
        max_subquestions: int,
        review_feedback: list[str] | None = None,
        review_feedback_structured: ReviewFeedback | None = None,
        previous_subquestions: list[SubQuestion] | None = None,
    ) -> list[SubQuestion]:
        review_feedback_structured = review_feedback_structured or ReviewFeedback()
        if self.llm_client is not None:
            previous_questions = previous_subquestions or []
            feedback_block = (
                "\n".join(f"- {item}" for item in (review_feedback or [])) or "无"
            )
            previous_block = (
                "\n".join(f"- {item.question}" for item in previous_questions) or "无"
            )
            structured_feedback_block = "\n".join(
                [
                    "结构化审查反馈：",
                    "未覆盖子问题："
                    + (
                        "；".join(review_feedback_structured.missing_subquestions)
                        if review_feedback_structured.missing_subquestions
                        else "无"
                    ),
                    "证据薄弱分节："
                    + (
                        "；".join(review_feedback_structured.weak_sections)
                        if review_feedback_structured.weak_sections
                        else "无"
                    ),
                    "低质量来源："
                    + (
                        "；".join(review_feedback_structured.low_quality_sources[:5])
                        if review_feedback_structured.low_quality_sources
                        else "无"
                    ),
                ]
            )
            prompt = f"""
请把研究主题拆成若干个彼此不重复、适合检索和写作的子问题。
对于复杂技术问题，优先拆成 3-6 个子问题；只有在主题明显很简单时，才少于 3 个。
如果调用方给了正整数上限，则不要超过该上限；如果上限为 0，则由你根据主题复杂度自行决定。
如果已有审查反馈，请优先根据反馈补足证据不足、结构不完整或引用不足的部分。
返回 JSON，格式如下：
{{
  "sub_questions": [
    {{"question": "...", "rationale": "..."}}
  ]
}}

研究主题：{topic}
上一轮子问题：
{previous_block}

审查反馈：
{feedback_block}

{structured_feedback_block}
""".strip()
            try:
                data = self.llm_client.generate_json(prompt=prompt, temperature=0.1)
                raw_items = data.get("sub_questions", [])
                questions: list[SubQuestion] = []
                limited_items = (
                    raw_items[:max_subquestions]
                    if max_subquestions and max_subquestions > 0
                    else raw_items
                )
                for index, item in enumerate(limited_items, start=1):
                    question = str(item.get("question", "")).strip()
                    if not question:
                        continue
                    questions.append(
                        SubQuestion(
                            index=index,
                            question=question,
                            rationale=str(item.get("rationale", "")).strip(),
                        )
                    )
                if questions:
                    return questions
            except Exception:
                pass

        templates: list[str] = []
        if review_feedback:
            templates.extend(
                [
                    "{topic} 里哪些关键结论最需要补充证据和引用？",
                    "{topic} 里哪些内容最容易因为资料不足而写得不完整？",
                ]
            )
        templates.extend(
            [
                "{topic} 的背景、定义和核心问题是什么？",
                "{topic} 的关键技术路线或核心方法有哪些？",
                "{topic} 当前的工程优化重点和实践难点是什么？",
                "{topic} 的典型应用场景、优势与局限是什么？",
                "{topic} 的发展趋势和未来值得关注的方向是什么？",
            ]
        )
        return [
            SubQuestion(
                index=index,
                question=template.format(topic=topic),
                rationale="LangGraph 模板规划",
            )
            for index, template in enumerate(
                templates[:max_subquestions] if max_subquestions and max_subquestions > 0 else templates,
                start=1,
            )
        ]

    def _research_subquestion(
        self,
        *,
        topic: str,
        sub_question: SubQuestion,
        max_rounds: int,
        use_web: bool,
        use_kb: bool,
        allow_query_rewrite: bool,
        review_feedback: list[str] | None = None,
    ) -> ResearchSection:
        query_history: list[str] = []
        evidence_items: list[EvidenceItem] = []
        seen_keys: set[str] = set()
        round_details: list[dict[str, Any]] = []

        if use_kb and not self.knowledge_base_tool.is_ready:
            self.knowledge_base_tool.load()

        for round_index in range(max_rounds):
            round_queries = self._generate_round_queries(
                topic=topic,
                original_question=sub_question.question,
                evidence_items=evidence_items,
                round_index=round_index,
                review_feedback=review_feedback,
                previous_queries=query_history,
            )
            if not round_queries:
                break

            query_history.extend(round_queries)
            round_items: list[EvidenceItem] = []
            round_detail: dict[str, Any] = {
                "round_index": round_index + 1,
                "queries": round_queries,
                "web_searches": [],
                "kb_searches": [],
            }

            for query in round_queries:
                for source in ("web", "kb"):
                    if source == "web" and not use_web:
                        continue
                    if source == "kb" and not use_kb:
                        continue
                    try:
                        if source == "web":
                            result = self.web_search_tool.search(query, 10)
                        else:
                            result = self.knowledge_base_tool.retrieve(query, 6)
                    except Exception as exc:  # noqa: BLE001
                        detail = {
                            "query": query,
                            "error": str(exc),
                            "result_count": 0,
                        }
                        if source == "web":
                            round_detail["web_searches"].append(detail)
                        else:
                            round_detail["kb_searches"].append(detail)
                        if self.tracer is not None:
                            self.tracer.add_event(
                                event_type="workflow",
                                step_name="researcher.query_fanout",
                                status="error",
                                input_summary=query,
                                output_summary=str(exc),
                                metadata={
                                    "sub_question": sub_question.question,
                                    "round_index": round_index + 1,
                                    "source": source,
                                    "query": query,
                                    "result_count": 0,
                                },
                            )
                        continue

                    round_items.extend(result.items)
                    detail = {
                        "query": query,
                        "result_count": result.total,
                        "metadata": result.metadata,
                    }
                    if source == "web":
                        round_detail["web_searches"].append(detail)
                    else:
                        round_detail["kb_searches"].append(detail)
                    if self.tracer is not None:
                        self.tracer.add_event(
                            event_type="workflow",
                            step_name="researcher.query_fanout",
                            status="success",
                            input_summary=query,
                            output_summary=f"{source} 命中 {result.total} 条结果",
                            metadata={
                                "sub_question": sub_question.question,
                                "round_index": round_index + 1,
                                "source": source,
                                "query": query,
                                "result_count": result.total,
                            },
                        )

            for item in round_items:
                dedupe_key = (
                    item.source_url
                    or f"{item.title}-{item.chunk_id}-{item.snippet[:80]}"
                )
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                item.metadata["sub_question"] = sub_question.question
                evidence_items.append(item)

            round_detail["evidence_count_after_round"] = len(evidence_items)
            round_details.append(round_detail)
            if self.tracer is not None:
                self.tracer.add_event(
                    event_type="workflow",
                    step_name="researcher.round",
                    status="success",
                    input_summary=sub_question.question,
                    output_summary=f"第 {round_index + 1} 轮累计证据 {len(evidence_items)} 条",
                    metadata={
                        "sub_question": sub_question.question,
                        "round_index": round_index + 1,
                        "queries": round_queries,
                        "web_query_count": len(round_detail["web_searches"]),
                        "kb_query_count": len(round_detail["kb_searches"]),
                        "evidence_count_after_round": len(evidence_items),
                    },
                )

            if self._has_enough_evidence(evidence_items):
                break

            if not allow_query_rewrite:
                continue

        return ResearchSection(
            sub_question=sub_question.question,
            query_history=query_history,
            evidence_items=evidence_items,
            summary=self._summarize_evidence(sub_question.question, evidence_items),
            metadata={
                "round_details": round_details,
                "final_evidence_count": len(evidence_items),
            },
        )

    def _generate_round_queries(
        self,
        *,
        topic: str,
        original_question: str,
        evidence_items: list[EvidenceItem],
        round_index: int,
        review_feedback: list[str] | None = None,
        previous_queries: list[str] | None = None,
    ) -> list[str]:
        previous_queries = previous_queries or []
        previous_set = set(previous_queries)
        current_query = previous_queries[-1] if previous_queries else original_question

        llm_candidates = self._generate_query_candidates(
            topic=topic,
            original_question=original_question,
            current_query=current_query,
            evidence_items=evidence_items,
            review_feedback=review_feedback,
        )
        heuristic_candidates = self._heuristic_query_candidates(
            topic=topic,
            original_question=original_question,
            current_query=current_query,
            evidence_items=evidence_items,
            review_feedback=review_feedback,
        )

        combined: list[str] = []
        seen: set[str] = set()
        for query in [original_question, *llm_candidates, *heuristic_candidates]:
            normalized = str(query).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            combined.append(normalized)

        unseen = [query for query in combined if query not in previous_set]
        fanout = 2
        if unseen:
            return unseen[:fanout]

        fallback = self._rewrite_query(
            topic=topic,
            original_question=original_question,
            current_query=current_query,
            evidence_items=evidence_items,
            round_index=round_index,
            review_feedback=review_feedback,
        )
        if fallback and fallback not in previous_set:
            return [fallback]
        return []

    def _heuristic_query_candidates(
        self,
        *,
        topic: str,
        original_question: str,
        current_query: str,
        evidence_items: list[EvidenceItem],
        review_feedback: list[str] | None = None,
    ) -> list[str]:
        simplified = (
            original_question.replace("是什么", "")
            .replace("有哪些", "")
            .replace("？", "")
            .replace("?", "")
            .strip()
        )
        candidates = [
            original_question,
            f"{topic} {simplified}".strip(),
            simplified.replace("RAG", "retrieval augmented generation RAG").replace(
                "LLM", "large language model LLM"
            ),
            f"{simplified} 原理 工程 实践".strip(),
            f"{simplified} 综述".strip(),
            f"{topic} {simplified} 优化".strip(),
        ]
        if not evidence_items:
            candidates.append(f"{topic} {original_question} 综述".strip())
        else:
            latest_titles = " ".join(item.title for item in evidence_items[-2:])
            feedback_hint = " ".join(review_feedback[:2]) if review_feedback else ""
            candidates.append(f"{current_query} {latest_titles} {feedback_hint}".strip())

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(normalized)
        return unique_candidates

    def _rewrite_query(
        self,
        *,
        topic: str,
        original_question: str,
        current_query: str,
        evidence_items: list[EvidenceItem],
        round_index: int,
        review_feedback: list[str] | None = None,
    ) -> str:
        llm_candidates = self._generate_query_candidates(
            topic=topic,
            original_question=original_question,
            current_query=current_query,
            evidence_items=evidence_items,
            review_feedback=review_feedback,
        )
        if round_index + 1 < len(llm_candidates):
            return llm_candidates[round_index + 1]

        heuristic_candidates = self._heuristic_query_candidates(
            topic=topic,
            original_question=original_question,
            current_query=current_query,
            evidence_items=evidence_items,
            review_feedback=review_feedback,
        )

        if round_index + 1 < len(heuristic_candidates):
            candidate = heuristic_candidates[round_index + 1]
            if review_feedback:
                return f"{candidate} {' '.join(review_feedback[:2])}".strip()
            return candidate

        if not evidence_items:
            fallback = f"{topic} {original_question} 综述"
            if review_feedback:
                return f"{fallback} {' '.join(review_feedback[:2])}".strip()
            return fallback

        latest_titles = " ".join(item.title for item in evidence_items[-2:])
        feedback_hint = " ".join(review_feedback[:2]) if review_feedback else ""
        return f"{current_query} {latest_titles} {feedback_hint}".strip()

    def _generate_query_candidates(
        self,
        *,
        topic: str,
        original_question: str,
        current_query: str,
        evidence_items: list[EvidenceItem],
        review_feedback: list[str] | None = None,
    ) -> list[str]:
        if self.llm_client is None:
            return []

        evidence_titles = [item.title for item in evidence_items[-5:]]
        feedback_block = "；".join(review_feedback[:3]) if review_feedback else "无"
        evidence_block = "；".join(evidence_titles) if evidence_titles else "无"
        prompt = f"""
你是研究检索 Query Rewrite 助手。请针对一个研究子问题，生成多个可用于 Web Search / 本地知识库检索的候选查询。
要求：
1. 返回 4 到 6 个彼此不同的 query。
2. query 要尽量覆盖：原始问法、关键词精简版、英文术语展开版、综述/原理/实践导向版本。
3. 不要输出无关解释，不要输出编号，只返回 JSON。
4. JSON 格式固定为：
{{
  "queries": ["...", "..."]
}}

研究主题：{topic}
原始子问题：{original_question}
当前查询：{current_query}
已有证据标题：{evidence_block}
审查/补检索反馈：{feedback_block}
""".strip()

        try:
            payload = self.llm_client.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_tokens=800,
            )
        except Exception:
            return []

        raw_queries = payload.get("queries", [])
        if not isinstance(raw_queries, list):
            return []

        candidates: list[str] = []
        seen: set[str] = set()
        for value in [original_question, *raw_queries]:
            query = str(value).strip()
            if not query or query in seen:
                continue
            seen.add(query)
            candidates.append(query)
        return candidates

    @staticmethod
    def _has_enough_evidence(evidence_items: list[EvidenceItem]) -> bool:
        source_types = {item.source_type for item in evidence_items}
        return len(evidence_items) >= 4 and len(source_types) >= 1

    @staticmethod
    def _summarize_evidence(
        question: str, evidence_items: list[EvidenceItem]
    ) -> str:
        if not evidence_items:
            return f"围绕“{question}”暂未检索到足够证据。"
        lines = [f"围绕“{question}”共收集到 {len(evidence_items)} 条证据："]
        for item in evidence_items[:5]:
            excerpt = item.snippet or item.content[:120]
            lines.append(f"- [{item.evidence_id}] {item.title}：{excerpt[:120]}")
        return "\n".join(lines)

    def _review_report(
        self,
        draft: str,
        valid_ids: set[str],
        *,
        sub_questions: list[SubQuestion] | None = None,
        sections: list[ResearchSection] | None = None,
        references: list[EvidenceItem] | None = None,
        minimum_total: int = 12,
    ) -> DraftReview:
        sub_questions = sub_questions or []
        sections = sections or []
        references = references or []

        citations = self.citation_checker.extract_citations(draft)
        has_valid_citations, invalid_ids = self.citation_checker.validate(
            draft, valid_ids
        )
        covered_subquestions, missing_subquestions = self._evaluate_subquestion_coverage(
            draft, sub_questions, sections
        )
        weak_sections = [
            section.sub_question for section in sections if len(section.evidence_items) < 2
        ]
        weak_single_source_sections = [
            section.sub_question
            for section in sections
            if len(section.evidence_items) == 1
            and not self._is_high_quality_source(section.evidence_items[0])
        ]
        high_quality_count = sum(
            1 for item in references if self._is_high_quality_source(item)
        )
        low_quality_sources = [
            item.source_url or item.title
            for item in references
            if not self._is_high_quality_source(item)
        ]
        source_quality_ratio = (
            high_quality_count / len(references) if references else 0.0
        )

        factual_support = (
            5
            if citations and not weak_sections and source_quality_ratio >= 0.6
            else 4
            if citations and not weak_single_source_sections
            else 2
            if citations
            else 1
        )
        citation_coverage = (
            5 if citations and has_valid_citations else 2 if citations else 0
        )
        coherence = 4 if "##" in draft and len(draft) > 300 else 2
        completeness = (
            5
            if sub_questions
            and covered_subquestions == len(sub_questions)
            and not weak_sections
            else 4
            if sub_questions and covered_subquestions >= max(1, len(sub_questions) - 1)
            else 3
            if draft.count("## ") >= 2
            else 2
        )

        score = ReviewScore(
            factual_support=factual_support,
            citation_coverage=citation_coverage,
            coherence=coherence,
            completeness=completeness,
        )

        suggestions: list[str] = []
        if not citations:
            suggestions.append("需要补充引用编号，保证每个关键结论可追溯。")
        if invalid_ids:
            suggestions.append(f"存在无效引用编号：{', '.join(invalid_ids)}。")
        if draft.count("## ") < 2:
            suggestions.append("建议增加分节结构，增强报告可读性。")
        if len(draft) < 300:
            suggestions.append("当前内容偏短，建议补充更多证据总结。")
        if missing_subquestions:
            suggestions.append(
                "以下子问题未被充分回答：" + "；".join(missing_subquestions[:3]) + "。"
            )
        if weak_sections:
            suggestions.append(
                "以下分节证据不足（少于 2 条）：" + "；".join(weak_sections[:3]) + "。"
            )
        if weak_single_source_sections:
            suggestions.append(
                "以下分节仅依赖单条弱来源证据："
                + "；".join(weak_single_source_sections[:3])
                + "。"
            )
        if references and source_quality_ratio < 0.4:
            suggestions.append(
                "当前引用中低质量来源占比偏高，建议补充官方文档、仓库 README 或高质量媒体来源。"
            )

        accepted = (
            score.total >= minimum_total
            and has_valid_citations
            and (not sub_questions or covered_subquestions == len(sub_questions))
            and not weak_single_source_sections
        )

        feedback = ReviewFeedback(
            missing_subquestions=missing_subquestions,
            weak_sections=list(dict.fromkeys([*weak_sections, *weak_single_source_sections])),
            low_quality_sources=list(dict.fromkeys(low_quality_sources[:5])),
        )

        return DraftReview(
            accepted=accepted,
            score=score,
            suggestions=suggestions,
            feedback=feedback,
        )

    def _evaluate_subquestion_coverage(
        self,
        draft: str,
        sub_questions: list[SubQuestion],
        sections: list[ResearchSection],
    ) -> tuple[int, list[str]]:
        if not sub_questions:
            return 0, []
        normalized_draft = self._normalize_text(draft)
        normalized_section_map = {
            self._normalize_text(section.sub_question): section for section in sections
        }
        covered = 0
        missing: list[str] = []
        for item in sub_questions:
            normalized_question = self._normalize_text(item.question)
            matched_section = normalized_section_map.get(normalized_question)
            has_section = matched_section is not None
            has_evidence = bool(matched_section and matched_section.evidence_items)

            keyword_signals = self._extract_subquestion_keywords(item.question)
            hit_count = sum(
                1 for signal in keyword_signals if signal and signal in normalized_draft
            )

            if (has_section and has_evidence and hit_count >= 1) or hit_count >= 2:
                covered += 1
            else:
                missing.append(item.question)
        return covered, missing

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[\W_]+", "", text.lower())

    def _extract_subquestion_keywords(self, question: str) -> list[str]:
        stopwords = {
            "什么",
            "哪些",
            "如何",
            "为什么",
            "以及",
            "分别",
            "通过",
            "采用",
            "实现",
            "进行",
            "有关",
            "方面",
            "关键",
            "哪些关",
            "what",
            "which",
            "how",
            "why",
            "does",
            "with",
            "using",
        }
        candidates: list[str] = []
        seen: set[str] = set()
        for token in tokenize_text(question):
            normalized = self._normalize_text(token)
            if len(normalized) < 2:
                continue
            if normalized in stopwords:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(normalized)
        return candidates[:10]

    def _is_high_quality_source(self, item: EvidenceItem) -> bool:
        if item.source_type == "kb":
            return True
        url = item.source_url.strip().lower()
        if not url:
            return False
        try:
            hostname = (urlparse(url).hostname or "").lower()
        except Exception:
            return False
        high_quality_domains = (
            "github.com",
            "raw.githubusercontent.com",
            "openai.com",
            "help.openai.com",
            "docs.vllm.ai",
            "developers.llamaindex.ai",
            "docs.llamaindex.ai",
            "milvus.io",
            "qdrant.tech",
            "api.qdrant.tech",
            "ithome.com",
            "ithome.com.tw",
        )
        if any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in high_quality_domains
        ):
            if "/issues/" in url or "/forum/" in url or "/discussions/" in url:
                return False
            return True
        return False
