from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.reviewer import ReviewerAgent
from agents.writer import WriterAgent
from core.llm_client import LLMClient
from core.schemas import (
    DraftReview,
    EvidenceItem,
    FinalReport,
    ResearchSection,
    SubQuestion,
    WorkflowResult,
)
from core.settings import Settings
from core.tracer import TraceCollector
from tools.citation_checker import CitationChecker
from tools.knowledge_base import KnowledgeBaseTool
from tools.web_search import WebSearchTool


class WorkflowState(str, Enum):
    INIT = "INIT"
    PLANNED = "PLANNED"
    RESEARCHING = "RESEARCHING"
    WRITING = "WRITING"
    REVIEWING = "REVIEWING"
    DONE = "DONE"
    FAILED = "FAILED"


class WorkflowContext(BaseModel):
    topic: str
    state: WorkflowState = WorkflowState.INIT
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    sections: list[ResearchSection] = Field(default_factory=list)
    draft_report: str = ""
    final_report: FinalReport | None = None
    review: DraftReview | None = None


class ResearchWorkflow:
    def __init__(
        self, settings: Settings, tracer: TraceCollector | None = None
    ) -> None:
        self.settings = settings
        self.tracer = tracer or TraceCollector()
        self.llm_client = (
            LLMClient(settings, self.tracer) if settings.has_llm_credentials else None
        )
        self.web_search_tool = WebSearchTool(
            settings, self.tracer, llm_client=self.llm_client
        )
        self.knowledge_base_tool = KnowledgeBaseTool(settings, self.tracer)
        self.planner = PlannerAgent(self.llm_client, self.tracer)
        self.researcher = ResearcherAgent(
            self.web_search_tool, self.knowledge_base_tool, self.tracer
        )
        self.writer = WriterAgent(self.llm_client, self.tracer)
        self.reviewer = ReviewerAgent(self.tracer, self.llm_client)
        self.citation_checker = CitationChecker()

    def create_context(self, topic: str) -> WorkflowContext:
        context = WorkflowContext(topic=topic)
        self.tracer.add_event(
            event_type="workflow",
            step_name="workflow.create_context",
            status="success",
            input_summary=topic,
            output_summary="已创建工作流上下文",
            metadata={"state": context.state.value},
        )
        return context

    def update_state(
        self, context: WorkflowContext, new_state: WorkflowState
    ) -> WorkflowContext:
        old_state = context.state
        context.state = new_state
        self.tracer.add_event(
            event_type="workflow",
            step_name="workflow.update_state",
            status="success",
            input_summary=f"{old_state.value} -> {new_state.value}",
            output_summary="状态更新成功",
            metadata={"old_state": old_state.value, "new_state": new_state.value},
        )
        return context

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
        progress_callback: Callable[[str], None] | None = None,
    ) -> WorkflowResult:
        context = self.create_context(topic)
        self._notify(progress_callback, f"已创建工作流上下文：{topic}")

        review_feedback: list[str] = []
        previous_subquestions: list[SubQuestion] = []
        final_report: FinalReport | None = None
        review = DraftReview(accepted=False)

        for replan_cycle in range(self.settings.max_replan_cycles + 1):
            if replan_cycle > 0:
                self.tracer.add_event(
                    event_type="workflow",
                    step_name="workflow.replan_cycle",
                    status="success",
                    input_summary=topic,
                    output_summary=f"进入第 {replan_cycle} 次重新规划",
                    metadata={"review_feedback": review_feedback},
                )
                self._notify(
                    progress_callback,
                    f"审查未通过，开始第 {replan_cycle} 次重新规划与补检索",
                )

            context.sub_questions = self.planner.plan(
                topic,
                self.settings.max_subquestions
                if max_subquestions is None
                else max_subquestions,
                review_feedback=review_feedback,
                previous_subquestions=previous_subquestions,
            )
            previous_subquestions = list(context.sub_questions)
            self.update_state(context, WorkflowState.PLANNED)
            self._notify(
                progress_callback,
                f"规划完成，生成 {len(context.sub_questions)} 个子问题",
            )

            self.update_state(context, WorkflowState.RESEARCHING)
            self._notify(progress_callback, "开始检索阶段")
            sections = []
            all_evidence = []
            for sub_question in context.sub_questions:
                section = self.researcher.research(
                    topic=topic,
                    sub_question=sub_question,
                    max_rounds=max_search_rounds or self.settings.max_search_rounds,
                    use_web=use_web,
                    use_kb=use_kb,
                    allow_query_rewrite=allow_query_rewrite,
                    review_feedback=review_feedback,
                )
                sections.append(section)
                all_evidence.extend(section.evidence_items)
                self._notify(
                    progress_callback,
                    f"已完成子问题检索：{sub_question.question}（证据 {len(section.evidence_items)} 条）",
                )

            context.sections = sections
            context.evidence_items = all_evidence

            self.update_state(context, WorkflowState.WRITING)
            self._notify(progress_callback, "开始写作阶段")
            final_report = self.writer.write(
                topic,
                context.sub_questions,
                sections,
                report_language=report_language or self.settings.report_language,
                review_feedback=review_feedback,
            )
            context.draft_report = final_report.content

            self.update_state(context, WorkflowState.REVIEWING)
            self._notify(progress_callback, "开始审查阶段")
            valid_ids = {item.evidence_id for item in final_report.references}
            if enable_reviewer:
                review = self.reviewer.review(
                    final_report.content,
                    valid_ids,
                    sub_questions=context.sub_questions,
                    sections=context.sections,
                    references=final_report.references,
                )
                final_report.review = review
                if review.accepted:
                    break
                review_feedback = list(review.suggestions)
                if replan_cycle >= self.settings.max_replan_cycles:
                    break
                continue

            review = DraftReview(accepted=True)
            final_report.review = review
            break

        assert final_report is not None
        final_report.content = self._append_references(final_report)
        context.final_report = final_report
        context.review = review

        self.update_state(
            context, WorkflowState.DONE if review.accepted else WorkflowState.FAILED
        )
        self._notify(progress_callback, f"流程结束：{context.state.value}")

        result = WorkflowResult(
            topic=topic,
            sub_questions=context.sub_questions,
            sections=context.sections,
            final_report=final_report,
            trace=self.tracer.events,
        )
        self._persist_run_artifact(result)
        return result

    def _append_references(self, final_report: FinalReport) -> str:
        references_markdown = self.citation_checker.build_reference_markdown(
            final_report.references
        )
        return f"{final_report.content}\n\n{references_markdown}".strip()

    def _persist_run_artifact(self, result: WorkflowResult) -> None:
        self.settings.trace_path.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        web_evidence = [
            item
            for section in result.sections
            for item in section.evidence_items
            if item.source_type == "web"
        ]
        payload["run_summary"] = {
            "reference_count": len(result.final_report.references),
            "section_count": len(result.sections),
            "trace_count": len(result.trace),
            "web_evidence_count": len(web_evidence),
            "writer_react_turn_count": len(result.final_report.writer_trace),
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
    def _notify(progress_callback: Callable[[str], None] | None, message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)
