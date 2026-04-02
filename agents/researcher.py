from __future__ import annotations

from core.schemas import EvidenceItem, ResearchSection, SearchResult, SubQuestion
from core.tracer import TraceCollector
from tools.knowledge_base import KnowledgeBaseTool
from tools.web_search import WebSearchTool


class ResearcherAgent:
    def __init__(
        self,
        web_search_tool: WebSearchTool | None = None,
        knowledge_base_tool: KnowledgeBaseTool | None = None,
        tracer: TraceCollector | None = None,
    ) -> None:
        self.web_search_tool = web_search_tool
        self.knowledge_base_tool = knowledge_base_tool
        self.tracer = tracer

    def research(
        self,
        *,
        topic: str,
        sub_question: SubQuestion,
        max_rounds: int = 3,
        use_web: bool = True,
        use_kb: bool = True,
        allow_query_rewrite: bool = True,
        review_feedback: list[str] | None = None,
    ) -> ResearchSection:
        query_history: list[str] = []
        evidence_items: list[EvidenceItem] = []
        seen_keys: set[str] = set()
        current_query = sub_question.question

        for round_index in range(max_rounds):
            query_history.append(current_query)
            round_items: list[EvidenceItem] = []
            round_metadata: dict[str, object] = {"query": current_query}

            if use_web and self.web_search_tool is not None:
                web_result = self.web_search_tool.search(current_query, max_results=4)
                round_items.extend(web_result.items)
                round_metadata["web_search"] = web_result.metadata

            if use_kb and self.knowledge_base_tool is not None:
                kb_result = self.knowledge_base_tool.retrieve(current_query, top_k=4)
                round_items.extend(kb_result.items)
                round_metadata["kb_search"] = kb_result.metadata

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

            if self._has_enough_evidence(evidence_items):
                break

            if not allow_query_rewrite:
                continue

            current_query = self._rewrite_query(
                topic=topic,
                original_question=sub_question.question,
                current_query=current_query,
                evidence_items=evidence_items,
                round_index=round_index,
                review_feedback=review_feedback,
            )

        summary = self._summarize_evidence(sub_question.question, evidence_items)
        if self.tracer is not None:
            self.tracer.add_event(
                event_type="agent",
                step_name="agent.researcher",
                status="success",
                input_summary=sub_question.question,
                output_summary=f"收集 {len(evidence_items)} 条证据",
                metadata={"queries": query_history, "final_evidence_count": len(evidence_items)},
            )

        return ResearchSection(
            sub_question=sub_question.question,
            query_history=query_history,
            evidence_items=evidence_items,
            summary=summary,
        )

    def _build_query_candidates(
        self, topic: str, question: str, allow_query_rewrite: bool
    ) -> list[str]:
        simplified = (
            question.replace("是什么", "")
            .replace("有哪些", "")
            .replace("？", "")
            .replace("?", "")
            .strip()
        )
        if not allow_query_rewrite:
            return [question]

        expanded = simplified.replace("RAG", "retrieval augmented generation RAG")
        expanded = expanded.replace("LLM", "large language model LLM")

        candidates = [
            question,
            f"{topic} {simplified}".strip(),
            expanded.strip(),
            f"{simplified} 原理 工程 实践".strip(),
            f"{simplified} 综述".strip(),
            f"{topic} {simplified} 优化".strip(),
        ]
        seen: set[str] = set()
        unique: list[str] = []
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            unique.append(candidate)
        return unique

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
        candidates = self._build_query_candidates(topic, original_question, True)
        if round_index + 1 < len(candidates):
            candidate = candidates[round_index + 1]
            if review_feedback:
                feedback_hint = " ".join(review_feedback[:2])
                return f"{candidate} {feedback_hint}".strip()
            return candidate

        if not evidence_items:
            fallback = f"{topic} {original_question} 综述"
            if review_feedback:
                return f"{fallback} {' '.join(review_feedback[:2])}".strip()
            return fallback

        latest_titles = " ".join(item.title for item in evidence_items[-2:])
        feedback_hint = " ".join(review_feedback[:2]) if review_feedback else ""
        return f"{current_query} {latest_titles} {feedback_hint}".strip()

    def _has_enough_evidence(self, evidence_items: list[EvidenceItem]) -> bool:
        source_types = {item.source_type for item in evidence_items}
        return len(evidence_items) >= 4 and len(source_types) >= 1

    def _summarize_evidence(
        self, question: str, evidence_items: list[EvidenceItem]
    ) -> str:
        if not evidence_items:
            return f"围绕“{question}”暂未检索到足够证据。"

        lines = [f"围绕“{question}”共收集到 {len(evidence_items)} 条证据："]
        for item in evidence_items[:5]:
            excerpt = item.snippet or item.content[:120]
            lines.append(f"- [{item.evidence_id}] {item.title}：{excerpt[:120]}")
        return "\n".join(lines)
