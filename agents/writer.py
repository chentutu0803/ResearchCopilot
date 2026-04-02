from __future__ import annotations

from itertools import chain

from core.llm_client import LLMClient
from core.schemas import EvidenceItem, FinalReport, ResearchSection, SubQuestion
from core.tracer import TraceCollector
from tools.citation_checker import CitationChecker


class WriterAgent:
    def __init__(
        self, llm_client: LLMClient | None = None, tracer: TraceCollector | None = None
    ) -> None:
        self.llm_client = llm_client
        self.tracer = tracer
        self.citation_checker = CitationChecker()

    def write(
        self,
        topic: str,
        sub_questions: list[SubQuestion],
        sections: list[ResearchSection],
        report_language: str = "中文",
        review_feedback: list[str] | None = None,
    ) -> FinalReport:
        references = self._collect_unique_references(sections)
        valid_ids = {item.evidence_id for item in references}
        content = (
            self._write_with_llm(
                topic,
                sub_questions,
                sections,
                report_language,
                review_feedback=review_feedback,
            )
            if self.llm_client
            else ""
        )
        ok = False
        if content:
            ok, _ = self.citation_checker.validate(content, valid_ids)
        if not ok:
            content = self._write_with_template(
                topic,
                sections,
                report_language,
                review_feedback=review_feedback,
            )

        if self.tracer is not None:
            self.tracer.add_event(
                event_type="agent",
                step_name="agent.writer",
                status="success",
                input_summary=topic,
                output_summary=f"生成报告正文，引用数 {len(references)}",
            )

        return FinalReport(
            topic=topic,
            outline=[item.question for item in sub_questions],
            content=content,
            references=references,
        )

    def _write_with_llm(
        self,
        topic: str,
        sub_questions: list[SubQuestion],
        sections: list[ResearchSection],
        report_language: str,
        review_feedback: list[str] | None = None,
    ) -> str:
        assert self.llm_client is not None
        evidence_lines: list[str] = []
        for section in sections:
            evidence_lines.append(f"### 子问题：{section.sub_question}")
            for item in section.evidence_items:
                text = item.content[:220] if item.content else item.snippet
                evidence_lines.append(
                    f"[{item.evidence_id}] 标题：{item.title}\n来源：{item.source_url}\n内容：{text}"
                )

        feedback_block = (
            "\n".join(f"- {item}" for item in (review_feedback or [])) or "无"
        )
        prompt = f"""
请基于下面的证据，为主题“{topic}”生成一篇结构化{report_language}研究报告。

要求：
1. 只允许使用提供的证据，不要编造没有证据支持的结论。
2. 每个关键判断后面必须用方括号标注证据编号，例如 [KB-01-001]。
3. 使用 Markdown 二级标题组织内容。
4. 至少覆盖这些子问题：{", ".join(item.question for item in sub_questions)}。

如果有审查反馈，请优先修复这些问题：
{feedback_block}

证据如下：
{"\n\n".join(evidence_lines)}
""".strip()

        try:
            return self.llm_client.generate_text(
                prompt=prompt, temperature=0.2, max_tokens=2200
            )
        except Exception:
            return ""

    def _write_with_template(
        self,
        topic: str,
        sections: list[ResearchSection],
        report_language: str,
        review_feedback: list[str] | None = None,
    ) -> str:
        if report_language == "English":
            lines = [
                f"# Research Report: {topic}",
                "",
                "## Summary",
                f"This report studies '{topic}' based on collected evidence.",
                "",
            ]
        else:
            lines = [
                f"# {topic} 研究报告",
                "",
                "## 摘要",
                f"本文围绕“{topic}”展开，基于检索到的证据进行结构化总结。",
                "",
            ]

        if review_feedback:
            lines.append(
                "## 本轮修订重点"
                if report_language != "English"
                else "## Revision Focus"
            )
            for item in review_feedback:
                lines.append(f"- {item}")
            lines.append("")

        for section in sections:
            lines.append(f"## {section.sub_question}")
            if not section.evidence_items:
                lines.append(
                    "There is not enough evidence for this section."
                    if report_language == "English"
                    else "当前没有足够证据支持这一部分的详细结论。"
                )
                lines.append("")
                continue
            lines.append(section.summary)
            for item in section.evidence_items[:3]:
                detail = item.snippet or item.content[:150]
                if report_language == "English":
                    lines.append(
                        f"- According to [{item.evidence_id}], {detail[:140]}."
                    )
                else:
                    lines.append(f"- 根据 [{item.evidence_id}]，{detail[:140]}。")
            lines.append("")

        return "\n".join(lines).strip()

    def _collect_unique_references(
        self, sections: list[ResearchSection]
    ) -> list[EvidenceItem]:
        seen: set[str] = set()
        references: list[EvidenceItem] = []
        for item in chain.from_iterable(section.evidence_items for section in sections):
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            references.append(item)
        return references
