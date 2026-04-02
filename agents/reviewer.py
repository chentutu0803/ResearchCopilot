from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from core.llm_client import LLMClient
from core.schemas import (
    DraftReview,
    EvidenceItem,
    ResearchSection,
    ReviewFeedback,
    ReviewScore,
    SubQuestion,
)
from core.tracer import TraceCollector
from tools.citation_checker import CitationChecker


class ReviewerAgent:
    def __init__(
        self,
        tracer: TraceCollector | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.tracer = tracer
        self.llm_client = llm_client
        self.citation_checker = CitationChecker()

    def review(
        self,
        draft: str,
        valid_ids: set[str],
        sub_questions: list[SubQuestion] | None = None,
        sections: list[ResearchSection] | None = None,
        references: list[EvidenceItem] | None = None,
        minimum_total: int = 12,
    ) -> DraftReview:
        sub_questions = sub_questions or []
        sections = sections or []
        references = references or []

        feature_bundle = self._build_feature_bundle(
            draft=draft,
            valid_ids=valid_ids,
            sub_questions=sub_questions,
            sections=sections,
            references=references,
        )
        review = self._review_with_llm(
            draft=draft,
            sub_questions=sub_questions,
            feature_bundle=feature_bundle,
            minimum_total=minimum_total,
        )
        if review is None:
            review = self._review_with_rules(
                feature_bundle=feature_bundle,
                minimum_total=minimum_total,
            )

        review = self._apply_hard_constraints(
            review=review,
            feature_bundle=feature_bundle,
            minimum_total=minimum_total,
        )

        if self.tracer is not None:
            self.tracer.add_event(
                event_type="agent",
                step_name="agent.reviewer",
                status="success",
                input_summary=f"draft_len={len(draft)}",
                output_summary=f"score={review.score.total}, accepted={review.accepted}",
                metadata={
                    "citation_count": feature_bundle["citation_count"],
                    "invalid_ids": feature_bundle["invalid_ids"],
                    "covered_subquestions": feature_bundle["covered_subquestions"],
                    "subquestion_count": len(sub_questions),
                    "weak_candidate_count": len(feature_bundle["weak_candidates"]),
                    "llm_review_used": self.llm_client is not None,
                },
            )

        return review

    def _build_feature_bundle(
        self,
        *,
        draft: str,
        valid_ids: set[str],
        sub_questions: list[SubQuestion],
        sections: list[ResearchSection],
        references: list[EvidenceItem],
    ) -> dict[str, object]:
        citations = self.citation_checker.extract_citations(draft)
        citation_set = set(citations)
        has_valid_citations, invalid_ids = self.citation_checker.validate(draft, valid_ids)
        covered_subquestions, missing_candidates = self._evaluate_subquestion_coverage(
            draft=draft,
            sub_questions=sub_questions,
            sections=sections,
        )

        subquestion_details = self._build_subquestion_details(
            citations=citation_set,
            sub_questions=sub_questions,
            sections=sections,
        )
        weak_candidates = [
            question
            for question, detail in subquestion_details.items()
            if detail["rule_signals"]
        ]

        global_quality_summary = self._quality_breakdown(references, citation_set)
        low_quality_sources = self._collect_low_quality_sources(references)

        return {
            "citations": citations,
            "citation_count": len(citations),
            "has_valid_citations": has_valid_citations,
            "invalid_ids": invalid_ids,
            "draft_length": len(draft),
            "has_markdown_headings": "##" in draft,
            "covered_subquestions": covered_subquestions,
            "missing_candidates": missing_candidates,
            "weak_candidates": weak_candidates,
            "subquestion_details": subquestion_details,
            "source_quality_summary": global_quality_summary,
            "low_quality_sources": low_quality_sources,
        }

    def _build_subquestion_details(
        self,
        *,
        citations: set[str],
        sub_questions: list[SubQuestion],
        sections: list[ResearchSection],
    ) -> dict[str, dict[str, object]]:
        details: dict[str, dict[str, object]] = {}
        section_map = {section.sub_question: section for section in sections}
        for item in sub_questions:
            section = section_map.get(item.question)
            details[item.question] = self._build_single_subquestion_detail(
                question=item.question,
                section=section,
                citations=citations,
            )

        for section in sections:
            if section.sub_question not in details:
                details[section.sub_question] = self._build_single_subquestion_detail(
                    question=section.sub_question,
                    section=section,
                    citations=citations,
                )
        return details

    def _build_single_subquestion_detail(
        self,
        *,
        question: str,
        section: ResearchSection | None,
        citations: set[str],
    ) -> dict[str, object]:
        evidence_items = list(section.evidence_items) if section is not None else []
        section_citations = (
            set(self.citation_checker.extract_citations(section.draft_section))
            if section is not None and section.draft_section
            else set()
        )
        effective_citations = (
            section_citations
            if section_citations
            else {item.evidence_id for item in evidence_items if item.evidence_id in citations}
        )

        source_records: list[dict[str, object]] = []
        for evidence in evidence_items:
            quality_label, quality_reason = self._classify_source_quality(evidence)
            source_records.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "source_type": evidence.source_type,
                    "title": evidence.title,
                    "source_url": evidence.source_url,
                    "quality_label": quality_label,
                    "quality_reason": quality_reason,
                    "cited_in_draft": evidence.evidence_id in citations,
                    "cited_in_subquestion": evidence.evidence_id in effective_citations,
                }
            )

        quality_summary = self._quality_breakdown(evidence_items, effective_citations)
        cited_count = len(effective_citations)
        rule_signals: list[str] = []
        if evidence_items and len(evidence_items) < 4:
            rule_signals.append("retrieved_evidence_below_target")
        if evidence_items and cited_count == 0:
            rule_signals.append("draft_not_using_section_evidence")
        if cited_count == 1:
            rule_signals.append("draft_relies_on_single_citation")
        if cited_count > 0 and quality_summary["cited_high"] == 0 and quality_summary["cited_medium"] == 0:
            rule_signals.append("draft_relies_on_low_quality_citations")
        if evidence_items and quality_summary["all_high"] == 0 and quality_summary["all_medium"] == 0:
            rule_signals.append("retrieved_sources_all_low_quality")
        if self._has_low_source_diversity(evidence_items):
            rule_signals.append("source_diversity_low")

        return {
            "question": question,
            "query_history": list(section.query_history) if section is not None else [],
            "retrieved_evidence_count": len(evidence_items),
            "cited_evidence_count": cited_count,
            "quality_summary": quality_summary,
            "rule_signals": rule_signals,
            "source_map": source_records,
        }

    def _review_with_llm(
        self,
        *,
        draft: str,
        sub_questions: list[SubQuestion],
        feature_bundle: dict[str, object],
        minimum_total: int,
    ) -> DraftReview | None:
        if self.llm_client is None:
            return None

        prompt = self._build_llm_prompt(
            draft=draft,
            sub_questions=sub_questions,
            feature_bundle=feature_bundle,
            minimum_total=minimum_total,
        )
        try:
            payload = self.llm_client.generate_json(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1800,
            )
        except Exception:
            return None

        allowed_questions = [item.question for item in sub_questions]
        source_candidates = feature_bundle["low_quality_sources"]
        score = self._parse_score(payload.get("score"))
        suggestions = self._normalize_string_list(payload.get("suggestions", []), limit=8)
        feedback = ReviewFeedback(
            missing_subquestions=self._normalize_question_list(
                payload.get("missing_subquestions", []),
                allowed_questions,
            ),
            weak_sections=self._normalize_question_list(
                payload.get("weak_sections", []),
                allowed_questions,
            ),
            low_quality_sources=self._normalize_source_list(
                payload.get("low_quality_sources", []),
                source_candidates if isinstance(source_candidates, list) else [],
            ),
            subquestion_source_map=self._extract_subquestion_source_map(feature_bundle),
        )
        accepted = bool(payload.get("accepted", False))

        return DraftReview(
            accepted=accepted,
            score=score,
            suggestions=suggestions,
            feedback=feedback,
        )

    def _review_with_rules(
        self,
        *,
        feature_bundle: dict[str, object],
        minimum_total: int,
    ) -> DraftReview:
        citation_count = int(feature_bundle["citation_count"])
        invalid_ids = feature_bundle["invalid_ids"]
        missing_candidates = feature_bundle["missing_candidates"]
        weak_candidates = feature_bundle["weak_candidates"]
        source_quality_summary = feature_bundle["source_quality_summary"]

        factual_support = 1
        if citation_count >= 2:
            factual_support = 3
        if citation_count >= 4:
            factual_support = 4
        if (
            citation_count >= 4
            and not weak_candidates
            and source_quality_summary["all_high"] + source_quality_summary["all_medium"] > 0
        ):
            factual_support = 5

        if citation_count == 0:
            citation_coverage = 0
        elif invalid_ids:
            citation_coverage = 2
        elif citation_count >= 4:
            citation_coverage = 5
        else:
            citation_coverage = 4

        coherence = 2
        if feature_bundle["has_markdown_headings"]:
            coherence = 3
        if citation_count >= 2 and int(feature_bundle["draft_length"]) >= 200:
            coherence = 4
        if (
            citation_count >= 4
            and int(feature_bundle["draft_length"]) >= 400
            and feature_bundle["has_markdown_headings"]
        ):
            coherence = 5

        covered_subquestions = int(feature_bundle["covered_subquestions"])
        total_subquestions = len(feature_bundle["subquestion_details"])
        completeness = 2
        if total_subquestions == 0:
            completeness = 3
        elif covered_subquestions == total_subquestions and not missing_candidates:
            completeness = 5
        elif covered_subquestions >= max(1, total_subquestions - 1):
            completeness = 4

        score = ReviewScore(
            factual_support=factual_support,
            citation_coverage=citation_coverage,
            coherence=coherence,
            completeness=completeness,
        )
        feedback = ReviewFeedback(
            missing_subquestions=list(missing_candidates),
            weak_sections=list(weak_candidates),
            low_quality_sources=list(feature_bundle["low_quality_sources"])[:6],
            subquestion_source_map=self._extract_subquestion_source_map(feature_bundle),
        )
        suggestions = self._build_rule_suggestions(feature_bundle)
        accepted = (
            score.total >= minimum_total
            and citation_count > 0
            and not invalid_ids
            and not feedback.missing_subquestions
            and not feedback.weak_sections
        )
        return DraftReview(
            accepted=accepted,
            score=score,
            suggestions=suggestions,
            feedback=feedback,
        )

    def _apply_hard_constraints(
        self,
        *,
        review: DraftReview,
        feature_bundle: dict[str, object],
        minimum_total: int,
    ) -> DraftReview:
        citation_count = int(feature_bundle["citation_count"])
        invalid_ids = list(feature_bundle["invalid_ids"])
        missing_candidates = list(feature_bundle["missing_candidates"])
        weak_candidates = list(feature_bundle["weak_candidates"])
        low_quality_sources = list(feature_bundle["low_quality_sources"])

        review.feedback.subquestion_source_map = self._extract_subquestion_source_map(
            feature_bundle
        )
        review.feedback.missing_subquestions = self._merge_unique(
            review.feedback.missing_subquestions,
            missing_candidates,
        )
        review.feedback.weak_sections = self._merge_unique(
            review.feedback.weak_sections,
            weak_candidates,
        )
        review.feedback.low_quality_sources = self._merge_unique(
            review.feedback.low_quality_sources,
            low_quality_sources,
        )[:6]

        hard_suggestions = self._build_rule_suggestions(feature_bundle)
        review.suggestions = self._merge_unique(hard_suggestions, review.suggestions)[:8]
        review.accepted = (
            review.accepted
            and review.score.total >= minimum_total
            and citation_count > 0
            and not invalid_ids
            and not review.feedback.missing_subquestions
            and not review.feedback.weak_sections
        )
        return review

    def _build_llm_prompt(
        self,
        *,
        draft: str,
        sub_questions: list[SubQuestion],
        feature_bundle: dict[str, object],
        minimum_total: int,
    ) -> str:
        summary = {
            "citation_count": feature_bundle["citation_count"],
            "invalid_ids": feature_bundle["invalid_ids"],
            "covered_subquestions": feature_bundle["covered_subquestions"],
            "missing_candidates": feature_bundle["missing_candidates"],
            "weak_candidates": feature_bundle["weak_candidates"],
            "source_quality_summary": feature_bundle["source_quality_summary"],
            "subquestion_details": feature_bundle["subquestion_details"],
        }
        return f"""
你是研究型 Agent Workflow 的 Reviewer。你现在要做的是一次统一审查，而不是“规则审一次，再让 LLM 补一句”。

下面给你的内容有两部分：
1. 报告正文。
2. 规则层先抽取出来的结构化特征，它们只是证据和信号，不是最终结论。

请你把“正文内容”和“结构化特征”一起看，统一输出最终审查结果。

审查重点：
1. 报告是否真的覆盖了所有子问题。
2. 每个子问题对应的报告段落是否真正用了它自己的证据。
3. 引用是否合法，是否存在无效 evidence_id。
4. 每个子问题对应的来源质量是否可靠，是否明显依赖低质量来源。
5. 是否需要退回 Planner / Researcher / Writer 补证据或重写。

硬约束：
1. 如果有无效引用 id，accepted 必须是 false。
2. 如果关键子问题没有被回答，accepted 必须是 false。
3. 如果某个子问题虽然检索到了很多证据，但在草稿里几乎没用到，或者只靠单条弱来源支撑，accepted 应该倾向 false。
4. score 四个维度都用 0-5 的整数。
5. missing_subquestions 和 weak_sections 尽量复用原始子问题文本。
6. low_quality_sources 尽量填写来源 url 或 title。
7. suggestions 要短、直接、可执行。
8. 只有当总分 >= {minimum_total}，且没有明显硬伤时，accepted 才能为 true。

返回 JSON，格式固定如下：
{{
  "accepted": false,
  "score": {{
    "factual_support": 0,
    "citation_coverage": 0,
    "coherence": 0,
    "completeness": 0
  }},
  "missing_subquestions": ["..."],
  "weak_sections": ["..."],
  "low_quality_sources": ["..."],
  "suggestions": ["..."]
}}

原始子问题列表：
{json.dumps([item.question for item in sub_questions], ensure_ascii=False, indent=2)}

规则抽取出来的结构化特征：
{json.dumps(summary, ensure_ascii=False, indent=2)}

当前报告正文：
{draft[:8000]}
""".strip()

    def _build_rule_suggestions(self, feature_bundle: dict[str, object]) -> list[str]:
        suggestions: list[str] = []
        if int(feature_bundle["citation_count"]) == 0:
            suggestions.append("补上关键结论对应的 evidence_id 引用。")
        invalid_ids = list(feature_bundle["invalid_ids"])
        if invalid_ids:
            suggestions.append(f"修正无效引用编号：{', '.join(invalid_ids[:4])}。")
        missing_candidates = list(feature_bundle["missing_candidates"])
        if missing_candidates:
            suggestions.append(
                "优先补写这些子问题："
                + "；".join(missing_candidates[:3])
                + "。"
            )
        weak_candidates = list(feature_bundle["weak_candidates"])
        if weak_candidates:
            suggestions.append(
                "针对这些子问题补证据并重写："
                + "；".join(weak_candidates[:3])
                + "。"
            )
        low_quality_sources = list(feature_bundle["low_quality_sources"])
        if low_quality_sources:
            suggestions.append("尽量补充官方文档、官方仓库或知识库来源，降低弱来源占比。")
        return suggestions

    @staticmethod
    def _extract_subquestion_source_map(
        feature_bundle: dict[str, object],
    ) -> dict[str, list[dict[str, object]]]:
        details = feature_bundle["subquestion_details"]
        if not isinstance(details, dict):
            return {}
        source_map: dict[str, list[dict[str, object]]] = {}
        for question, detail in details.items():
            if not isinstance(question, str) or not isinstance(detail, dict):
                continue
            records = detail.get("source_map", [])
            if isinstance(records, list):
                source_map[question] = records
        return source_map

    def _evaluate_subquestion_coverage(
        self,
        *,
        draft: str,
        sub_questions: list[SubQuestion],
        sections: list[ResearchSection],
    ) -> tuple[int, list[str]]:
        if not sub_questions:
            return 0, []

        normalized_draft = self._normalize_text(draft)
        section_map = {section.sub_question: section for section in sections}
        covered = 0
        missing: list[str] = []
        for item in sub_questions:
            section = section_map.get(item.question)
            has_section = section is not None
            has_evidence = bool(section and section.evidence_items)
            question_hit = item.question in draft
            signal_hits = sum(
                1
                for signal in self._extract_question_signals(item.question)
                if signal and signal in normalized_draft
            )
            if question_hit or (has_section and has_evidence and signal_hits >= 1) or signal_hits >= 2:
                covered += 1
            else:
                missing.append(item.question)
        return covered, missing

    def _extract_question_signals(self, question: str) -> list[str]:
        stopwords = {
            "什么",
            "哪些",
            "如何",
            "为什么",
            "以及",
            "实现",
            "问题",
            "方法",
            "原理",
            "介绍",
            "分析",
            "总结",
        }
        signals: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9_+#.-]+|[\u4e00-\u9fff]{2,}", question):
            normalized = self._normalize_text(token)
            if len(normalized) < 2 or normalized in stopwords or normalized in seen:
                continue
            seen.add(normalized)
            signals.append(normalized)
        return signals[:8]

    def _quality_breakdown(
        self, items: list[EvidenceItem], cited_ids: set[str]
    ) -> dict[str, int]:
        counts = {
            "all_high": 0,
            "all_medium": 0,
            "all_low": 0,
            "cited_high": 0,
            "cited_medium": 0,
            "cited_low": 0,
        }
        for item in items:
            label, _ = self._classify_source_quality(item)
            counts[f"all_{label}"] += 1
            if item.evidence_id in cited_ids:
                counts[f"cited_{label}"] += 1
        return counts

    def _collect_low_quality_sources(self, items: list[EvidenceItem]) -> list[str]:
        low_quality_sources: list[str] = []
        seen: set[str] = set()
        for item in items:
            label, _ = self._classify_source_quality(item)
            if label != "low":
                continue
            value = item.source_url or item.title or item.evidence_id
            if value in seen:
                continue
            seen.add(value)
            low_quality_sources.append(value)
        return low_quality_sources

    def _has_low_source_diversity(self, items: list[EvidenceItem]) -> bool:
        web_domains = set()
        kb_count = 0
        for item in items:
            if item.source_type == "kb":
                kb_count += 1
                continue
            hostname = self._extract_hostname(item.source_url)
            if hostname:
                web_domains.add(hostname)
        return kb_count == 0 and len(items) >= 3 and len(web_domains) <= 1

    def _classify_source_quality(self, item: EvidenceItem) -> tuple[str, str]:
        if item.source_type == "kb":
            return "high", "来自本地知识库，属于预筛过的稳定资料。"

        url = item.source_url.strip().lower()
        if not url:
            return "low", "缺少稳定来源 URL，难以判断出处。"

        hostname = self._extract_hostname(url)
        path = (urlparse(url).path or "").lower()

        community_markers = ("/issues/", "/pull/", "/discussions/", "/forum/", "/answers/")
        if any(marker in path for marker in community_markers):
            return "low", "社区讨论或 issue 页面，参考价值有但噪声较大。"

        if any(token in url for token in ("search?", "/search", "login", "signup", "download")):
            return "low", "导航页、搜索页或登录下载页，不适合作为正文证据。"

        high_quality_domains = {
            "github.com",
            "raw.githubusercontent.com",
            "openai.com",
            "help.openai.com",
            "platform.openai.com",
            "qwenlm.github.io",
            "huggingface.co",
            "docs.vllm.ai",
            "developers.llamaindex.ai",
            "docs.llamaindex.ai",
            "milvus.io",
            "qdrant.tech",
            "api.qdrant.tech",
            "pytorch.org",
            "tensorflow.org",
            "arxiv.org",
            "aclanthology.org",
        }
        medium_quality_domains = {
            "medium.com",
            "towardsdatascience.com",
            "infoq.com",
            "ithome.com",
            "ithome.com.tw",
        }

        if hostname in high_quality_domains or any(
            hostname.endswith(f".{domain}") for domain in high_quality_domains
        ):
            return "high", "官方文档、官方仓库或一手技术资料。"

        if hostname in medium_quality_domains or any(
            hostname.endswith(f".{domain}") for domain in medium_quality_domains
        ):
            return "medium", "技术媒体或经验文章，可辅助参考但不宜单独定结论。"

        return "low", "非官方或来源稳定性较弱，默认按低质量来源处理。"

    @staticmethod
    def _extract_hostname(url: str) -> str:
        try:
            return (urlparse(url).hostname or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[\W_]+", "", text.lower())

    @staticmethod
    def _normalize_string_list(values: object, limit: int = 6) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
            if len(normalized) >= limit:
                break
        return normalized

    def _normalize_question_list(
        self,
        values: object,
        allowed_questions: list[str],
    ) -> list[str]:
        normalized_values = self._normalize_string_list(values)
        if not allowed_questions:
            return normalized_values

        mapping = {
            self._normalize_text(question): question for question in allowed_questions
        }
        resolved: list[str] = []
        seen: set[str] = set()
        for value in normalized_values:
            normalized = self._normalize_text(value)
            target = mapping.get(normalized)
            if target is None:
                for key, question in mapping.items():
                    if normalized and (normalized in key or key in normalized):
                        target = question
                        break
            if target is None:
                continue
            if target in seen:
                continue
            seen.add(target)
            resolved.append(target)
        return resolved

    def _normalize_source_list(
        self,
        values: object,
        allowed_sources: list[str],
    ) -> list[str]:
        normalized_values = self._normalize_string_list(values)
        if not allowed_sources:
            return normalized_values

        resolved: list[str] = []
        seen: set[str] = set()
        for value in normalized_values:
            selected = None
            for source in allowed_sources:
                if value == source or value in source or source in value:
                    selected = source
                    break
            if selected is None:
                continue
            if selected in seen:
                continue
            seen.add(selected)
            resolved.append(selected)
        return resolved

    @staticmethod
    def _parse_score(value: object) -> ReviewScore:
        if not isinstance(value, dict):
            return ReviewScore()

        def parse_int(key: str) -> int:
            raw = value.get(key, 0)
            try:
                score = int(raw)
            except Exception:
                score = 0
            return max(0, min(5, score))

        return ReviewScore(
            factual_support=parse_int("factual_support"),
            citation_coverage=parse_int("citation_coverage"),
            coherence=parse_int("coherence"),
            completeness=parse_int("completeness"),
        )

    @staticmethod
    def _merge_unique(left: list[str], right: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for item in [*left, *right]:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged
