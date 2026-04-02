from agents.reviewer import ReviewerAgent
from core.schemas import EvidenceItem, ResearchSection, SubQuestion


def test_reviewer_returns_structured_feedback() -> None:
    reviewer = ReviewerAgent()
    draft = "## 问题一\n这里先回答问题一，并引用 [WEB-1]。"
    valid_ids = {"WEB-1"}
    sub_questions = [
        SubQuestion(index=1, question="问题一"),
        SubQuestion(index=2, question="问题二"),
    ]
    sections = [
        ResearchSection(
            sub_question="问题一",
            evidence_items=[
                EvidenceItem(
                    evidence_id="WEB-1",
                    title="论坛讨论",
                    source_url="https://github.com/openai/codex/issues/123",
                    source_type="web",
                    snippet="弱来源",
                    content="弱来源",
                )
            ],
            draft_section="这里先回答问题一，并引用 [WEB-1]。",
        ),
        ResearchSection(sub_question="问题二", evidence_items=[]),
    ]
    references = sections[0].evidence_items

    review = reviewer.review(
        draft,
        valid_ids,
        sub_questions=sub_questions,
        sections=sections,
        references=references,
    )

    assert "问题二" in review.feedback.missing_subquestions
    assert "问题一" in review.feedback.weak_sections
    assert review.feedback.low_quality_sources
    assert "问题一" in review.feedback.subquestion_source_map
