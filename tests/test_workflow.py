from typing import Any

from core.schemas import (
    DraftReview,
    EvidenceItem,
    FinalReport,
    ResearchSection,
    ReviewScore,
    SubQuestion,
)
from core.settings import Settings
from core.workflow import ResearchWorkflow, WorkflowState


def test_workflow_context_initial_state() -> None:
    workflow = ResearchWorkflow(
        Settings(KB_BACKEND="memory", EMBEDDING_BACKEND="lightweight")
    )
    context = workflow.create_context("测试主题")

    assert context.topic == "测试主题"
    assert context.state == WorkflowState.INIT


def test_workflow_runs_end_to_end_with_kb_only() -> None:
    workflow = ResearchWorkflow(
        Settings(KB_BACKEND="memory", EMBEDDING_BACKEND="lightweight")
    )
    result = workflow.run(
        topic="RAG 系统中 hybrid retrieval 的价值", use_web=False, use_kb=True
    )

    assert len(result.sub_questions) >= 1
    assert "## 参考资料" in result.final_report.content
    assert len(result.final_report.references) >= 1


def test_workflow_supports_english_report_and_reviewer_toggle() -> None:
    workflow = ResearchWorkflow(
        Settings(KB_BACKEND="memory", EMBEDDING_BACKEND="lightweight")
    )
    result = workflow.run(
        topic="Agent workflow 和 autonomous agent 的取舍",
        use_web=False,
        use_kb=True,
        report_language="English",
        enable_reviewer=False,
        allow_query_rewrite=False,
    )

    assert result.final_report.review is not None
    assert result.final_report.review.accepted is True
    assert "## 参考资料" in result.final_report.content


def test_workflow_replans_when_review_fails() -> None:
    settings = Settings(
        MAX_REPLAN_CYCLES=1,
        KB_BACKEND="memory",
        EMBEDDING_BACKEND="lightweight",
    )
    workflow = ResearchWorkflow(settings)

    plan_calls: list[list[str]] = []

    def fake_plan(
        topic: str,
        max_subquestions: int = 5,
        review_feedback: list[str] | None = None,
        previous_subquestions: list[SubQuestion] | None = None,
    ) -> list[SubQuestion]:
        plan_calls.append(list(review_feedback or []))
        suffix = "补证据" if review_feedback else "初始"
        return [SubQuestion(index=1, question=f"{topic}-{suffix}", rationale="test")]

    def fake_research(**kwargs: Any) -> ResearchSection:
        sub_question = kwargs["sub_question"]
        question = str(sub_question.question)
        return ResearchSection(
            sub_question=question,
            query_history=[question],
            evidence_items=[
                EvidenceItem(
                    evidence_id="KB-TEST-001",
                    title="测试证据",
                    source_url="kb://test",
                    source_type="kb",
                    snippet="测试摘要",
                    content="测试内容",
                )
            ],
            summary="测试总结",
        )

    def fake_write(
        topic: str,
        sub_questions: list[SubQuestion],
        sections: list[ResearchSection],
        report_language: str = "中文",
        review_feedback: list[str] | None = None,
    ) -> FinalReport:
        content = (
            "## 初稿\n缺少足够引用"
            if not review_feedback
            else "## 重写后报告\n根据 [KB-TEST-001] 补充了关键证据。"
        )
        return FinalReport(
            topic=topic,
            outline=[item.question for item in sub_questions],
            content=content,
            references=sections[0].evidence_items,
        )

    review_calls = {"count": 0}

    def fake_review(
        draft: str,
        valid_ids: set[str],
        sub_questions: list[SubQuestion] | None = None,
        sections: list[ResearchSection] | None = None,
        references: list[EvidenceItem] | None = None,
        minimum_total: int = 12,
    ) -> DraftReview:
        review_calls["count"] += 1
        if review_calls["count"] == 1:
            return DraftReview(
                accepted=False,
                score=ReviewScore(
                    factual_support=2, citation_coverage=1, coherence=2, completeness=2
                ),
                suggestions=["需要补充证据", "需要补充引用"],
            )
        return DraftReview(
            accepted=True,
            score=ReviewScore(
                factual_support=4, citation_coverage=4, coherence=4, completeness=4
            ),
            suggestions=[],
        )

    workflow.planner.plan = fake_plan  # type: ignore[method-assign]
    workflow.researcher.research = fake_research  # type: ignore[method-assign]
    workflow.writer.write = fake_write  # type: ignore[method-assign]
    workflow.reviewer.review = fake_review  # type: ignore[method-assign]

    result = workflow.run(topic="测试主题", use_web=False, use_kb=True)

    assert review_calls["count"] == 2
    assert len(plan_calls) == 2
    assert plan_calls[0] == []
    assert plan_calls[1] == ["需要补充证据", "需要补充引用"]
    assert result.final_report.review is not None
    assert result.final_report.review.accepted is True
    assert "[KB-TEST-001]" in result.final_report.content


def test_workflow_stops_when_max_replan_reached() -> None:
    settings = Settings(
        MAX_REPLAN_CYCLES=1,
        KB_BACKEND="memory",
        EMBEDDING_BACKEND="lightweight",
    )
    workflow = ResearchWorkflow(settings)

    plan_calls = {"count": 0}

    def fake_plan(
        topic: str,
        max_subquestions: int = 5,
        review_feedback: list[str] | None = None,
        previous_subquestions: list[SubQuestion] | None = None,
    ) -> list[SubQuestion]:
        plan_calls["count"] += 1
        suffix = str(plan_calls["count"])
        return [
            SubQuestion(index=1, question=f"{topic}-round-{suffix}", rationale="test")
        ]

    def fake_research(**kwargs: Any) -> ResearchSection:
        sub_question = kwargs["sub_question"]
        question = str(sub_question.question)
        return ResearchSection(
            sub_question=question,
            query_history=[question],
            evidence_items=[
                EvidenceItem(
                    evidence_id="KB-TEST-002",
                    title="测试证据2",
                    source_url="kb://test2",
                    source_type="kb",
                    snippet="测试摘要2",
                    content="测试内容2",
                )
            ],
            summary="测试总结2",
        )

    def fake_write(
        topic: str,
        sub_questions: list[SubQuestion],
        sections: list[ResearchSection],
        report_language: str = "中文",
        review_feedback: list[str] | None = None,
    ) -> FinalReport:
        return FinalReport(
            topic=topic,
            outline=[item.question for item in sub_questions],
            content="## 始终不通过的草稿\n根据 [KB-TEST-002] 写了内容。",
            references=sections[0].evidence_items,
        )

    def fake_review(
        draft: str,
        valid_ids: set[str],
        sub_questions: list[SubQuestion] | None = None,
        sections: list[ResearchSection] | None = None,
        references: list[EvidenceItem] | None = None,
        minimum_total: int = 12,
    ) -> DraftReview:
        return DraftReview(
            accepted=False,
            score=ReviewScore(
                factual_support=2,
                citation_coverage=2,
                coherence=2,
                completeness=2,
            ),
            suggestions=["需要继续补证据"],
        )

    workflow.planner.plan = fake_plan  # type: ignore[method-assign]
    workflow.researcher.research = fake_research  # type: ignore[method-assign]
    workflow.writer.write = fake_write  # type: ignore[method-assign]
    workflow.reviewer.review = fake_review  # type: ignore[method-assign]

    result = workflow.run(topic="停止测试主题", use_web=False, use_kb=True)

    assert plan_calls["count"] == 2
    assert result.final_report.review is not None
    assert result.final_report.review.accepted is False
