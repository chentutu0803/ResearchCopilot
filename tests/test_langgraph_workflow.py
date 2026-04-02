from core.langgraph_workflow import LangGraphResearchWorkflow
from core.schemas import DraftReview, EvidenceItem, FinalReport, ResearchSection, ReviewFeedback, ReviewScore, SearchResult, SubQuestion
from core.settings import Settings


def test_langgraph_workflow_replans_when_reviewer_rejects() -> None:
    workflow = LangGraphResearchWorkflow(
        Settings(
            KB_BACKEND="memory",
            EMBEDDING_BACKEND="lightweight",
            MAX_REPLAN_CYCLES=1,
            MODEL_NAME="gpt-5.4",
        )
    )

    planner_calls: list[list[str]] = []
    research_calls: list[str] = []
    review_calls = {"count": 0}

    def fake_plan_subquestions(
        *,
        topic: str,
        max_subquestions: int,
        review_feedback: list[str] | None = None,
        review_feedback_structured: ReviewFeedback | None = None,
        previous_subquestions: list[SubQuestion] | None = None,
    ) -> list[SubQuestion]:
        planner_calls.append(list(review_feedback or []))
        return [
            SubQuestion(index=1, question=f"{topic}-A", rationale="test"),
            SubQuestion(index=2, question=f"{topic}-B", rationale="test"),
        ]

    def fake_research_subquestion(
        *,
        topic: str,
        sub_question: SubQuestion,
        max_rounds: int,
        use_web: bool,
        use_kb: bool,
        allow_query_rewrite: bool,
        review_feedback: list[str] | None = None,
    ) -> ResearchSection:
        research_calls.append(sub_question.question)
        return ResearchSection(
            sub_question=sub_question.question,
            query_history=[sub_question.question],
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

    def fake_writer_node(state):  # type: ignore[no-untyped-def]
        content = (
            "## 初稿\n缺少引用"
            if not state.get("review_feedback")
            else "## 修订稿\n根据 [KB-TEST-001] 完成补充。"
        )
        final_report = FinalReport(
            topic=state["topic"],
            outline=[item.question for item in state.get("sub_questions", [])],
            content=content,
            references=[
                EvidenceItem(
                    evidence_id="KB-TEST-001",
                    title="测试证据",
                    source_url="kb://test",
                    source_type="kb",
                    snippet="测试摘要",
                    content="测试内容",
                )
            ],
        )
        return {**state, "final_report": final_report}

    def fake_review(
        draft: str,
        valid_ids: set[str],
        *,
        sub_questions=None,
        sections=None,
        references=None,
        minimum_total: int = 12,
    ) -> DraftReview:
        review_calls["count"] += 1
        if review_calls["count"] == 1:
            return DraftReview(
                accepted=False,
                score=ReviewScore(
                    factual_support=2,
                    citation_coverage=1,
                    coherence=2,
                    completeness=2,
                ),
                suggestions=["需要补充证据"],
                feedback=ReviewFeedback(
                    missing_subquestions=[],
                    weak_sections=["测试主题-B"],
                    low_quality_sources=["kb://test"],
                ),
            )
        return DraftReview(
            accepted=True,
            score=ReviewScore(
                factual_support=4,
                citation_coverage=4,
                coherence=4,
                completeness=4,
            ),
            suggestions=[],
        )

    workflow._plan_subquestions = fake_plan_subquestions  # type: ignore[method-assign]
    workflow._research_subquestion = fake_research_subquestion  # type: ignore[method-assign]
    workflow._writer_node = fake_writer_node  # type: ignore[method-assign]
    workflow.reviewer.review = fake_review  # type: ignore[method-assign]
    workflow.graph = workflow._build_graph()

    result = workflow.run(
        topic="测试主题",
        use_web=False,
        use_kb=True,
        max_subquestions=1,
        max_search_rounds=1,
        enable_reviewer=True,
    )

    assert review_calls["count"] == 2
    assert planner_calls == [[]]
    assert research_calls == ["测试主题-A", "测试主题-B", "测试主题-B"]
    assert result.final_report.review is not None
    assert result.final_report.review.accepted is True
    assert "[KB-TEST-001]" in result.final_report.content


def test_rewrite_query_prefers_llm_generated_candidates() -> None:
    workflow = LangGraphResearchWorkflow(
        Settings(
            KB_BACKEND="memory",
            EMBEDDING_BACKEND="lightweight",
            MODEL_NAME="gpt-5.4",
        )
    )

    class FakeLLMClient:
        def generate_json(self, *, prompt: str, temperature: float = 0.1, max_tokens: int = 2048):
            assert "Query Rewrite" in prompt
            return {
                "queries": [
                    "vLLM architecture acceleration",
                    "vLLM paged attention continuous batching",
                    "vLLM inference engine overview",
                ]
            }

    workflow.llm_client = FakeLLMClient()  # type: ignore[assignment]

    rewritten = workflow._rewrite_query(
        topic="vLLM",
        original_question="vLLM 为什么能推理加速？",
        current_query="vLLM 为什么能推理加速？",
        evidence_items=[],
        round_index=0,
        review_feedback=None,
    )

    assert rewritten == "vLLM architecture acceleration"


def test_research_subquestion_fans_out_multiple_queries_per_round() -> None:
    workflow = LangGraphResearchWorkflow(
        Settings(
            KB_BACKEND="memory",
            EMBEDDING_BACKEND="lightweight",
            MODEL_NAME="gpt-5.4",
        )
    )

    class FakeLLMClient:
        def generate_json(self, *, prompt: str, temperature: float = 0.1, max_tokens: int = 2048):
            return {
                "queries": [
                    "vLLM architecture overview",
                    "vLLM paged attention",
                    "vLLM continuous batching",
                    "vLLM prefix caching",
                ]
            }

    class FakeWebSearchTool:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search(self, query: str, max_results: int = 10) -> SearchResult:
            self.calls.append(query)
            slug = query.replace(" ", "_")
            return SearchResult(
                query=query,
                items=[
                    EvidenceItem(
                        evidence_id=f"WEB-{len(self.calls)}",
                        title=query,
                        source_url=f"https://example.com/{slug}",
                        source_type="web",
                        snippet=f"snippet {query}",
                        content=f"content {query}",
                    )
                ],
                total=1,
                metadata={"query": query},
            )

    class FakeKnowledgeBaseTool:
        def __init__(self) -> None:
            self.is_ready = True
            self.calls: list[str] = []

        def load(self) -> None:
            self.is_ready = True

        def retrieve(self, query: str, top_k: int = 6) -> SearchResult:
            self.calls.append(query)
            slug = query.replace(" ", "_")
            return SearchResult(
                query=query,
                items=[
                    EvidenceItem(
                        evidence_id=f"KB-{len(self.calls)}",
                        title=f"kb {query}",
                        source_url=f"kb://{slug}",
                        source_type="kb",
                        snippet=f"kb snippet {query}",
                        content=f"kb content {query}",
                    )
                ],
                total=1,
                metadata={"query": query},
            )

    workflow.llm_client = FakeLLMClient()  # type: ignore[assignment]
    workflow.web_search_tool = FakeWebSearchTool()  # type: ignore[assignment]
    workflow.knowledge_base_tool = FakeKnowledgeBaseTool()  # type: ignore[assignment]

    section = workflow._research_subquestion(
        topic="vLLM",
        sub_question=SubQuestion(index=1, question="vLLM 为什么能推理加速？", rationale="test"),
        max_rounds=1,
        use_web=True,
        use_kb=True,
        allow_query_rewrite=True,
        review_feedback=None,
    )

    assert len(section.query_history) == 2
    assert len(workflow.web_search_tool.calls) == 2  # type: ignore[attr-defined]
    assert len(workflow.knowledge_base_tool.calls) == 2  # type: ignore[attr-defined]
    assert section.metadata["round_details"][0]["queries"] == section.query_history
    assert section.metadata["final_evidence_count"] >= 4
    round_events = [
        event for event in workflow.tracer.events if event.step_name == "researcher.round"
    ]
    fanout_events = [
        event
        for event in workflow.tracer.events
        if event.step_name == "researcher.query_fanout"
    ]
    assert len(round_events) == 1
    assert round_events[0].metadata["sub_question"] == "vLLM 为什么能推理加速？"
    assert round_events[0].metadata["round_index"] == 1
    assert round_events[0].metadata["queries"] == section.query_history
    assert round_events[0].metadata["web_query_count"] == 2
    assert round_events[0].metadata["kb_query_count"] == 2
    assert round_events[0].metadata["evidence_count_after_round"] >= 4
    assert len(fanout_events) == 4
    assert {event.metadata["source"] for event in fanout_events} == {"web", "kb"}
    assert all(isinstance(event.metadata["result_count"], int) for event in fanout_events)
