from core.settings import Settings
from tools.knowledge_base import KnowledgeBaseTool


def test_knowledge_base_returns_results() -> None:
    settings = Settings(KB_BACKEND="memory", EMBEDDING_BACKEND="lightweight")
    tool = KnowledgeBaseTool(settings=settings)
    result = tool.retrieve("Hybrid Retrieval 的价值", top_k=3)

    assert result.total >= 1
    assert any(item.source_type == "kb" for item in result.items)


def test_knowledge_base_supports_multiple_retrieval_methods() -> None:
    settings = Settings(KB_BACKEND="memory", EMBEDDING_BACKEND="lightweight")
    tool = KnowledgeBaseTool(settings=settings)

    bm25_result = tool.retrieve_by_method(
        "工作流和多智能体的取舍", method="bm25", top_k=3
    )
    vector_result = tool.retrieve_by_method(
        "工作流和多智能体的取舍", method="vector", top_k=3
    )
    hybrid_result = tool.retrieve_by_method(
        "工作流和多智能体的取舍", method="hybrid", top_k=3
    )

    assert bm25_result.total >= 1
    assert vector_result.total >= 1
    assert hybrid_result.total >= 1
