from core.settings import Settings
from tools.web_search import WebSearchTool


class DummyLLMClient:
    def __init__(self, selected_ids: list[int]) -> None:
        self.selected_ids = selected_ids

    def generate_json(self, **kwargs):  # type: ignore[no-untyped-def]
        return {"selected_ids": self.selected_ids}


def test_web_search_filters_blacklisted_domains() -> None:
    tool = WebSearchTool(Settings())
    items = [
        {"title": "知乎", "url": "https://www.zhihu.com/question/123", "content": ""},
        {"title": "CSDN", "url": "https://blog.csdn.net/a/b", "content": ""},
        {"title": "GitHub", "url": "https://github.com/openai/codex", "content": ""},
    ]

    filtered = tool._filter_search_results(items)

    assert [item["url"] for item in filtered] == ["https://github.com/openai/codex"]


def test_pre_rank_prefers_release_over_issue() -> None:
    tool = WebSearchTool(Settings())
    items = [
        {
            "title": "openai/codex issue",
            "url": "https://github.com/openai/codex/issues/123",
            "content": "issue detail",
        },
        {
            "title": "Releases · openai/codex",
            "url": "https://github.com/openai/codex/releases",
            "content": "release notes",
        },
    ]

    ranked = tool._pre_rank_candidates(items)

    assert ranked[0]["url"] == "https://github.com/openai/codex/releases"


def test_llm_rerank_returns_selected_order() -> None:
    tool = WebSearchTool(Settings(), llm_client=DummyLLMClient([2, 1, 3]))
    items = [
        {"title": "媒体报道", "url": "https://www.ithome.com/a", "content": "news"},
        {"title": "OpenAI Codex", "url": "https://github.com/openai/codex", "content": "repo"},
        {"title": "Docs", "url": "https://docs.example.com/page", "content": "docs"},
    ]

    ranked = tool._rank_search_results(
        query="OpenAI GPT-5.4 的定位与能力变化",
        items=items * 4,
        max_results=2,
    )

    assert ranked[0]["url"] == "https://github.com/openai/codex"
    assert ranked[1]["url"] == "https://www.ithome.com/a"
    assert len(ranked) >= 2
