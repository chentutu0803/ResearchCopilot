from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm_client import LLMClient
from core.settings import Settings
from tools.web_search import WebSearchTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单独调试 Web 搜索链路")
    parser.add_argument("--query", required=True, help="搜索查询")
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="最终希望保留给抓取阶段的目标条数",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings()
    llm_client = LLMClient(settings) if settings.has_llm_credentials else None
    tool = WebSearchTool(settings, llm_client=llm_client)

    result = tool.search(args.query, max_results=args.max_results)

    print(f"query={args.query}")
    print(f"搜索候选数={result.metadata.get('candidate_count', 0)}")
    print(f"LLM筛选后URL数={result.metadata.get('llm_selected_count', 0)}")
    print(f"fetch成功数={result.metadata.get('fetch_success_count', 0)}")
    print(f"最终保留证据数={result.total}")
    print("")
    print("搜索尝试：")
    for attempt in result.metadata.get("search_attempts", []):
        print(attempt)
    print("")
    print("Raw candidates 前20条：")
    for index, item in enumerate(result.metadata.get("raw_candidates", [])[:20], start=1):
        print(f"[RAW-{index}] {item.get('title','')} | {item.get('url','')}")
    print("")
    print("Filtered candidates 前20条：")
    for index, item in enumerate(result.metadata.get("filtered_candidates", [])[:20], start=1):
        print(f"[FILTERED-{index}] {item.get('title','')} | {item.get('url','')}")
    print("")
    print("Selected candidates：")
    for index, item in enumerate(result.metadata.get("selected_candidates", []), start=1):
        print(f"[SELECTED-{index}] {item.get('title','')} | {item.get('url','')}")
    print("")
    print("Fetch details：")
    for item in result.metadata.get("fetch_details", []):
        print(item)
    print("")
    print("最终保留结果：")
    for index, item in enumerate(result.items, start=1):
        print(f"[{index}] {item.title}")
        print(f"URL: {item.source_url}")
        print(f"SNIPPET: {item.snippet[:220]}")
        print(f"CONTENT_HEAD: {item.content[:220]}")
        print("")


if __name__ == "__main__":
    main()
