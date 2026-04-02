from __future__ import annotations

import sys
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from core.llm_client import LLMClient
from core.schemas import EvidenceItem, SearchResult
from core.settings import Settings
from core.tracer import TraceCollector


class WebSearchTool:
    DOMAIN_BLACKLIST = {
        "zhihu.com",
        "www.zhihu.com",
        "zhuanlan.zhihu.com",
        "csdn.net",
        "www.csdn.net",
        "blog.csdn.net",
        "bilibili.com",
        "www.bilibili.com",
        "apps.microsoft.com",
        "apps.apple.com",
        "sj.qq.com",
        "baike.baidu.com",
        "m.baidu.com",
    }

    POSITIVE_PATH_KEYWORDS = (
        "release",
        "releases",
        "readme",
        "docs",
        "documentation",
        "guide",
        "guides",
        "manual",
        "reference",
        "api",
        "blog",
        "announcement",
        "changelog",
        "news",
    )

    NEGATIVE_PATH_KEYWORDS = (
        "issue",
        "issues",
        "plugin",
        "plugins",
        "forum",
        "discussion",
        "discussions",
        "image",
        "images",
        "download",
        "tag",
        "category",
        "search",
        "login",
        "signup",
        "register",
    )

    def __init__(
        self,
        settings: Settings,
        tracer: TraceCollector | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer
        self.llm_client = llm_client

    def _debug(self, message: str) -> None:
        if self.settings.log_level.upper() != "DEBUG":
            return
        print(f"[DEBUG web_search] {message}", file=sys.stderr, flush=True)

    def search(self, query: str, max_results: int = 10) -> SearchResult:
        started_at = time.perf_counter()
        self._debug(f"start query={query} max_results={max_results}")
        try:
            pipeline = self._run_search_pipeline(query=query, max_results=max_results)
            candidates = pipeline["filtered_candidates"]
            if not candidates:
                self._debug(f"empty query={query}")
                return self._empty_result(started_at, query)

            ranked_candidates = pipeline["selected_candidates"]

            evidence_items: list[EvidenceItem] = []
            fetch_details: list[dict[str, Any]] = []
            for index, item in enumerate(ranked_candidates, start=1):
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                snippet = str(item.get("content", "")).strip()
                raw_content = item.get("raw_content") or item.get("content") or ""
                content = str(raw_content)[:8000]
                if not self._is_usable_page_content(content):
                    fetch_details.append(
                        {
                            "url": url,
                            "title": title,
                            "success": False,
                            "reason": "unusable_content",
                            "content_length": len(content),
                        }
                    )
                    continue
                evidence_id = f"WEB-{index}"
                evidence_items.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        title=title,
                        source_url=url,
                        source_type="web",
                        snippet=snippet[:500],
                        content=content,
                        metadata={
                            "query": query,
                            "tavily_score": item.get("score"),
                            "pre_rank_score": item.get("pre_rank_score"),
                        },
                    )
                )
                fetch_details.append(
                    {
                        "url": url,
                        "title": title,
                        "success": True,
                        "reason": "ok",
                        "content_length": len(content),
                        "evidence_id": evidence_id,
                    }
                )

            metadata = {
                "backend": "tavily",
                "candidate_count": len(pipeline["raw_candidates"]),
                "raw_candidates": pipeline["raw_candidates"],
                "filtered_candidates": pipeline["filtered_candidates"],
                "llm_selected_count": len(ranked_candidates),
                "selected_candidates": ranked_candidates,
                "fetch_success_count": len(evidence_items),
                "fetch_details": fetch_details,
                "max_results": max_results,
                "search_attempts": pipeline["search_attempts"],
            }

            duration_ms = int((time.perf_counter() - started_at) * 1000)
            if self.tracer is not None:
                self.tracer.add_event(
                    event_type="tool",
                    step_name="tool.web_search",
                    status="success",
                    tool_name="tavily-search",
                    input_summary=query,
                    output_summary=f"命中 {len(evidence_items)} 条结果",
                    duration_ms=duration_ms,
                    metadata=metadata | {"result_count": len(evidence_items)},
                )

            self._debug(
                f"done query={query} candidate_count={len(pipeline['raw_candidates'])} "
                f"selected={len(ranked_candidates)} fetched={len(evidence_items)}"
            )
            return SearchResult(
                query=query,
                items=evidence_items,
                total=len(evidence_items),
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            self._debug(f"error query={query} error={exc}")
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            if self.tracer is not None:
                self.tracer.add_event(
                    event_type="tool",
                    step_name="tool.web_search",
                    status="error",
                    tool_name="tavily-search",
                    input_summary=query,
                    output_summary=str(exc),
                    duration_ms=duration_ms,
                )
            return SearchResult(
                query=query,
                items=[],
                total=0,
                metadata={
                    "error": str(exc),
                    "backend": "tavily",
                    "candidate_count": 0,
                    "raw_candidates": [],
                    "filtered_candidates": [],
                    "llm_selected_count": 0,
                    "selected_candidates": [],
                    "fetch_success_count": 0,
                    "fetch_details": [],
                    "search_attempts": [],
                },
            )

    def _run_search_pipeline(
        self, *, query: str, max_results: int
    ) -> dict[str, Any]:
        if not self.settings.tavily_api_key.strip():
            raise RuntimeError("未配置 TAVILY_API_KEY，无法调用 Tavily Search。")

        candidate_limit = max(40, max_results * 8)
        attempts: list[dict[str, Any]] = []
        candidates = self._call_tavily_search(query=query, max_results=candidate_limit)
        filtered = self._filter_search_results(candidates)
        attempts.append(
            {
                "query": query,
                "raw_count": len(candidates),
                "filtered_count": len(filtered),
            }
        )
        pre_ranked = self._pre_rank_candidates(filtered)[:candidate_limit]
        selected = self._rank_search_results(
            query=query,
            items=pre_ranked,
            max_results=max_results,
        )
        return {
            "raw_candidates": candidates,
            "filtered_candidates": filtered,
            "selected_candidates": selected,
            "search_attempts": attempts,
        }

    def _call_tavily_search(
        self,
        *,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            "topic": "general",
            "max_results": min(max_results, 20),
            "include_answer": False,
            "include_raw_content": "markdown",
            "exclude_domains": sorted(self.DOMAIN_BLACKLIST),
        }

        with httpx.Client(
            timeout=self.settings.web_search_timeout_seconds,
            follow_redirects=True,
        ) as client:
            response = client.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {self.settings.tavily_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        return [item for item in results if isinstance(item, dict)]

    def _filter_search_results(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in items:
            url = str(item.get("url", "")).strip()
            if not url or self._is_blacklisted_url(url):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            filtered.append(item)
        return filtered

    def _pre_rank_candidates(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rescored: list[tuple[int, dict[str, Any]]] = []
        for item in items:
            score = self._candidate_priority_score(item)
            updated = dict(item)
            updated["pre_rank_score"] = score
            rescored.append((score, updated))
        rescored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in rescored]

    def _candidate_priority_score(self, item: dict[str, Any]) -> int:
        url = str(item.get("url", "")).strip().lower()
        title = str(item.get("title", "")).strip().lower()
        snippet = str(item.get("content", "")).strip().lower()
        combined = f"{title}\n{url}\n{snippet}"

        score = 0
        if "github.com/" in url:
            score += 30
        if "github.com/openai/" in url:
            score += 40
        if "raw.githubusercontent.com/openai/" in url:
            score += 40
        if any(keyword in combined for keyword in self.POSITIVE_PATH_KEYWORDS):
            score += 25
        if any(keyword in combined for keyword in self.NEGATIVE_PATH_KEYWORDS):
            score -= 35
        if "/releases" in url:
            score += 60
        if "/readme" in url or "readme" in combined:
            score += 45
        if self._is_repo_root(url):
            score += 25
        if "/issues/" in url:
            score -= 70
        return score

    def _rank_search_results(
        self, *, query: str, items: list[dict[str, Any]], max_results: int
    ) -> list[dict[str, Any]]:
        target_results = max(10, max_results)
        if len(items) <= target_results:
            return items
        if self.llm_client is None:
            return items[:target_results]

        candidate_lines = []
        for index, item in enumerate(items, start=1):
            candidate_lines.append(
                f"{index}. title={item.get('title','')} | url={item.get('url','')} | snippet={item.get('content','')}"
            )

        prompt = (
            "你是搜索结果筛选器。请根据用户查询，从候选网页中删掉不相关结果，保留所有相关结果，并按相关性从高到低排序。"
            "第一优先：release / releases / official docs / repo README / official announcement / changelog / API reference。"
            "第二优先：官方仓库主页、官方博客、主流媒体直接报道。"
            "低优先级：issue、plugin、forum、image result、download、聚合页。"
            "只返回 JSON。格式：{\"selected_ids\":[1,2,3]}。selected_ids 表示按相关性排序后的所有相关结果编号，不要额外解释。\n"
            f"用户查询：{query}\n"
            "请保留所有真正相关的结果，不要为了凑数保留无关页面。\n"
            "候选结果如下：\n"
            + "\n".join(candidate_lines)
        )

        try:
            payload = self.llm_client.generate_json(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1200,
            )
        except Exception:
            return items[:target_results]

        selected_ids = payload.get("selected_ids", [])
        if not isinstance(selected_ids, list):
            return items[:target_results]

        ranked: list[dict[str, Any]] = []
        seen: set[int] = set()
        for value in selected_ids:
            if not isinstance(value, int):
                continue
            if value < 1 or value > len(items) or value in seen:
                continue
            seen.add(value)
            ranked.append(items[value - 1])
            if len(ranked) >= target_results:
                break
        return ranked or items[:target_results]

    def _is_blacklisted_url(self, url: str) -> bool:
        try:
            hostname = (urlparse(url).hostname or "").lower()
        except Exception:  # noqa: BLE001
            return True
        return any(
            hostname == blocked or hostname.endswith(f".{blocked}")
            for blocked in self.DOMAIN_BLACKLIST
        )

    @staticmethod
    def _is_repo_root(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != "github.com":
            return False
        parts = [part for part in parsed.path.split("/") if part]
        return len(parts) == 2

    @staticmethod
    def _is_usable_page_content(content: str) -> bool:
        stripped = content.strip()
        if not stripped:
            return False
        if stripped.startswith("<!-- Error fetching"):
            return False
        if stripped.startswith("403 Forbidden"):
            return False
        return True

    def _empty_result(self, started_at: float, query: str) -> SearchResult:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        if self.tracer is not None:
            self.tracer.add_event(
                event_type="tool",
                step_name="tool.web_search",
                status="success",
                tool_name="tavily-search",
                input_summary=query,
                output_summary="搜索结果为空",
                duration_ms=duration_ms,
                metadata={"result_count": 0},
            )
        return SearchResult(
            query=query,
            items=[],
            total=0,
            metadata={"empty": True, "backend": "tavily"},
        )
