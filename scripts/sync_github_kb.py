from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RAW_BASE = "https://raw.githubusercontent.com"
USER_AGENT = "ResearchCopilot-KB-Sync/1.0"


@dataclass(frozen=True)
class GitHubDocSource:
    repo: str
    paths: tuple[str, ...]
    title: str
    category: str
    output_slug: str
    branches: tuple[str, ...] = ("main", "master")


SOURCES: tuple[GitHubDocSource, ...] = (
    GitHubDocSource(
        repo="langchain-ai/langgraph",
        paths=("README.md",),
        title="LangGraph Overview",
        category="agent_workflow",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="deepset-ai/haystack",
        paths=("README.md",),
        title="Haystack RAG Framework",
        category="rag_framework",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="openai/openai-cookbook",
        paths=("README.md",),
        title="OpenAI Cookbook",
        category="llm_application",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="kevwan/rag-agent",
        paths=("README.md", "readme.md"),
        title="RAG Agent",
        category="agent_rag",
        output_slug="overview",
        branches=("main", "master"),
    ),
    GitHubDocSource(
        repo="castorini/pyserini",
        paths=("README.md",),
        title="Pyserini Retrieval Toolkit",
        category="retrieval",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="FlagOpen/FlagEmbedding",
        paths=("README.md",),
        title="FlagEmbedding",
        category="embedding_reranker",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="stanford-futuredata/ColBERT",
        paths=("README.md",),
        title="ColBERT",
        category="retrieval",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="qdrant/qdrant",
        paths=("README.md",),
        title="Qdrant Overview",
        category="vector_database",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="qdrant/qdrant",
        paths=("docs/QUICK_START.md",),
        title="Qdrant Quick Start",
        category="vector_database",
        output_slug="quick_start",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("README.md",),
        title="vLLM Overview",
        category="inference_engine",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/serving/openai_compatible_server.md",),
        title="vLLM OpenAI Compatible Server",
        category="inference_engine",
        output_slug="openai_compatible_server",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/serving/offline_inference.md",),
        title="vLLM Offline Inference",
        category="inference_engine",
        output_slug="offline_inference",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/serving/parallelism_scaling.md",),
        title="vLLM Parallelism Scaling",
        category="inference_engine",
        output_slug="parallelism_scaling",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/configuration/optimization.md",),
        title="vLLM Optimization Guide",
        category="inference_engine",
        output_slug="optimization",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/design/prefix_caching.md",),
        title="vLLM Prefix Caching Design",
        category="inference_engine",
        output_slug="prefix_caching_design",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/features/automatic_prefix_caching.md",),
        title="vLLM Automatic Prefix Caching",
        category="inference_engine",
        output_slug="automatic_prefix_caching",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/deployment/frameworks/retrieval_augmented_generation.md",),
        title="vLLM Retrieval Augmented Generation",
        category="inference_engine",
        output_slug="retrieval_augmented_generation",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/serving/integrations/langchain.md",),
        title="vLLM LangChain Integration",
        category="inference_engine",
        output_slug="langchain_integration",
    ),
    GitHubDocSource(
        repo="vllm-project/vllm",
        paths=("docs/serving/integrations/llamaindex.md",),
        title="vLLM LlamaIndex Integration",
        category="inference_engine",
        output_slug="llamaindex_integration",
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/userGuide/search-query-get/multi-vector-search.md",),
        title="Milvus Multi Vector Search",
        category="vector_database",
        output_slug="multi_vector_search",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/userGuide/search-query-get/full-text-search.md",),
        title="Milvus Full Text Search",
        category="vector_database",
        output_slug="full_text_search",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/userGuide/search-query-get/filtered-search.md",),
        title="Milvus Filtered Search",
        category="vector_database",
        output_slug="filtered_search",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/tutorials/hybrid_search_with_milvus.md",),
        title="Milvus Hybrid Search Tutorial",
        category="vector_database",
        output_slug="hybrid_search_tutorial",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/tutorials/build-rag-with-milvus.md",),
        title="Build RAG With Milvus",
        category="vector_database",
        output_slug="build_rag",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/tutorials/contextual_retrieval_with_milvus.md",),
        title="Contextual Retrieval With Milvus",
        category="vector_database",
        output_slug="contextual_retrieval",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/integrations/langchain/milvus_hybrid_search_retriever.md",),
        title="Milvus LangChain Hybrid Search Retriever",
        category="vector_database",
        output_slug="langchain_hybrid_search_retriever",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/integrations/llamaindex_milvus_hybrid_search.md",),
        title="Milvus LlamaIndex Hybrid Search",
        category="vector_database",
        output_slug="llamaindex_hybrid_search",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="milvus-io/milvus-docs",
        paths=("site/en/integrations/milvus_rag_with_vllm.md",),
        title="Milvus RAG With vLLM",
        category="vector_database",
        output_slug="rag_with_vllm",
        branches=("v2.6.x",),
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("README.md",),
        title="LlamaIndex Overview",
        category="rag_framework",
        output_slug="overview",
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("docs/src/content/docs/framework/getting_started/starter_example.mdx",),
        title="LlamaIndex Starter Example",
        category="rag_framework",
        output_slug="starter_example",
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("docs/src/content/docs/framework/module_guides/indexing/vector_store_index.mdx",),
        title="LlamaIndex Vector Store Index",
        category="rag_framework",
        output_slug="vector_store_index",
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("docs/src/content/docs/framework/module_guides/storing/vector_stores.md",),
        title="LlamaIndex Vector Stores",
        category="rag_framework",
        output_slug="vector_stores",
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("docs/src/content/docs/framework/optimizing/building_rag_from_scratch.md",),
        title="LlamaIndex Building RAG From Scratch",
        category="rag_framework",
        output_slug="building_rag_from_scratch",
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("docs/src/content/docs/framework/optimizing/production_rag.md",),
        title="LlamaIndex Production RAG",
        category="rag_framework",
        output_slug="production_rag",
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("docs/src/content/docs/framework/optimizing/advanced_retrieval/advanced_retrieval.md",),
        title="LlamaIndex Advanced Retrieval",
        category="rag_framework",
        output_slug="advanced_retrieval",
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("docs/src/content/docs/framework/optimizing/advanced_retrieval/query_transformations.md",),
        title="LlamaIndex Query Transformations",
        category="rag_framework",
        output_slug="query_transformations",
    ),
    GitHubDocSource(
        repo="run-llama/llama_index",
        paths=("docs/src/content/docs/framework/optimizing/agentic_strategies/agentic_strategies.md",),
        title="LlamaIndex Agentic Strategies",
        category="rag_framework",
        output_slug="agentic_strategies",
    ),
)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _fetch_text(url: str) -> str | None:
    for attempt in range(3):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=12) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                return None
            time.sleep(1.0 + attempt)
    return None


def _resolve_source(source: GitHubDocSource) -> tuple[str, str] | None:
    for branch in source.branches:
        for relative_path in source.paths:
            url = f"{RAW_BASE}/{source.repo}/{branch}/{relative_path}"
            content = _fetch_text(url)
            if content:
                return url, content
    return None


def _build_document(source: GitHubDocSource, url: str, content: str) -> str:
    header = [
        f"# {source.title}",
        "",
        f"- Repository: {source.repo}",
        f"- Category: {source.category}",
        f"- Path: {next(path for path in source.paths if url.endswith(path))}",
        f"- Source: {url}",
        "",
        "---",
        "",
    ]
    return "\n".join(header) + content.strip() + "\n"


def _load_previous_manifest(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [
        item["target_path"]
        for item in payload.get("sources", [])
        if isinstance(item, dict) and "target_path" in item
    ]


def _cleanup_previous_sync(project_root: Path, manifest_path: Path) -> None:
    for relative_path in _load_previous_manifest(manifest_path):
        target_path = project_root / relative_path
        if target_path.exists() and target_path.is_file():
            target_path.unlink()


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "github"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    _cleanup_previous_sync(project_root, manifest_path)

    manifest: list[dict[str, str]] = []
    success_count = 0
    failed: list[str] = []

    for source in SOURCES:
        print(f"Syncing {source.repo} :: {source.output_slug} ...", flush=True)
        resolved = _resolve_source(source)
        if resolved is None:
            failed.append(f"{source.repo}:{source.output_slug}")
            continue

        url, content = resolved
        repo_dir = raw_dir / _slugify(source.repo.replace("/", "_"))
        repo_dir.mkdir(parents=True, exist_ok=True)
        target_path = repo_dir / f"{source.output_slug}.md"
        target_path.write_text(
            _build_document(source, url, content),
            encoding="utf-8",
        )
        manifest.append(
            {
                "repo": source.repo,
                "title": source.title,
                "category": source.category,
                "source_url": url,
                "output_slug": source.output_slug,
                "target_path": str(target_path.relative_to(project_root)),
            }
        )
        success_count += 1

    manifest_path.write_text(
        json.dumps(
            {
                "total_sources": len(SOURCES),
                "synced_sources": success_count,
                "failed_sources": failed,
                "sources": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Synced {success_count}/{len(SOURCES)} GitHub documents.")
    print(f"Manifest written to: {manifest_path}")
    if failed:
        print("Failed sources:")
        for repo in failed:
            print(f"- {repo}")
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
