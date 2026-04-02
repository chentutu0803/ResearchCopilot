from __future__ import annotations

from pathlib import Path
import sys

from core.schemas import EvidenceItem, SearchResult
from core.settings import Settings
from core.tracer import TraceCollector
from rag.chunker import Chunker
from rag.retriever import HybridRetriever


class KnowledgeBaseTool:
    def __init__(
        self, settings: Settings, tracer: TraceCollector | None = None
    ) -> None:
        self.settings = settings
        self.tracer = tracer
        self.chunker = Chunker()
        self.retriever = HybridRetriever(settings)
        self.documents: list[EvidenceItem] = []
        self.is_ready = False

    def _debug(self, message: str) -> None:
        if self.settings.log_level.upper() != "DEBUG":
            return
        print(f"[DEBUG knowledge_base] {message}", file=sys.stderr, flush=True)

    def _load_documents(self) -> list[EvidenceItem]:
        raw_dir = self.settings.data_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            [
                file_path
                for pattern in ("**/*.md", "**/*.txt")
                for file_path in raw_dir.glob(pattern)
                if file_path.is_file()
            ]
        )

        documents: list[EvidenceItem] = []
        for file_index, file_path in enumerate(files, start=1):
            content = file_path.read_text(encoding="utf-8")
            relative_path = file_path.relative_to(raw_dir)
            source_name = relative_path.with_suffix("").as_posix()
            chunks = self.chunker.split(content)
            for chunk_index, chunk in enumerate(chunks, start=1):
                evidence_id = f"KB-{file_index:02d}-{chunk_index:03d}"
                documents.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        title=source_name.replace("/", " :: "),
                        source_url=str(file_path),
                        source_type="kb",
                        snippet=chunk[:180],
                        content=chunk,
                        chunk_id=evidence_id,
                        metadata={
                            "file_name": source_name,
                            "relative_path": relative_path.as_posix(),
                            "modified_at": file_path.stat().st_mtime,
                        },
                        )
                )

        return documents

    def build(self, force_rebuild: bool = False) -> None:
        documents = self._load_documents()
        self.documents = documents
        self.retriever.build(documents, force_rebuild=force_rebuild)
        self.is_ready = True

    def rebuild(self) -> None:
        self.build(force_rebuild=True)

    def load(self) -> None:
        documents = self._load_documents()
        self.documents = documents
        if self.settings.kb_backend.lower() == "qdrant":
            self.retriever.attach_documents(documents)
        else:
            self.retriever.build(documents)
        self.is_ready = True

    def retrieve(self, query: str, top_k: int | None = None) -> SearchResult:
        if not self.is_ready:
            self.load()
        self._debug(f"start query={query} top_k={top_k or self.settings.top_k_rerank}")

        if not self.documents:
            self._debug(f"empty query={query}")
            return SearchResult(
                query=query, items=[], total=0, metadata={"source": "kb"}
            )

        items = self.retriever.retrieve(
            query,
            top_k_bm25=self.settings.top_k_bm25,
            top_k_dense=self.settings.top_k_dense,
            top_k_rerank=top_k or self.settings.top_k_rerank,
        )

        if self.tracer is not None:
            self.tracer.add_event(
                event_type="tool",
                step_name="tool.knowledge_base",
                status="success",
                tool_name="knowledge_base",
                input_summary=query,
                output_summary=f"知识库命中 {len(items)} 条结果",
                metadata={"result_count": len(items)},
            )

        self._debug(f"done query={query} result_count={len(items)}")
        return SearchResult(
            query=query, items=items, total=len(items), metadata={"source": "kb"}
        )

    def retrieve_by_method(
        self, query: str, method: str, top_k: int | None = None
    ) -> SearchResult:
        if not self.is_ready:
            self.load()

        if not self.documents:
            return SearchResult(
                query=query,
                items=[],
                total=0,
                metadata={"source": "kb", "method": method},
            )

        limit = top_k or self.settings.top_k_rerank
        if method == "bm25":
            items = self.retriever.retrieve_bm25(query, top_k=limit)
        elif method == "vector":
            items = self.retriever.retrieve_vector(query, top_k=limit)
        else:
            items = self.retriever.retrieve(
                query,
                top_k_bm25=self.settings.top_k_bm25,
                top_k_dense=self.settings.top_k_dense,
                top_k_rerank=limit,
            )

        return SearchResult(
            query=query,
            items=items,
            total=len(items),
            metadata={"source": "kb", "method": method},
        )
