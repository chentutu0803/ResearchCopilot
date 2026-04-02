from __future__ import annotations

from core.schemas import EvidenceItem
from core.settings import Settings
from rag.bm25_index import BM25Index
from rag.reranker import Reranker
from rag.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.documents: list[EvidenceItem] = []
        self.bm25 = BM25Index(self.settings)
        self.vector_store = VectorStore(self.settings)
        self.reranker = Reranker()

    def build(
        self, documents: list[EvidenceItem], force_rebuild: bool = False
    ) -> None:
        self.documents = documents
        texts = [self._document_text(item) for item in documents]
        self.bm25.build(texts)
        self.vector_store.build(texts, force_rebuild=force_rebuild)

    def attach_documents(self, documents: list[EvidenceItem]) -> None:
        self.documents = documents
        texts = [self._document_text(item) for item in documents]
        self.bm25.attach_documents(texts)
        self.vector_store.attach_documents(texts)

    def retrieve(
        self,
        query: str,
        top_k_bm25: int = 8,
        top_k_dense: int = 8,
        top_k_rerank: int = 5,
    ) -> list[EvidenceItem]:
        merged: dict[str, EvidenceItem] = {}

        for index, score in self.bm25.top_k(query, top_k_bm25):
            item = self.documents[index].model_copy(deep=True)
            item.score = score
            item.metadata["retrieval_method"] = "bm25"
            merged[item.evidence_id] = item

        for index, score in self.vector_store.top_k(query, top_k_dense):
            item = self.documents[index].model_copy(deep=True)
            if item.evidence_id in merged:
                merged[item.evidence_id].score = (
                    merged[item.evidence_id].score or 0.0
                ) + score
                merged[item.evidence_id].metadata["retrieval_method"] = "hybrid"
                merged[item.evidence_id].metadata["dense_score"] = score
                continue
            item.score = score
            item.metadata["retrieval_method"] = "vector"
            item.metadata["dense_score"] = score
            merged[item.evidence_id] = item

        candidates = list(merged.values())
        return self.reranker.rerank(query, candidates, top_k=top_k_rerank)

    def retrieve_bm25(self, query: str, top_k: int = 5) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for index, score in self.bm25.top_k(query, top_k):
            item = self.documents[index].model_copy(deep=True)
            item.score = score
            item.metadata["retrieval_method"] = "bm25"
            items.append(item)
        return items

    def retrieve_vector(self, query: str, top_k: int = 5) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for index, score in self.vector_store.top_k(query, top_k):
            item = self.documents[index].model_copy(deep=True)
            item.score = score
            item.metadata["retrieval_method"] = "vector"
            item.metadata["dense_score"] = score
            items.append(item)
        return items

    @staticmethod
    def _document_text(item: EvidenceItem) -> str:
        return "\n".join([item.title, item.snippet, item.content]).strip()
