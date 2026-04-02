from __future__ import annotations

import math

from core.settings import Settings
from rag.embeddings import EmbeddingService
from rag.qdrant_store import QdrantVectorStore
from rag.text_utils import cosine_similarity


class InMemoryVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.embedding_service = EmbeddingService(
            prefer_real_model=True,
            backend=settings.embedding_backend,
            model_name=settings.embedding_model_name,
            hf_endpoint=settings.hf_endpoint,
        )
        self.documents: list[str] = []
        self.document_vectors: list[object] = []

    def build(self, documents: list[str], force_rebuild: bool = False) -> None:
        self.documents = documents
        self.embedding_service.fit(documents)
        self.document_vectors = [
            self.embedding_service.embed(document) for document in documents
        ]

    def attach_documents(self, documents: list[str]) -> None:
        self.build(documents)

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        query_vector = self.embedding_service.embed(query)
        scored = [
            (index, self._score(query_vector, vector))
            for index, vector in enumerate(self.document_vectors)
        ]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        return [item for item in ranked[:k] if item[1] > 0]

    def _score(self, query_vector: object, document_vector: object) -> float:
        if self.embedding_service.mode == "lightweight":
            return cosine_similarity(query_vector, document_vector)  # type: ignore[arg-type]
        if not isinstance(query_vector, list) or not isinstance(document_vector, list):
            return 0.0
        numerator = sum(
            left * right
            for left, right in zip(query_vector, document_vector, strict=False)
        )
        left_norm = math.sqrt(sum(value * value for value in query_vector))
        right_norm = math.sqrt(sum(value * value for value in document_vector))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        backend = self.settings.kb_backend.lower()
        if backend == "qdrant":
            self.backend = QdrantVectorStore(
                path=str(self.settings.qdrant_path_resolved),
                collection_name=self.settings.qdrant_collection,
                embedding_backend=self.settings.embedding_backend,
                embedding_model_name=self.settings.embedding_model_name,
                hf_endpoint=self.settings.hf_endpoint,
            )
        else:
            self.backend = InMemoryVectorStore(self.settings)

    def build(self, documents: list[str], force_rebuild: bool = False) -> None:
        self.backend.build(documents, force_rebuild=force_rebuild)

    def attach_documents(self, documents: list[str]) -> None:
        self.backend.attach_documents(documents)

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.backend.top_k(query, k)
