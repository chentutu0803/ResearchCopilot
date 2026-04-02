from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.embeddings import EmbeddingService


class QdrantVectorStore:
    def __init__(
        self,
        *,
        path: str,
        collection_name: str,
        embedding_backend: str,
        embedding_model_name: str,
        hf_endpoint: str,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.cache_meta_path = self.path / f"{self.collection_name}_build_meta.json"
        self.embedding_service = EmbeddingService(
            prefer_real_model=True,
            backend=embedding_backend,
            model_name=embedding_model_name,
            hf_endpoint=hf_endpoint,
        )
        self.client = QdrantClient(path=str(self.path))
        self.documents: list[str] = []

    def build(self, documents: list[str], force_rebuild: bool = False) -> None:
        self.documents = documents
        fingerprint = self._fingerprint_documents(documents)
        if not documents:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            if self.cache_meta_path.exists():
                self.cache_meta_path.unlink()
            return

        if not force_rebuild and self._is_cache_valid(fingerprint):
            return

        vectors = self.embedding_service.embed_batch(documents)
        if not vectors or not isinstance(vectors[0], list):
            raise RuntimeError("Qdrant 后端需要真实 dense embedding，当前 embedding 不可用。")

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE,
            ),
        )

        points = [
            models.PointStruct(
                id=index + 1,
                vector=vector,
                payload={"doc_index": index},
            )
            for index, vector in enumerate(vectors)
        ]
        self.client.upsert(self.collection_name, points=points, wait=True)
        self.cache_meta_path.write_text(
            json.dumps(
                {
                    "collection_name": self.collection_name,
                    "document_count": len(documents),
                    "fingerprint": fingerprint,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def attach_documents(self, documents: list[str]) -> None:
        self.documents = documents
        fingerprint = self._fingerprint_documents(documents)
        if not self._is_cache_valid(fingerprint):
            raise RuntimeError(
                "知识库向量索引不存在或已过期，请先运行 scripts/build_kb.py，"
                "文档更新后运行 scripts/rebuild_kb.py。"
            )

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.documents:
            return []

        query_vector = self.embedding_service.embed(query)
        if not isinstance(query_vector, list):
            return []

        response = self.client.query_points(
            self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        results: list[tuple[int, float]] = []
        for point in response.points:
            doc_index = point.payload.get("doc_index") if point.payload else None
            if not isinstance(doc_index, int):
                continue
            results.append((doc_index, float(point.score)))
        return results

    def _is_cache_valid(self, fingerprint: str) -> bool:
        if not self.client.collection_exists(self.collection_name):
            return False
        if not self.cache_meta_path.exists():
            return False
        try:
            payload = json.loads(self.cache_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return (
            payload.get("collection_name") == self.collection_name
            and payload.get("fingerprint") == fingerprint
            and payload.get("document_count") == len(self.documents)
        )

    @staticmethod
    def _fingerprint_documents(documents: list[str]) -> str:
        digest = hashlib.sha256()
        for text in documents:
            digest.update(text.encode("utf-8", errors="ignore"))
            digest.update(b"\n<doc>\n")
        return digest.hexdigest()
