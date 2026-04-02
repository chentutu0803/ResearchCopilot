from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core.settings import Settings
from rag.text_utils import tokenize_text


def _load_qdrant_modules() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "未安装 qdrant-client，无法启用 Qdrant 官方 BM25 检索。"
        ) from exc
    return QdrantClient, models


class LocalBM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        self.avg_doc_len = 0.0

    def build(self, documents: list[str]) -> None:
        self.documents = documents
        self.doc_tokens = [tokenize_text(document) for document in documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = (
            sum(len(tokens) for tokens in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens
            else 0.0
        )

        document_frequency: defaultdict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                document_frequency[token] += 1

        total_docs = len(self.doc_tokens)
        self.idf = {
            token: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for token, freq in document_frequency.items()
        }

    def attach_documents(self, documents: list[str]) -> None:
        self.build(documents)

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize_text(query)
        scores: list[float] = []
        for tokens, frequencies in zip(self.doc_tokens, self.doc_freqs, strict=False):
            doc_len = len(tokens) or 1
            total = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if frequency == 0:
                    continue
                idf = self.idf.get(token, 0.0)
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_len / (self.avg_doc_len or 1)
                )
                total += idf * numerator / denominator
            scores.append(total)
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [item for item in ranked[:k] if item[1] > 0]


class QdrantOfficialBM25Index:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.documents: list[str] = []
        self.cache_meta_path = (
            settings.project_root
            / "data"
            / "qdrant"
            / f"{settings.qdrant_bm25_collection}_bm25_meta.json"
        )
        self.cache_meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = self._build_client()

    def _build_client(self) -> Any:
        QdrantClient, _ = _load_qdrant_modules()
        return QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            cloud_inference=self.settings.qdrant_cloud_inference,
        )

    def build(self, documents: list[str]) -> None:
        self.documents = documents
        fingerprint = self._fingerprint_documents(documents)
        if not documents:
            if self.client.collection_exists(self.settings.qdrant_bm25_collection):
                self.client.delete_collection(self.settings.qdrant_bm25_collection)
            if self.cache_meta_path.exists():
                self.cache_meta_path.unlink()
            return

        if self._is_cache_valid(fingerprint):
            return

        _, models = _load_qdrant_modules()
        collection_name = self.settings.qdrant_bm25_collection
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                self.settings.qdrant_bm25_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

        points = [
            models.PointStruct(
                id=index + 1,
                vector={
                    self.settings.qdrant_bm25_vector_name: models.Document(
                        text=document,
                        model="qdrant/bm25",
                        options=self._document_options(),
                    )
                },
                payload={"doc_index": index},
            )
            for index, document in enumerate(documents)
        ]
        self.client.upsert(collection_name=collection_name, wait=True, points=points)
        self.cache_meta_path.write_text(
            json.dumps(
                {
                    "collection_name": collection_name,
                    "document_count": len(documents),
                    "fingerprint": fingerprint,
                    "tokenizer": self.settings.qdrant_bm25_tokenizer,
                    "language": self.settings.qdrant_bm25_language,
                    "ascii_folding": self.settings.qdrant_bm25_ascii_folding,
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
                "Qdrant 官方 BM25 索引不存在或已过期，请先运行 scripts/build_kb.py。"
            )

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.documents:
            return []

        _, models = _load_qdrant_modules()
        response = self.client.query_points(
            collection_name=self.settings.qdrant_bm25_collection,
            query=models.Document(
                text=query,
                model="qdrant/bm25",
                options=self._document_options(),
            ),
            using=self.settings.qdrant_bm25_vector_name,
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

    def _document_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.settings.qdrant_bm25_tokenizer.strip():
            options["tokenizer"] = self.settings.qdrant_bm25_tokenizer.strip()
        if self.settings.qdrant_bm25_language.strip():
            options["language"] = self.settings.qdrant_bm25_language.strip()
        if self.settings.qdrant_bm25_ascii_folding:
            options["lowercase"] = True
            options["ascii_folding"] = True
        return options

    def _is_cache_valid(self, fingerprint: str) -> bool:
        if not self.client.collection_exists(self.settings.qdrant_bm25_collection):
            return False
        if not self.cache_meta_path.exists():
            return False
        try:
            payload = json.loads(self.cache_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return (
            payload.get("collection_name") == self.settings.qdrant_bm25_collection
            and payload.get("fingerprint") == fingerprint
            and payload.get("document_count") == len(self.documents)
            and payload.get("tokenizer") == self.settings.qdrant_bm25_tokenizer
            and payload.get("language") == self.settings.qdrant_bm25_language
            and payload.get("ascii_folding")
            == self.settings.qdrant_bm25_ascii_folding
        )

    @staticmethod
    def _fingerprint_documents(documents: list[str]) -> str:
        digest = hashlib.sha256()
        for text in documents:
            digest.update(text.encode("utf-8", errors="ignore"))
            digest.update(b"\n<doc>\n")
        return digest.hexdigest()


class BM25Index:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        if self.settings.use_qdrant_official_bm25:
            self.backend: LocalBM25Index | QdrantOfficialBM25Index = (
                QdrantOfficialBM25Index(self.settings)
            )
        else:
            self.backend = LocalBM25Index()

    def build(self, documents: list[str]) -> None:
        self.backend.build(documents)

    def attach_documents(self, documents: list[str]) -> None:
        self.backend.attach_documents(documents)

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.backend.top_k(query, k)
