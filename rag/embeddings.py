from __future__ import annotations

import math
import os
from collections import defaultdict
from importlib import import_module
from typing import Any

from rag.text_utils import term_frequency, tokenize_text


def _load_sentence_transformer() -> Any | None:
    try:
        module = import_module("sentence_transformers")
        return getattr(module, "SentenceTransformer", None)
    except Exception:  # noqa: BLE001
        return None

def _configure_hf_endpoint(hf_endpoint: str) -> None:
    if not hf_endpoint.strip():
        return
    os.environ["HF_ENDPOINT"] = hf_endpoint
    try:
        constants_module = import_module("huggingface_hub.constants")
        import_module("importlib").reload(constants_module)
    except Exception:  # noqa: BLE001
        return


class LightweightEmbeddingService:
    def __init__(self) -> None:
        self.document_frequencies: defaultdict[str, int] = defaultdict(int)
        self.total_documents = 0
        self.mode = "lightweight"

    def fit(self, texts: list[str]) -> None:
        self.document_frequencies.clear()
        self.total_documents = len(texts)
        for text in texts:
            tokens = set(tokenize_text(text))
            for token in tokens:
                self.document_frequencies[token] += 1

    def embed(self, text: str) -> dict[str, float]:
        tokens = tokenize_text(text)
        frequencies = term_frequency(tokens)
        total = sum(frequencies.values()) or 1

        vector: dict[str, float] = {}
        for token, count in frequencies.items():
            tf = count / total
            df = self.document_frequencies.get(token, 0)
            idf = math.log((1 + self.total_documents) / (1 + df)) + 1.0
            vector[token] = tf * idf
        return vector

    def embed_batch(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.embed(text) for text in texts]

class SentenceTransformerEmbeddingService:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        hf_endpoint: str = "https://hf-mirror.com",
    ) -> None:
        sentence_transformer_cls = _load_sentence_transformer()
        if sentence_transformer_cls is None:
            raise RuntimeError(
                "sentence-transformers 未安装，无法启用真实 embedding 模型。"
            )
        _configure_hf_endpoint(hf_endpoint)
        self.model = sentence_transformer_cls(model_name)
        self.mode = "sentence_transformer"

    def fit(self, texts: list[str]) -> None:
        return None

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]


class EmbeddingService:
    def __init__(
        self,
        prefer_real_model: bool = True,
        backend: str = "sentence_transformer",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        hf_endpoint: str = "https://hf-mirror.com",
    ) -> None:
        backend_name = backend.lower()
        if backend_name == "lightweight":
            self.backend = LightweightEmbeddingService()
        elif prefer_real_model and _load_sentence_transformer() is not None:
            self.backend = SentenceTransformerEmbeddingService(
                model_name=model_name,
                hf_endpoint=hf_endpoint,
            )
        else:
            self.backend = LightweightEmbeddingService()
        self.mode = self.backend.mode

    def fit(self, texts: list[str]) -> None:
        self.backend.fit(texts)

    def embed(self, text: str) -> Any:
        return self.backend.embed(text)

    def embed_batch(self, texts: list[str]) -> list[Any]:
        if hasattr(self.backend, "embed_batch"):
            return self.backend.embed_batch(texts)
        return [self.backend.embed(text) for text in texts]
