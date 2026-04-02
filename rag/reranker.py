from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from core.schemas import EvidenceItem
from rag.text_utils import tokenize_text


def _load_cross_encoder() -> Any | None:
    try:
        module = import_module("sentence_transformers")
        return getattr(module, "CrossEncoder", None)
    except Exception:  # noqa: BLE001
        return None


class Reranker:
    def __init__(self, prefer_real_model: bool = True) -> None:
        self.mode = "heuristic"
        self.cross_encoder = None
        cross_encoder_cls = cast(type[Any] | None, _load_cross_encoder())
        if prefer_real_model and cross_encoder_cls is not None:
            try:
                self.cross_encoder = cross_encoder_cls("BAAI/bge-reranker-v2-m3")
                self.mode = "cross_encoder"
            except Exception:  # noqa: BLE001
                self.cross_encoder = None

    def rerank(
        self, query: str, candidates: list[EvidenceItem], top_k: int = 5
    ) -> list[EvidenceItem]:
        if self.cross_encoder is not None and candidates:
            pairs = [
                [query, f"{item.title}\n{item.snippet}\n{item.content}"]
                for item in candidates
            ]
            scores = self.cross_encoder.predict(pairs)
            rescored: list[EvidenceItem] = []
            for candidate, score in zip(candidates, scores, strict=False):
                updated = candidate.model_copy(deep=True)
                updated.score = float(score)
                updated.metadata["rerank_mode"] = self.mode
                updated.metadata["rerank_score"] = float(score)
                rescored.append(updated)
            return sorted(rescored, key=lambda item: item.score or 0.0, reverse=True)[
                :top_k
            ]

        query_tokens = set(tokenize_text(query))
        rescored: list[EvidenceItem] = []

        for candidate in candidates:
            title_tokens = set(tokenize_text(candidate.title))
            body_tokens = set(
                tokenize_text(candidate.snippet + " " + candidate.content)
            )
            title_overlap = len(query_tokens & title_tokens)
            body_overlap = len(query_tokens & body_tokens)
            base_score = candidate.score or 0.0
            rerank_score = base_score + title_overlap * 2.0 + body_overlap * 0.5

            updated = candidate.model_copy(deep=True)
            updated.score = rerank_score
            updated.metadata["rerank_mode"] = self.mode
            updated.metadata["rerank_score"] = rerank_score
            rescored.append(updated)

        return sorted(rescored, key=lambda item: item.score or 0.0, reverse=True)[
            :top_k
        ]
