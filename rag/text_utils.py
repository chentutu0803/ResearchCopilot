from __future__ import annotations

import math
import re
from collections import Counter

EN_WORD_PATTERN = re.compile(r"[a-zA-Z0-9_+\-\.]+")
ZH_SEGMENT_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def normalize_text(text: str) -> str:
    return (
        " ".join(text.replace("\u3000", " ").replace("\n", " ").split()).strip().lower()
    )


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens: list[str] = []
    tokens.extend(EN_WORD_PATTERN.findall(normalized))

    for segment in ZH_SEGMENT_PATTERN.findall(normalized):
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))

    return [token for token in tokens if token.strip()]


def term_frequency(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0

    numerator = sum(left[key] * right.get(key, 0.0) for key in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
