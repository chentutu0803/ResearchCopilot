from __future__ import annotations


class Chunker:
    def __init__(self, chunk_size: int = 420, overlap: int = 60) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        paragraphs = [
            paragraph.strip()
            for paragraph in normalized.split("\n")
            if paragraph.strip()
        ]
        if not paragraphs:
            paragraphs = [normalized]

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(current) + len(paragraph) + 1 <= self.chunk_size:
                current = f"{current}\n{paragraph}".strip()
                continue

            if current:
                chunks.append(current)

            if len(paragraph) <= self.chunk_size:
                current = paragraph
                continue

            start = 0
            while start < len(paragraph):
                end = start + self.chunk_size
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = max(0, end - self.overlap)
            current = ""

        if current:
            chunks.append(current)

        return chunks
