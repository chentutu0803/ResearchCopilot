from __future__ import annotations

import re

from core.schemas import EvidenceItem


class CitationChecker:
    citation_pattern = re.compile(r"\[([^\[\]]+)\]")

    def extract_citations(self, text: str) -> list[str]:
        return self.citation_pattern.findall(text)

    def validate(self, text: str, valid_ids: set[str]) -> tuple[bool, list[str]]:
        found = self.extract_citations(text)
        invalid = [citation_id for citation_id in found if citation_id not in valid_ids]
        return len(invalid) == 0, invalid

    def build_reference_markdown(self, references: list[EvidenceItem]) -> str:
        lines = ["## 参考资料"]
        for item in references:
            source = item.source_url or "无来源链接"
            lines.append(f"- [{item.evidence_id}] {item.title} - {source}")
        return "\n".join(lines)
