from tools.citation_checker import CitationChecker


def test_citation_checker_detects_invalid_ids() -> None:
    checker = CitationChecker()
    ok, invalid = checker.validate("这是一个测试 [A1] [B2]", {"A1"})

    assert ok is False
    assert invalid == ["B2"]


def test_citation_checker_builds_reference_markdown() -> None:
    from core.schemas import EvidenceItem

    checker = CitationChecker()
    markdown = checker.build_reference_markdown(
        [
            EvidenceItem(
                evidence_id="KB-1", title="标题", source_url="https://example.com"
            )
        ]
    )

    assert "## 参考资料" in markdown
    assert "KB-1" in markdown
