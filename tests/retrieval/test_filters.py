"""
tests/retrieval/test_filters.py  -  Unit tests for MetadataFilter
=================================================================
All tests use synthetic deterministic data. No I/O. No external calls.
"""

import pytest
from src.retrieval.models import RetrievalResult
from src.retrieval.filters import MetadataFilter


def make_result(
    chunk_id:     str = "abc",
    section:      str = "Risk Factors",
    content_type: str = "text",
    page_start:   int = 1,
    page_end:     int = 2,
    source:       str = "test.pdf",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id      = chunk_id,
        text          = f"Content of {chunk_id}",
        source        = source,
        page_start    = page_start,
        page_end      = page_end,
        section       = section,
        section_path  = [section],
        content_type  = content_type,
        metadata      = {},
    )


class TestMetadataFilter:

    def setup_method(self):
        self.f = MetadataFilter()

    # ── No filter (passthrough) ───────────────────────────────────────────────

    def test_none_filters_returns_all(self):
        candidates = [make_result("A"), make_result("B")]
        result = self.f.apply(candidates, filters=None)
        assert len(result) == 2

    def test_empty_filters_returns_all(self):
        candidates = [make_result("A"), make_result("B")]
        result = self.f.apply(candidates, filters={})
        assert len(result) == 2

    def test_empty_candidates(self):
        result = self.f.apply([], filters={"section": "Risk"})
        assert result == []

    # ── Section filter ────────────────────────────────────────────────────────

    def test_section_exact_match(self):
        candidates = [
            make_result("A", section="Risk Factors"),
            make_result("B", section="Financial Performance"),
        ]
        result = self.f.apply(candidates, {"section": "Risk Factors"})
        assert len(result) == 1
        assert result[0].chunk_id == "A"

    def test_section_substring_match(self):
        """Substring: 'risk' matches 'Key Risks' and 'Risk Factors'."""
        candidates = [
            make_result("A", section="Risk Factors"),
            make_result("B", section="Key Risks"),
            make_result("C", section="Financial Performance"),
        ]
        result = self.f.apply(candidates, {"section": "risk"})
        ids = {r.chunk_id for r in result}
        assert "A" in ids
        assert "B" in ids
        assert "C" not in ids

    def test_section_case_insensitive(self):
        candidates = [make_result("A", section="RISK FACTORS")]
        result = self.f.apply(candidates, {"section": "risk factors"})
        assert len(result) == 1

    def test_section_no_match(self):
        candidates = [make_result("A", section="Financial Performance")]
        result = self.f.apply(candidates, {"section": "Risk"})
        assert result == []

    # ── Content type filter ───────────────────────────────────────────────────

    def test_content_type_exact(self):
        candidates = [
            make_result("A", content_type="table"),
            make_result("B", content_type="text"),
            make_result("C", content_type="list"),
        ]
        result = self.f.apply(candidates, {"content_type": "table"})
        assert len(result) == 1
        assert result[0].chunk_id == "A"

    def test_content_type_case_insensitive(self):
        candidates = [make_result("A", content_type="TABLE")]
        result = self.f.apply(candidates, {"content_type": "table"})
        assert len(result) == 1

    # ── Page range filter ─────────────────────────────────────────────────────

    def test_page_start_filter(self):
        """Filter: page_start >= 3."""
        candidates = [
            make_result("A", page_start=1, page_end=2),
            make_result("B", page_start=3, page_end=4),
            make_result("C", page_start=5, page_end=5),
        ]
        result = self.f.apply(candidates, {"page_start": 3})
        ids = {r.chunk_id for r in result}
        assert "A" not in ids
        assert "B" in ids
        assert "C" in ids

    def test_page_end_filter(self):
        """Filter: page_end <= 3."""
        candidates = [
            make_result("A", page_start=1, page_end=2),
            make_result("B", page_start=3, page_end=4),
            make_result("C", page_start=1, page_end=3),
        ]
        result = self.f.apply(candidates, {"page_end": 3})
        ids = {r.chunk_id for r in result}
        assert "A" in ids
        assert "B" not in ids
        assert "C" in ids

    def test_page_range_combined(self):
        """Filter: pages 2-4 only."""
        candidates = [
            make_result("A", page_start=1, page_end=1),
            make_result("B", page_start=2, page_end=3),
            make_result("C", page_start=4, page_end=5),
        ]
        result = self.f.apply(candidates, {"page_start": 2, "page_end": 4})
        ids = {r.chunk_id for r in result}
        assert "A" not in ids
        assert "B" in ids
        # C: page_end=5 > 4 -> filtered out
        assert "C" not in ids

    # ── Source filename filter ────────────────────────────────────────────────

    def test_source_filename_filter(self):
        candidates = [
            make_result("A", source="OrionVault.pdf"),
            make_result("B", source="AcmeCorp.pdf"),
        ]
        result = self.f.apply(candidates, {"source_filename": "orionvault"})
        assert len(result) == 1
        assert result[0].chunk_id == "A"

    # ── Combined filters ──────────────────────────────────────────────────────

    def test_multiple_filters_AND_logic(self):
        """All filters must pass (AND semantics)."""
        candidates = [
            make_result("A", section="Risk Factors", content_type="text"),
            make_result("B", section="Risk Factors", content_type="table"),
            make_result("C", section="Financial",    content_type="table"),
        ]
        result = self.f.apply(candidates, {"section": "risk", "content_type": "table"})
        assert len(result) == 1
        assert result[0].chunk_id == "B"

    # ── Explain method ────────────────────────────────────────────────────────

    def test_explain_returns_string(self):
        candidates = [make_result("A"), make_result("B")]
        explanation = self.f.explain(candidates, {"content_type": "table"})
        assert isinstance(explanation, str)
        assert "PASS" in explanation or "FAIL" in explanation

    def test_explain_no_filters(self):
        result = self.f.explain([], {})
        assert "No filters" in result

    # ── Order preservation ────────────────────────────────────────────────────

    def test_relative_order_preserved(self):
        """Filter must not change the relative order of passing candidates."""
        candidates = [
            make_result("A", content_type="text"),
            make_result("B", content_type="table"),   # filtered out
            make_result("C", content_type="text"),
        ]
        result = self.f.apply(candidates, {"content_type": "text"})
        assert [r.chunk_id for r in result] == ["A", "C"]
