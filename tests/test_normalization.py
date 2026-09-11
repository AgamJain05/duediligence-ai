"""
tests/test_normalization.py  ─  Unit tests for normalization.py
═══════════════════════════════════════════════════════════════
Tests:
  1. Multiple blank lines collapse to one
  2. Repeated spaces within a line collapse to one
  3. Wording remains intact after normalization
  4. Hyphen-broken words are repaired
  5. Ligature characters are replaced
  6. Repeated header/footer lines are detected
  7. Repeated lines are stripped from page text
  8. normalize_pages returns correct tuple structure
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.ingestion.normalization import (
    normalize_text,
    detect_repeated_lines,
    strip_repeated_lines,
    normalize_pages,
)


# ── 1. Blank line collapsing ──────────────────────────────────────────────────

def test_multiple_blank_lines_collapse():
    """3 or more consecutive blank lines should become at most 1."""
    text = "Line A\n\n\n\n\nLine B"
    result = normalize_text(text)
    assert "\n\n\n" not in result, f"Still has 3+ blank lines: {repr(result)}"


def test_two_blank_lines_preserved_or_collapsed():
    """Two blank lines (one blank line between paragraphs) should be handled."""
    text = "Paragraph A.\n\nParagraph B."
    result = normalize_text(text)
    # Should still have a separator (1 blank line is fine)
    assert "Paragraph A." in result
    assert "Paragraph B." in result


def test_single_blank_line_unchanged():
    """A single blank line between paragraphs should be preserved."""
    text = "First paragraph.\n\nSecond paragraph."
    result = normalize_text(text)
    assert "First paragraph." in result
    assert "Second paragraph." in result


# ── 2. Whitespace within lines ────────────────────────────────────────────────

def test_double_spaces_collapse_to_single():
    """Multiple spaces within a line should collapse to one space."""
    text = "Revenue  grew  by  22%  in  FY2024."
    result = normalize_text(text)
    assert "  " not in result, f"Double space still present: {repr(result)}"
    assert "Revenue" in result
    assert "22%" in result


def test_tab_within_line_collapses():
    """Tabs within a line should be collapsed."""
    text = "Revenue\t\tgrew\tby 22%."
    result = normalize_text(text)
    assert "\t\t" not in result


# ── 3. Wording preserved ──────────────────────────────────────────────────────

def test_actual_words_not_altered():
    """Normalization must not change actual words."""
    text = "OrionVault Systems provides cloud-native data management platforms."
    result = normalize_text(text)
    assert "OrionVault" in result
    assert "cloud-native" in result
    assert "data management" in result


def test_numbers_not_altered():
    """Numbers and percentages must survive normalization."""
    text = "Revenue was $148 million, up 22% year-over-year."
    result = normalize_text(text)
    assert "$148" in result
    assert "22%" in result


def test_punctuation_not_altered():
    """Terminal punctuation must be preserved."""
    text = "The risks include: (1) concentration, (2) supply chain."
    result = normalize_text(text)
    assert "(1)" in result
    assert "(2)" in result
    assert "supply chain." in result


# ── 4. Hyphen repair ──────────────────────────────────────────────────────────

def test_broken_hyphen_repaired():
    """Word broken with hyphen across lines should be joined."""
    text = "The company's sup-\nply chain depends on AWS."
    result = normalize_text(text)
    assert "supply" in result or "sup-" not in result.split("\n")[0]


def test_intentional_compound_hyphen_preserved():
    """Compound words like 'cloud-native' should NOT be joined."""
    text = "The cloud-native platform is scalable."
    result = normalize_text(text)
    assert "cloud-native" in result


# ── 5. Ligatures ──────────────────────────────────────────────────────────────

def test_fi_ligature_replaced():
    """Unicode fi ligature (\ufb01) → 'fi'."""
    text = "\ufb01nancial performance is strong."
    result = normalize_text(text)
    assert "\ufb01" not in result
    assert "financial" in result.lower()


def test_fl_ligature_replaced():
    """Unicode fl ligature (\ufb02) → 'fl'."""
    text = "Cash\ufb02ow grew significantly."
    result = normalize_text(text)
    assert "\ufb02" not in result
    assert "flow" in result.lower() or "Cashflow" in result


def test_smart_quotes_normalized():
    """Smart quotes should be converted to plain ASCII."""
    text = "\u201cOrionVault is exceptional.\u201d"
    result = normalize_text(text)
    assert "\u201c" not in result
    assert "\u201d" not in result
    assert "OrionVault is exceptional." in result


def test_non_breaking_space_replaced():
    """Non-breaking space (\u00a0) should be replaced with regular space."""
    text = "Revenue\u00a0grew\u00a022%."
    result = normalize_text(text)
    assert "\u00a0" not in result


# ── 6. Repeated header/footer detection ──────────────────────────────────────

def test_repeated_line_detected():
    """A line appearing on ≥60% of pages should be detected as header/footer."""
    # Create 5 pages, all with the same first line (simulating a footer)
    pages = [
        "OrionVault Systems — Confidential\n\nPage content for page " + str(i)
        for i in range(1, 6)
    ]
    repeated = detect_repeated_lines(pages)
    assert "OrionVault Systems — Confidential" in repeated


def test_non_repeated_line_not_detected():
    """A line appearing on only 1 of 5 pages should NOT be flagged."""
    pages = [
        "Page content for page 1.",
        "Page content for page 2.",
        "Page content for page 3.",
        "Page content for page 4.",
        "Unique content for page 5 with special text XYZ.",
    ]
    repeated = detect_repeated_lines(pages)
    assert "Unique content for page 5 with special text XYZ." not in repeated


def test_empty_pages_no_crash():
    """detect_repeated_lines should not crash on empty input."""
    result = detect_repeated_lines([])
    assert result == set()


# ── 7. Strip repeated lines ───────────────────────────────────────────────────

def test_repeated_line_stripped():
    """strip_repeated_lines should remove lines in the repeated set."""
    text     = "OrionVault Systems — Confidential\n\nActual content here."
    repeated = {"OrionVault Systems — Confidential"}
    result   = strip_repeated_lines(text, repeated)
    assert "OrionVault Systems — Confidential" not in result
    assert "Actual content here." in result


def test_non_repeated_line_not_stripped():
    """Content that is not in repeated set must be preserved."""
    text     = "Financial Performance\n\nRevenue was $148M."
    repeated = {"SomeOtherLine"}
    result   = strip_repeated_lines(text, repeated)
    assert "Financial Performance" in result
    assert "Revenue was $148M." in result


def test_strip_with_empty_repeated_set():
    """Empty repeated set should leave text unchanged."""
    text   = "Some text."
    result = strip_repeated_lines(text, set())
    assert result == text


# ── 8. normalize_pages ────────────────────────────────────────────────────────

def test_normalize_pages_returns_pairs():
    """normalize_pages must return a list of (raw, normalized) pairs."""
    pages = ["Line A\n\n\n\nLine B.", "Line C  Line D."]
    result = normalize_pages(pages)
    assert len(result) == 2
    for raw, norm in result:
        assert isinstance(raw, str)
        assert isinstance(norm, str)


def test_normalize_pages_raw_unchanged():
    """The raw text in each pair should be the original input."""
    pages = ["Original text here.", "Another page."]
    result = normalize_pages(pages)
    assert result[0][0] == "Original text here."
    assert result[1][0] == "Another page."


def test_normalize_pages_empty_input():
    """normalize_pages on empty list returns empty list."""
    result = normalize_pages([])
    assert result == []
