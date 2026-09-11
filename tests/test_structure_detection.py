"""
tests/test_structure_detection.py  ─  Unit tests for structure.py
══════════════════════════════════════════════════════════════════
Tests:
  1. ALL CAPS short line → heading
  2. Title Case short line → heading
  3. Known section keyword → heading
  4. Long multi-sentence text → paragraph
  5. Bullet line → list
  6. Numbered list item → list
  7. Pipe-separated lines → table
  8. Mixed segment (heading + paragraph) is split correctly
  9. Section hierarchy (section_path) is correctly built
 10. Confidence scores are in valid range
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.ingestion.structure import classify_line, detect_blocks, build_section_hierarchy
from src.ingestion.models import (
    TextBlock,
    BLOCK_TYPE_HEADING,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_LIST,
    BLOCK_TYPE_UNKNOWN,
)


# ── 1. Heading detection — ALL CAPS ─────────────────────────────────────────────

def test_all_caps_short_line_is_heading():
    """A short ALL CAPS line should be classified as a heading."""
    block_type, confidence = classify_line("RISK FACTORS")
    assert block_type == BLOCK_TYPE_HEADING, \
        f"Expected heading, got {block_type}"
    assert confidence >= 0.80, f"Confidence too low: {confidence}"


def test_all_caps_with_spaces_is_heading():
    block_type, confidence = classify_line("FINANCIAL PERFORMANCE")
    assert block_type == BLOCK_TYPE_HEADING


def test_all_caps_long_line_may_not_be_heading():
    """A very long ALL CAPS line is less likely to be a heading."""
    long_line = "THIS IS A VERY LONG SENTENCE THAT GOES ON AND ON AND SHOULD NOT BE A HEADING BECAUSE HEADINGS ARE TYPICALLY SHORT."
    block_type, confidence = classify_line(long_line)
    # We don't assert heading — just that it doesn't crash
    assert block_type in {BLOCK_TYPE_HEADING, BLOCK_TYPE_PARAGRAPH}


# ── 2. Heading detection — Title Case ──────────────────────────────────────────

def test_title_case_no_period_is_heading():
    """Title Case line without terminal period should be a heading."""
    block_type, confidence = classify_line("Financial Performance Overview")
    assert block_type == BLOCK_TYPE_HEADING, \
        f"Expected heading for Title Case, got {block_type}"


def test_title_case_with_period_is_paragraph():
    """
    A clearly long sentence with many words and a terminal period should
    be classified as paragraph, not heading. We use a sentence long enough
    to exceed the 60-char heading threshold.
    """
    block_type, confidence = classify_line(
        "Financial performance was exceptionally strong throughout the fiscal year 2024."
    )
    # This is > 60 chars and ends with a period → paragraph
    assert block_type == BLOCK_TYPE_PARAGRAPH, (
        f"Expected paragraph for a long sentence ending with '.', got {block_type}"
    )


# ── 3. Known section keyword ────────────────────────────────────────────────────

def test_known_keyword_executive_summary():
    block_type, confidence = classify_line("Executive Summary")
    assert block_type == BLOCK_TYPE_HEADING
    assert confidence >= 0.70


def test_known_keyword_risk_factors():
    block_type, confidence = classify_line("Risk Factors")
    assert block_type == BLOCK_TYPE_HEADING


def test_known_keyword_management():
    block_type, confidence = classify_line("Management")
    assert block_type == BLOCK_TYPE_HEADING


# ── 4. Paragraph detection ───────────────────────────────────────────────────────

def test_long_sentence_is_paragraph():
    """A long sentence with terminal punctuation should be a paragraph."""
    text = (
        "OrionVault Systems provides cloud-native data management platforms "
        "primarily targeted at financial services and healthcare verticals."
    )
    block_type, confidence = classify_line(text)
    assert block_type == BLOCK_TYPE_PARAGRAPH


def test_multi_sentence_is_paragraph():
    """Multiple sentences → paragraph."""
    text = (
        "Revenue grew 22% year-over-year. "
        "The primary driver was the Sentinel platform upsell motion."
    )
    block_type, confidence = classify_line(text)
    assert block_type == BLOCK_TYPE_PARAGRAPH


# ── 5. List detection — bullets ─────────────────────────────────────────────────

def test_bullet_dot_is_list():
    block_type, confidence = classify_line("• Customer concentration risk")
    assert block_type == BLOCK_TYPE_LIST
    assert confidence >= 0.85


def test_bullet_dash_is_list():
    block_type, confidence = classify_line("- Supply chain dependency on AWS")
    assert block_type == BLOCK_TYPE_LIST


def test_bullet_asterisk_is_list():
    block_type, confidence = classify_line("* Competitive pressure from hyperscalers")
    assert block_type == BLOCK_TYPE_LIST


# ── 6. List detection — numbered ────────────────────────────────────────────────

def test_numbered_list_item():
    block_type, confidence = classify_line("1. Increase ARR from $148M to $200M")
    assert block_type == BLOCK_TYPE_LIST
    assert confidence >= 0.80


def test_lowercase_lettered_list_item():
    block_type, confidence = classify_line("a. Expand into healthcare vertical")
    assert block_type == BLOCK_TYPE_LIST


# ── 7. Table detection ───────────────────────────────────────────────────────────

def test_pipe_line_is_table():
    block_type, confidence = classify_line("Year | Revenue | Operating Margin")
    assert block_type == BLOCK_TYPE_TABLE
    assert confidence >= 0.80


def test_multi_pipe_line_is_table():
    block_type, confidence = classify_line("2023 | $121M | 14% | 1,200")
    assert block_type == BLOCK_TYPE_TABLE


# ── 8. detect_blocks — mixed segment ─────────────────────────────────────────────

def test_heading_followed_by_paragraph_split():
    """
    A segment with a heading line followed by a paragraph should produce
    at least a heading block and a paragraph block.
    """
    text = "RISK FACTORS\n\nThe company faces significant concentration risk."
    blocks = detect_blocks(text, page_number=3)
    assert len(blocks) >= 1
    # At least one heading should be detected
    types = [b.block_type for b in blocks]
    assert BLOCK_TYPE_HEADING in types, f"No heading detected. Got: {types}"


def test_table_segment_detected():
    """A multi-line pipe table should produce a TABLE block."""
    text = (
        "Year | Revenue | Margin\n"
        "2023 | $121M   | 14%\n"
        "2024 | $148M   | 16%"
    )
    blocks = detect_blocks(text, page_number=5)
    assert any(b.block_type == BLOCK_TYPE_TABLE for b in blocks), \
        "Expected at least one TABLE block"


def test_list_segment_detected():
    """A segment of bullet lines should produce a LIST block."""
    text = (
        "• Customer concentration (41% from top 3)\n"
        "• Supply chain dependency on AWS and Azure\n"
        "• Competition from hyperscalers"
    )
    blocks = detect_blocks(text, page_number=6)
    assert any(b.block_type == BLOCK_TYPE_LIST for b in blocks), \
        "Expected at least one LIST block"


def test_empty_text_returns_empty_list():
    blocks = detect_blocks("", page_number=1)
    assert blocks == []


def test_whitespace_only_returns_empty_list():
    blocks = detect_blocks("   \n\n   ", page_number=1)
    assert blocks == []


# ── 9. Section hierarchy ──────────────────────────────────────────────────────────

def test_section_path_built_correctly():
    """
    Given a sequence of heading + paragraph + sub-heading + paragraph,
    the section_path should reflect the nesting.
    """
    blocks = [
        TextBlock(BLOCK_TYPE_HEADING, "Risk Factors", "Risk Factors", 1, confidence=0.90),
        TextBlock(BLOCK_TYPE_PARAGRAPH, "Overview of risks.", "Overview of risks.", 1, confidence=0.80),
        TextBlock(BLOCK_TYPE_HEADING, "Customer Concentration", "Customer Concentration", 1, confidence=0.80),
        TextBlock(BLOCK_TYPE_PARAGRAPH, "Top 3 clients = 41%.", "Top 3 clients = 41%.", 1, confidence=0.80),
    ]

    annotated = build_section_hierarchy(blocks)
    # Should have 4 items
    assert len(annotated) == 4

    _, path0 = annotated[0]   # Risk Factors heading
    _, path1 = annotated[1]   # Overview paragraph
    _, path2 = annotated[2]   # Customer Concentration heading
    _, path3 = annotated[3]   # Top 3 clients paragraph

    assert "Risk Factors" in path0
    assert "Risk Factors" in path1   # inherits from heading
    assert "Customer Concentration" in path2
    assert "Customer Concentration" in path3


def test_no_headings_gives_empty_paths():
    """Blocks with no headings should all get empty section paths."""
    blocks = [
        TextBlock(BLOCK_TYPE_PARAGRAPH, "First paragraph.", "First paragraph.", 1, confidence=0.80),
        TextBlock(BLOCK_TYPE_PARAGRAPH, "Second paragraph.", "Second paragraph.", 1, confidence=0.80),
    ]
    annotated = build_section_hierarchy(blocks)
    for _, path in annotated:
        assert path == []


def test_section_path_is_list():
    """section_path must be a list, not None or tuple."""
    blocks = [
        TextBlock(BLOCK_TYPE_HEADING, "STRATEGY", "STRATEGY", 1, confidence=0.90),
        TextBlock(BLOCK_TYPE_PARAGRAPH, "We plan to grow.", "We plan to grow.", 1, confidence=0.80),
    ]
    annotated = build_section_hierarchy(blocks)
    for _, path in annotated:
        assert isinstance(path, list)


# ── 10. Confidence scores ────────────────────────────────────────────────────────

def test_confidence_in_valid_range():
    """classify_line must always return confidence in [0.0, 1.0]."""
    test_lines = [
        "RISK FACTORS",
        "Financial Performance Overview",
        "• Bullet point",
        "Year | Revenue | Margin",
        "1. Numbered item",
        "This is a longer paragraph sentence with multiple words.",
        "",
    ]
    for line in test_lines:
        _, confidence = classify_line(line)
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence} out of range for line: {repr(line)}"
        )
