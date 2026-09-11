"""
tests/test_parsing.py  ─  Unit tests for pdf_parser.py
═══════════════════════════════════════════════════════
Tests:
  1. ParsedDocument is returned with correct type
  2. Page numbers are correct (1-indexed)
  3. source_filename and document_id are populated
  4. Blank/empty pages do not crash the parser
  5. ParsedPage.blocks is a list
  6. All blocks have required fields

NOTE: These tests do NOT call the real PDF parser on disk — they test
      the parsing pipeline using synthetic text injected directly into
      the structure-detection functions. This avoids file I/O in tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.ingestion.models import (
    ParsedDocument,
    ParsedPage,
    TextBlock,
    BLOCK_TYPE_HEADING,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_LIST,
    BLOCK_TYPE_UNKNOWN,
    VALID_BLOCK_TYPES,
)
from src.ingestion.structure import detect_blocks
from src.ingestion.normalization import normalize_text


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_page(text: str, page_num: int = 1) -> ParsedPage:
    """Create a ParsedPage with structure detection applied — no file I/O."""
    normalized = normalize_text(text)
    blocks     = detect_blocks(normalized, page_num)
    return ParsedPage(
        page_number=page_num,
        document_id="test_doc_001",
        source_filename="test.pdf",
        raw_text=text,
        normalized_text=normalized,
        blocks=blocks,
    )


def _make_document(pages_text: list[str]) -> ParsedDocument:
    """Create a ParsedDocument from a list of page texts."""
    pages = [_make_page(text, i + 1) for i, text in enumerate(pages_text)]
    doc = ParsedDocument(
        document_id="test_doc_001",
        source_filename="test.pdf",
        title="Test Document",
        pages=pages,
    )
    return doc


# ── 1. ParsedDocument type and structure ────────────────────────────────────────

def test_parsed_document_type():
    """parse_pdf returns a ParsedDocument (tested here via _make_document)."""
    doc = _make_document(["Some text on page 1."])
    assert isinstance(doc, ParsedDocument)


def test_parsed_document_has_pages():
    doc = _make_document(["Page one text.", "Page two text."])
    assert len(doc.pages) == 2


def test_parsed_document_fields_populated():
    doc = _make_document(["OrionVault Systems overview."])
    assert doc.document_id == "test_doc_001"
    assert doc.source_filename == "test.pdf"
    assert doc.title == "Test Document"


# ── 2. Page numbers ─────────────────────────────────────────────────────────────

def test_page_numbers_are_one_indexed():
    """Pages should be numbered starting at 1."""
    doc = _make_document(["First page.", "Second page.", "Third page."])
    page_nums = [p.page_number for p in doc.pages]
    assert page_nums == [1, 2, 3]


def test_page_number_matches_block_page():
    """Every block in a page should carry that page's page_number."""
    page = _make_page(
        "Financial Performance\n\nRevenue grew 22% in FY2024.",
        page_num=4,
    )
    for block in page.blocks:
        assert block.page_number == 4


# ── 3. source_filename and document_id ──────────────────────────────────────────

def test_parsed_page_source_filename():
    page = _make_page("Some content.")
    assert page.source_filename == "test.pdf"


def test_parsed_page_document_id():
    page = _make_page("Some content.")
    assert page.document_id == "test_doc_001"


# ── 4. Empty / blank pages ───────────────────────────────────────────────────────

def test_blank_page_produces_no_blocks():
    """An empty page should produce zero blocks (not crash)."""
    page = _make_page("", page_num=3)
    assert page.blocks == []


def test_whitespace_only_page_produces_no_blocks():
    """A page with only whitespace should produce zero blocks."""
    page = _make_page("   \n\n   \t\n", page_num=2)
    assert page.blocks == []


def test_empty_document_is_valid():
    """A document with no pages should still be a ParsedDocument."""
    doc = ParsedDocument(
        document_id="empty",
        source_filename="empty.pdf",
        title="Empty",
    )
    assert doc.total_pages == 0
    assert doc.all_blocks == []


# ── 5. ParsedPage.blocks ─────────────────────────────────────────────────────────

def test_parsed_page_blocks_is_list():
    page = _make_page("OrionVault Systems is a company.\n\nWe serve financial clients.")
    assert isinstance(page.blocks, list)


def test_non_empty_page_has_at_least_one_block():
    page = _make_page("OrionVault Systems overview paragraph.")
    assert len(page.blocks) >= 1


# ── 6. All blocks have required fields ─────────────────────────────────────────

def test_blocks_have_required_fields():
    """Every TextBlock must have block_type, text, raw_text, page_number, confidence."""
    page = _make_page(
        "RISK FACTORS\n\nThe company faces supply chain risks.",
        page_num=5,
    )
    for block in page.blocks:
        assert hasattr(block, "block_type"),   "block missing block_type"
        assert hasattr(block, "text"),          "block missing text"
        assert hasattr(block, "raw_text"),      "block missing raw_text"
        assert hasattr(block, "page_number"),   "block missing page_number"
        assert hasattr(block, "confidence"),    "block missing confidence"


def test_block_types_are_valid():
    """All detected block types must be one of the defined constants."""
    page = _make_page(
        "FINANCIAL PERFORMANCE\n\nRevenue: $148M\n\n• Customer A\n• Customer B",
        page_num=2,
    )
    for block in page.blocks:
        assert block.block_type in VALID_BLOCK_TYPES, (
            f"Unexpected block_type: '{block.block_type}'"
        )


def test_block_confidence_is_in_range():
    """Block confidence must be between 0.0 and 1.0."""
    page = _make_page(
        "EXECUTIVE SUMMARY\n\nOrionVault provides cloud-native data management.",
        page_num=1,
    )
    for block in page.blocks:
        assert 0.0 <= block.confidence <= 1.0, (
            f"Confidence {block.confidence} out of [0, 1] for block: {block.text!r}"
        )


# ── 7. all_blocks property ─────────────────────────────────────────────────────

def test_all_blocks_returns_correct_count():
    """ParsedDocument.all_blocks should return blocks from all pages."""
    doc = _make_document([
        "HEADING ONE\n\nParagraph one.",
        "HEADING TWO\n\nParagraph two.",
    ])
    total = sum(len(p.blocks) for p in doc.pages)
    assert len(doc.all_blocks) == total


def test_all_blocks_order_is_document_order():
    """Blocks should come back in page 1, page 2, ... order."""
    doc = _make_document([
        "ALPHA SECTION\n\nAlpha content.",
        "BETA SECTION\n\nBeta content.",
    ])
    page_nums = [b.page_number for b in doc.all_blocks]
    assert page_nums == sorted(page_nums), "Blocks not in page order"
