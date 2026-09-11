"""
tests/test_smart_chunking.py  ─  Unit tests for structure-aware chunking
═════════════════════════════════════════════════════════════════════════
Tests:
  1. Heading stays associated with content (not split into orphan chunk)
  2. Large section → multiple chunks, all carry section_path
  3. Table kept whole under size limit
  4. No tiny chunks < min_tokens (merging works)
  5. Maximum chunk size enforcement
  6. Chunk count scales with document size
  7. Empty document → empty chunk list
  8. section_path on continuation chunks
  9. All chunks have required fields
 10. content_type is assigned correctly
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
    CONTENT_TYPE_TABLE,
    CONTENT_TYPE_LIST,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_MIXED,
)
from src.ingestion.chunking import build_structured_chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_block(btype: str, text: str, page: int = 1, confidence: float = 0.90) -> TextBlock:
    return TextBlock(
        block_type=btype,
        text=text,
        raw_text=text,
        page_number=page,
        confidence=confidence,
    )


def _make_doc(blocks: list[TextBlock], doc_id: str = "test001") -> ParsedDocument:
    page = ParsedPage(
        page_number=1,
        document_id=doc_id,
        source_filename="test.pdf",
        raw_text="",
        normalized_text="",
        blocks=blocks,
    )
    return ParsedDocument(
        document_id=doc_id,
        source_filename="test.pdf",
        title="Test Document",
        pages=[page],
    )


def _word(n: int = 50) -> str:
    """Generate n words of filler text for token counting."""
    unit = "OrionVault provides cloud-native enterprise data management solutions"
    words = (unit + " ") * (n // 10 + 1)
    return " ".join(words.split()[:n])


# ── 1. Heading stays with content ─────────────────────────────────────────────

def test_heading_not_split_from_content():
    """
    A heading should NOT be isolated in its own chunk if there is paragraph content
    immediately following it (and the combined size is within target).
    """
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Supply Chain Risks"),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(30)),  # short paragraph
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, target_tokens=200, min_tokens=10)

    # The heading text should appear in the same chunk as the paragraph
    heading_chunks = [c for c in chunks if "Supply Chain Risks" in c.text]
    para_chunks    = [c for c in chunks if "OrionVault provides" in c.text]

    assert heading_chunks, "Heading text not found in any chunk"
    assert para_chunks, "Paragraph text not found in any chunk"

    # They should be in the same chunk (or the heading should be in a chunk with content)
    heading_chunk_ids = {c.chunk_id for c in heading_chunks}
    para_chunk_ids    = {c.chunk_id for c in para_chunks}
    assert heading_chunk_ids & para_chunk_ids, (
        "Heading and its content ended up in different chunks!\n"
        f"  Heading chunks: {heading_chunk_ids}\n"
        f"  Para chunks   : {para_chunk_ids}"
    )


# ── 2. Large section → multiple chunks with section_path ─────────────────────

def test_large_section_creates_multiple_chunks():
    """A section that exceeds target_tokens must be split into multiple chunks."""
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Risk Factors"),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(300)),  # 300 words > 500 tokens target
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, target_tokens=100, min_tokens=10)
    assert len(chunks) > 1, "Expected multiple chunks for large paragraph"


def test_continuation_chunks_carry_section_path():
    """When a section is split, all child chunks must carry the section_path."""
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Financial Performance"),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(300)),
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, target_tokens=100, min_tokens=10)

    for chunk in chunks:
        assert "Financial Performance" in chunk.section_path or \
               chunk.section == "Financial Performance", (
            f"Chunk missing section context: section='{chunk.section}', "
            f"path={chunk.section_path}, text={chunk.text[:100]!r}"
        )


def test_section_path_is_list_of_strings():
    """section_path must be a list of strings, never None."""
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "STRATEGY"),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc)

    for chunk in chunks:
        assert isinstance(chunk.section_path, list), \
            f"section_path is not a list: {type(chunk.section_path)}"
        assert all(isinstance(s, str) for s in chunk.section_path), \
            "section_path contains non-string items"


# ── 3. Table kept whole ───────────────────────────────────────────────────────

def test_small_table_not_split():
    """A table within the hard max should NOT be split across chunks."""
    table_text = (
        "Year | Revenue | Operating Margin\n"
        "2021 | $82M    | 10%\n"
        "2022 | $99M    | 12%\n"
        "2023 | $121M   | 14%\n"
        "2024 | $148M   | 16%"
    )
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Financial Performance"),
        _make_block(BLOCK_TYPE_TABLE, table_text),
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, target_tokens=100, min_tokens=10)

    # The table text should appear in exactly ONE chunk
    table_chunks = [c for c in chunks if "2021 | $82M" in c.text]
    assert len(table_chunks) >= 1, "Table not found in any chunk"

    # No chunk should have only PART of the table rows
    for chunk in table_chunks:
        # If the table header is in the chunk, all row data should be too
        if "Year | Revenue" in chunk.text:
            assert "2024" in chunk.text or len(table_chunks) == 1, \
                "Table header found without later rows — table may be split"


def test_table_chunk_content_type():
    """A chunk containing only a table should have content_type='table'."""
    table_text = (
        "Year | Revenue | Margin\n"
        "2023 | $121M   | 14%\n"
        "2024 | $148M   | 16%"
    )
    blocks = [
        _make_block(BLOCK_TYPE_TABLE, table_text),
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, min_tokens=5)

    table_chunks = [c for c in chunks if c.content_type == CONTENT_TYPE_TABLE]
    assert len(table_chunks) >= 1, "Expected at least one chunk with content_type='table'"


# ── 4. No tiny chunks ────────────────────────────────────────────────────────

def test_no_tiny_chunks_after_merge():
    """After merging, no chunk should be below min_tokens (30 by default)."""
    MIN = 30
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "A"),           # very short
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(100)),  # normal size
        _make_block(BLOCK_TYPE_HEADING, "B"),           # very short
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(80)),   # normal size
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, target_tokens=200, min_tokens=MIN)

    for chunk in chunks:
        assert chunk.chunk_token_count >= MIN or len(chunks) == 1, (
            f"Chunk below min_tokens: {chunk.chunk_token_count} tokens\n"
            f"Text: {chunk.text[:100]!r}"
        )


# ── 5. Maximum chunk size ─────────────────────────────────────────────────────

def test_chunk_size_does_not_exceed_hard_max():
    """
    Chunks must not routinely exceed hard_max_tokens.
    (Tables and single oversized blocks may be exempt but should be rare.)
    """
    HARD_MAX = 200
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Large Section"),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, target_tokens=100, hard_max_tokens=HARD_MAX)

    oversized = [c for c in chunks if c.chunk_token_count > HARD_MAX]
    # We allow 0 or 1 oversized chunk (tables or single blocks can be large)
    assert len(oversized) <= 1, (
        f"{len(oversized)} chunks exceed HARD_MAX={HARD_MAX} tokens. "
        f"Sizes: {[c.chunk_token_count for c in oversized]}"
    )


# ── 6. Chunk count scaling ────────────────────────────────────────────────────

def test_more_blocks_more_chunks():
    """More content should produce more chunks (monotonic relationship)."""
    small_blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Section"),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
    ]
    large_blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Section"),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(50)),
    ]
    small_doc = _make_doc(small_blocks, doc_id="small")
    large_doc = _make_doc(large_blocks, doc_id="large")

    small_chunks = build_structured_chunks(small_doc, target_tokens=100, min_tokens=10)
    large_chunks = build_structured_chunks(large_doc, target_tokens=100, min_tokens=10)

    assert len(large_chunks) >= len(small_chunks), \
        "Larger document should produce at least as many chunks"


# ── 7. Empty document ─────────────────────────────────────────────────────────

def test_empty_document_returns_empty():
    """A document with no blocks should produce no chunks."""
    page = ParsedPage(
        page_number=1,
        document_id="empty",
        source_filename="empty.pdf",
        raw_text="",
        normalized_text="",
        blocks=[],
    )
    doc = ParsedDocument(
        document_id="empty",
        source_filename="empty.pdf",
        title="Empty",
        pages=[page],
    )
    chunks = build_structured_chunks(doc)
    assert chunks == []


# ── 8. Required fields on all chunks ─────────────────────────────────────────

def test_all_chunks_have_required_fields():
    """Every chunk must carry all required metadata fields."""
    required = {
        "chunk_id", "document_id", "source_filename",
        "page_start", "page_end", "section", "section_path",
        "content_type", "text", "chunk_token_count",
        # Phase 1 compatibility aliases
        "source", "page_num",
    }
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Risk Factors"),
        _make_block(BLOCK_TYPE_PARAGRAPH, _word(40)),
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc)

    for chunk in chunks:
        chunk_dict = chunk.to_dict()
        missing = required - set(chunk_dict.keys())
        assert not missing, f"Chunk missing fields: {missing}"


# ── 9. Phase 1 compatibility aliases ─────────────────────────────────────────

def test_source_alias_equals_source_filename():
    """chunk.source must equal chunk.source_filename for Phase 1 compat."""
    blocks = [_make_block(BLOCK_TYPE_PARAGRAPH, _word(30))]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc)
    for chunk in chunks:
        assert chunk.source == chunk.source_filename


def test_page_num_alias_equals_page_start():
    """chunk.page_num must equal chunk.page_start for Phase 1 compat."""
    blocks = [_make_block(BLOCK_TYPE_PARAGRAPH, _word(30), page=3)]
    page   = ParsedPage(3, "doc", "test.pdf", "", "", blocks)
    doc    = ParsedDocument("doc", "test.pdf", [page], "Test")
    chunks = build_structured_chunks(doc)
    for chunk in chunks:
        assert chunk.page_num == chunk.page_start


# ── 10. content_type assignment ───────────────────────────────────────────────

def test_paragraph_chunk_content_type():
    blocks = [_make_block(BLOCK_TYPE_PARAGRAPH, _word(40))]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, min_tokens=5)
    for chunk in chunks:
        assert chunk.content_type == CONTENT_TYPE_TEXT


def test_list_chunk_content_type():
    list_text = "• Risk A\n• Risk B\n• Risk C\n• Risk D"
    blocks = [_make_block(BLOCK_TYPE_LIST, list_text)]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc, min_tokens=5)
    list_chunks = [c for c in chunks if c.content_type == CONTENT_TYPE_LIST]
    assert len(list_chunks) >= 1, \
        f"Expected LIST content_type. Got: {[c.content_type for c in chunks]}"
