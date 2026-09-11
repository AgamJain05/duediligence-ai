"""
tests/test_metadata.py  ─  Unit tests for chunk metadata and validation
═══════════════════════════════════════════════════════════════════════
Tests:
  1. chunk_id is deterministic (same input → same ID)
  2. chunk_id is unique across different chunks
  3. section_path is a list of strings
  4. content_type is a valid value
  5. page_start ≤ page_end
  6. No duplicate chunk_ids in a full document
  7. Validation removes empty chunks
  8. Validation removes duplicate text
  9. Validation fixes impossible page ranges
 10. Validation summary has correct counts
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
    StructuredChunk,
    BLOCK_TYPE_HEADING,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_LIST,
    VALID_CONTENT_TYPES,
)
from src.ingestion.chunking import build_structured_chunks, _make_chunk_id
from src.ingestion.validation import validate_chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_block(btype: str, text: str, page: int = 1, conf: float = 0.90) -> TextBlock:
    return TextBlock(
        block_type=btype,
        text=text,
        raw_text=text,
        page_number=page,
        confidence=conf,
    )


def _make_doc(blocks: list[TextBlock], doc_id: str = "meta001") -> ParsedDocument:
    page = ParsedPage(
        page_number=1,
        document_id=doc_id,
        source_filename="orionvault.pdf",
        raw_text="",
        normalized_text="",
        blocks=blocks,
    )
    return ParsedDocument(
        document_id=doc_id,
        source_filename="orionvault.pdf",
        title="OrionVault Due Diligence",
        pages=[page],
    )


def _make_chunk(
    chunk_id: str = "abc123",
    text: str = "Some content about OrionVault.",
    section: str = "Risk Factors",
    section_path: list = None,
    content_type: str = "text",
    page_start: int = 1,
    page_end: int = 1,
    token_count: int = 50,
    source_filename: str = "orionvault.pdf",
    document_id: str = "test001",
) -> StructuredChunk:
    """Create a StructuredChunk with sensible defaults for testing validation."""
    return StructuredChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_filename=source_filename,
        page_start=page_start,
        page_end=page_end,
        section=section,
        section_path=section_path or [section] if section else [],
        parent_section="",
        content_type=content_type,
        text=text,
        chunk_token_count=token_count,
        document_title="Test",
    )


# ── 1. chunk_id is deterministic ──────────────────────────────────────────────

def test_chunk_id_is_deterministic():
    """Same document_id + chunk_index must always produce the same chunk_id."""
    id1 = _make_chunk_id("doc_abc", 5)
    id2 = _make_chunk_id("doc_abc", 5)
    assert id1 == id2


def test_chunk_id_different_for_different_index():
    """Different chunk indices for the same document → different IDs."""
    id1 = _make_chunk_id("doc_abc", 0)
    id2 = _make_chunk_id("doc_abc", 1)
    assert id1 != id2


def test_chunk_id_different_for_different_document():
    """Same chunk index but different documents → different IDs."""
    id1 = _make_chunk_id("doc_A", 0)
    id2 = _make_chunk_id("doc_B", 0)
    assert id1 != id2


def test_chunk_id_length():
    """chunk_id should be 12 hex characters."""
    cid = _make_chunk_id("test_doc", 42)
    assert len(cid) == 12
    assert all(c in "0123456789abcdef" for c in cid)


# ── 2. Unique chunk IDs across full document ─────────────────────────────────

def test_no_duplicate_chunk_ids_in_document():
    """A real document should not produce any duplicate chunk IDs."""
    blocks = [
        _make_block(BLOCK_TYPE_HEADING, "Risk Factors"),
        _make_block(BLOCK_TYPE_PARAGRAPH, "Customer concentration is 41%."),
        _make_block(BLOCK_TYPE_HEADING, "Financial Performance"),
        _make_block(BLOCK_TYPE_PARAGRAPH, "Revenue grew 22% to $148M."),
        _make_block(BLOCK_TYPE_HEADING, "Management"),
        _make_block(BLOCK_TYPE_PARAGRAPH, "The CEO has 15 years of experience."),
    ]
    doc    = _make_doc(blocks)
    chunks = build_structured_chunks(doc)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), (
        f"Duplicate chunk IDs found: "
        f"{[cid for cid in ids if ids.count(cid) > 1]}"
    )


# ── 3. section_path ───────────────────────────────────────────────────────────

def test_section_path_is_list():
    chunk = _make_chunk(section_path=["Risk Factors", "Supply Chain"])
    assert isinstance(chunk.section_path, list)


def test_section_path_is_list_of_strings():
    chunk = _make_chunk(section_path=["A", "B", "C"])
    for item in chunk.section_path:
        assert isinstance(item, str)


def test_section_path_can_be_empty_for_intro_content():
    """section_path can be empty for content before the first heading."""
    chunk = _make_chunk(section="", section_path=[])
    assert chunk.section_path == []


# ── 4. content_type validation ─────────────────────────────────────────────────

def test_content_type_is_valid():
    """content_type must be one of the defined constants."""
    for valid_type in VALID_CONTENT_TYPES:
        chunk = _make_chunk(content_type=valid_type)
        assert chunk.content_type in VALID_CONTENT_TYPES


def test_invalid_content_type_normalized_by_validator():
    """validate_chunks should normalize invalid content_type to 'text'."""
    chunk = _make_chunk(content_type="nonsense", chunk_id="bad001")
    valid_chunks, summary = validate_chunks([chunk])
    if valid_chunks:
        assert valid_chunks[0].content_type == "text"
    assert summary["warning_count"] >= 1


# ── 5. page_start ≤ page_end ──────────────────────────────────────────────────

def test_valid_page_range():
    chunk = _make_chunk(page_start=3, page_end=5)
    assert chunk.page_start <= chunk.page_end


def test_impossible_page_range_fixed_by_validator():
    """validate_chunks should swap page_start and page_end when inverted."""
    chunk = _make_chunk(page_start=7, page_end=3, chunk_id="bad002")
    valid_chunks, summary = validate_chunks([chunk])
    assert len(valid_chunks) == 1
    assert valid_chunks[0].page_start <= valid_chunks[0].page_end
    assert summary["warning_count"] >= 1


# ── 6. Validation — empty chunks removed ─────────────────────────────────────

def test_empty_text_chunk_removed():
    """validate_chunks must remove chunks with empty text."""
    empty_chunk  = _make_chunk(text="", chunk_id="empty001")
    normal_chunk = _make_chunk(text="Normal content.", chunk_id="normal001")

    valid, summary = validate_chunks([empty_chunk, normal_chunk])
    assert len(valid) == 1
    assert valid[0].chunk_id == "normal001"
    assert summary["removed_count"] >= 1


def test_whitespace_only_chunk_removed():
    """A chunk with only whitespace should be removed."""
    chunk = _make_chunk(text="   \n\n   ", chunk_id="ws001")
    valid, summary = validate_chunks([chunk])
    assert len(valid) == 0
    assert summary["removed_count"] >= 1


# ── 7. Validation — duplicate detection ───────────────────────────────────────

def test_duplicate_chunks_removed():
    """Chunks with identical text should be deduplicated."""
    text = "OrionVault's revenue was $148M in FY2024."
    chunk1 = _make_chunk(text=text, chunk_id="dup001")
    chunk2 = _make_chunk(text=text, chunk_id="dup002")   # same text, different id

    valid, summary = validate_chunks([chunk1, chunk2])
    assert len(valid) == 1
    assert summary["removed_count"] >= 1


def test_different_text_not_deduplicated():
    """Chunks with different text should both pass through."""
    chunk1 = _make_chunk(text="Revenue was $148M.", chunk_id="a001")
    chunk2 = _make_chunk(text="Revenue grew 22% YoY.", chunk_id="a002")

    valid, summary = validate_chunks([chunk1, chunk2])
    assert len(valid) == 2


# ── 8. Validation — missing source_filename ───────────────────────────────────

def test_missing_source_filename_removed():
    """Chunks with no source_filename must be removed (cannot cite)."""
    chunk = _make_chunk(source_filename="", chunk_id="nosrc001")
    valid, summary = validate_chunks([chunk])
    assert len(valid) == 0
    assert summary["removed_count"] >= 1


# ── 9. Validation summary accuracy ───────────────────────────────────────────

def test_validation_summary_counts():
    """Summary counts must be accurate."""
    chunks = [
        _make_chunk(text="Valid chunk A.", chunk_id="s001"),
        _make_chunk(text="",               chunk_id="s002"),   # empty → removed
        _make_chunk(text="Valid chunk B.", chunk_id="s003"),
    ]
    valid, summary = validate_chunks(chunks)

    assert summary["input_chunk_count"]  == 3
    assert summary["valid_chunk_count"]  == 2
    assert summary["removed_count"]      >= 1


def test_validation_summary_has_required_keys():
    """The summary dict must have all expected keys."""
    required_keys = {
        "input_chunk_count",
        "valid_chunk_count",
        "removed_count",
        "warning_count",
        "no_section_count",
        "table_chunk_count",
        "avg_tokens_per_chunk",
        "warnings",
    }
    _, summary = validate_chunks([_make_chunk()])
    missing = required_keys - set(summary.keys())
    assert not missing, f"Summary missing keys: {missing}"


# ── 10. to_dict() compatibility ──────────────────────────────────────────────

def test_to_dict_has_phase1_keys():
    """to_dict() must include 'source' and 'page_num' for Phase 1 compat."""
    chunk = _make_chunk()
    d = chunk.to_dict()
    assert "source"   in d, "Missing 'source' key (Phase 1 compat)"
    assert "page_num" in d, "Missing 'page_num' key (Phase 1 compat)"
    assert "text"     in d, "Missing 'text' key"
    assert "chunk_id" in d, "Missing 'chunk_id' key"


def test_to_dict_source_equals_source_filename():
    chunk = _make_chunk(source_filename="orionvault.pdf")
    d = chunk.to_dict()
    assert d["source"] == "orionvault.pdf"


def test_to_dict_page_num_equals_page_start():
    chunk = _make_chunk(page_start=4, page_end=5)
    d = chunk.to_dict()
    assert d["page_num"] == 4
