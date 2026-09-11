"""
src/ingestion/models.py  ─  Document representation dataclasses
════════════════════════════════════════════════════════════════
Phase 2: Internal document model that preserves structure.

WHY DATACLASSES?
────────────────
Phase 1 used plain dicts everywhere — fast to write but no type safety.
Dataclasses give us:
  • Named fields (self-documenting)
  • Type annotations (IDE support)
  • Default values
  • repr() for free (useful during debugging)
  • Still convert to dict easily (dataclasses.asdict)

We avoid Pydantic to keep the dependency list minimal. Stdlib dataclasses
are sufficient for the Phase 2 learning goal.

DOCUMENT HIERARCHY
──────────────────
ParsedDocument
    └── ParsedPage (one per PDF page)
            └── TextBlock (one per detected content region)

StructuredChunk
    (produced by chunking.py from the ParsedDocument tree)
    (compatible with Phase 1 retriever — same key names)

PHASE 1 COMPATIBILITY
─────────────────────
StructuredChunk exposes `source` and `page_num` aliases so the
existing retriever.py, generator.py, and app.py work unchanged.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional


# ── Block types ───────────────────────────────────────────────────────────────
# These are the possible structural roles a text region can play.
# We use string literals (not Enum) to keep serialization simple.
BLOCK_TYPE_HEADING   = "heading"
BLOCK_TYPE_PARAGRAPH = "paragraph"
BLOCK_TYPE_TABLE     = "table"
BLOCK_TYPE_LIST      = "list"
BLOCK_TYPE_UNKNOWN   = "unknown"

VALID_BLOCK_TYPES = {
    BLOCK_TYPE_HEADING,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_LIST,
    BLOCK_TYPE_UNKNOWN,
}

# ── Content types for chunks ──────────────────────────────────────────────────
CONTENT_TYPE_TEXT  = "text"
CONTENT_TYPE_TABLE = "table"
CONTENT_TYPE_LIST  = "list"
CONTENT_TYPE_MIXED = "mixed"

VALID_CONTENT_TYPES = {
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_TABLE,
    CONTENT_TYPE_LIST,
    CONTENT_TYPE_MIXED,
}


@dataclass
class TextBlock:
    """
    A single detected content region within a page.

    WHY KEEP raw_text SEPARATE?
    The normalization step transforms the text to remove artifacts.
    We store both so you can compare raw vs. normalized and verify
    the normalization did not accidentally alter meaning.

    WHY confidence?
    Structure detection is heuristic — we're not always sure if
    a short line is a heading or just a short sentence.  confidence
    lets downstream code treat low-confidence blocks with more caution.
    """
    block_type:   str            # BLOCK_TYPE_* constant
    text:         str            # normalized text (post-normalization)
    raw_text:     str            # original text (pre-normalization)
    page_number:  int            # 1-indexed page where this block appears
    confidence:   float = 1.0   # 0.0–1.0 confidence in block_type assignment
    # Optional: table-specific identifier for traceability
    table_id:     Optional[str] = None


@dataclass
class ParsedPage:
    """
    Represents one PDF page after parsing and structure detection.

    WHY store both raw_text and normalized_text at the page level?
    The normalization is applied globally (e.g., repeated header/footer
    detection requires seeing all pages). We store the page-level
    normalized text separately from the block-level normalized text.
    """
    page_number:      int
    document_id:      str
    source_filename:  str
    raw_text:         str
    normalized_text:  str
    blocks:           List[TextBlock] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """
    Top-level document representation.

    WHY title is best-effort?
    PDF metadata is unreliable. We try to extract the title from:
    1. PDF metadata (often blank or generic)
    2. The first detected heading on page 1
    3. The filename (fallback)
    """
    document_id:     str
    source_filename: str
    pages:           List[ParsedPage] = field(default_factory=list)
    title:           str = ""         # best-effort from metadata or first heading

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def all_blocks(self) -> List[TextBlock]:
        """Return all blocks from all pages in document order."""
        blocks = []
        for page in self.pages:
            blocks.extend(page.blocks)
        return blocks


@dataclass
class StructuredChunk:
    """
    A chunk produced by the structure-aware chunker.

    PHASE 1 COMPATIBILITY
    ─────────────────────
    The retriever.py and app.py code accesses:
        chunk["source"]   → mapped from source_filename
        chunk["page_num"] → mapped from page_start
        chunk["text"]     → present directly
        chunk["chunk_id"] → present directly
        chunk["score"]    → attached by retriever at query time

    The Phase 2 chunks.json stores all of these so retriever.py
    works WITHOUT modification.

    SECTION PATH EXAMPLE
    ────────────────────
    If the chunk came from:
        Risk Factors
            → Customer Concentration

    Then:
        section      = "Customer Concentration"
        section_path = ["Risk Factors", "Customer Concentration"]
        parent_section = "Risk Factors"

    This is the structural context that Phase 1 completely loses.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    chunk_id:          str

    # ── Source ────────────────────────────────────────────────────────────────
    document_id:       str
    source_filename:   str

    # ── Location ──────────────────────────────────────────────────────────────
    page_start:        int
    page_end:          int

    # ── Section hierarchy ─────────────────────────────────────────────────────
    section:           str           # immediate heading (empty if no heading found)
    section_path:      List[str]     # full path from document root
    parent_section:    str           # one level up (empty if top-level)

    # ── Content ───────────────────────────────────────────────────────────────
    content_type:      str           # CONTENT_TYPE_* constant
    text:              str

    # ── Size ──────────────────────────────────────────────────────────────────
    chunk_token_count: int

    # ── Document metadata (best-effort) ───────────────────────────────────────
    document_title:    str = ""
    company:           str = ""
    document_type:     str = ""

    # ── Phase 1 compatibility aliases (set in __post_init__) ─────────────────
    source:   str = field(init=False)
    page_num: int = field(init=False)

    def __post_init__(self) -> None:
        # Aliases so Phase 1 code (retriever, app, generator) can access these
        # without knowing about the new field names.
        self.source   = self.source_filename
        self.page_num = self.page_start

    def to_dict(self) -> dict:
        """
        Convert to a plain dict for JSON serialization.
        This is what gets stored in chunks_structure_aware.json.
        All Phase 1 keys are present so retriever.py works unchanged.
        """
        d = dataclasses.asdict(self)
        # Ensure Phase 1 compatibility keys are explicitly present
        d["source"]   = self.source_filename
        d["page_num"] = self.page_start
        return d
