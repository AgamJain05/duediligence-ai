"""
src/ingestion/chunking.py  ─  Structure-aware chunker
══════════════════════════════════════════════════════
Phase 2: Replace fixed-size chunking with structure-preserving chunking.

FUNDAMENTAL PROBLEM WITH FIXED-SIZE CHUNKING
─────────────────────────────────────────────
Given this document structure:

    [HEADING]  Risk Factors
    [HEADING]  Customer Concentration
    [PARA]     OrionVault's top 3 customers represent 41% of revenue...
    [PARA]     This concentration creates material risk because...

Phase 1 (fixed-size, 500 tokens) might produce:

    Chunk 47:  "...growth rate. Risk Factors Customer"
    Chunk 48:  "Concentration OrionVault's top 3 customers..."

Problems:
  1. The heading "Risk Factors" is split from its content.
  2. The sub-heading "Customer Concentration" loses its parent context.
  3. A query for "customer concentration risk" may match Chunk 48 but
     the chunk doesn't know it's about Risk Factors — it lost that context.

Phase 2 produces instead:

    Chunk A:
        section_path = ["Risk Factors", "Customer Concentration"]
        text = "Customer Concentration\n\nOrionVault's top 3 customers represent
                41% of revenue...\n\nThis concentration creates material risk..."

HOW THE CHUNKER WORKS
──────────────────────
1. Walk ALL blocks from ParsedDocument in page-document order.
2. Maintain a section stack (heading hierarchy).
3. Accumulate blocks into the current chunk.
4. When accumulated tokens ≥ TARGET_CHUNK_TOKENS:
   a. Seal the current chunk.
   b. Start a new chunk that BEGINS with the section path context.
5. Tables are kept whole (not split) unless > HARD_MAX_TOKENS.
6. After all blocks: seal the last chunk.
7. Drop any chunks below MIN_CHUNK_TOKENS (merge into next).

SECTION CONTEXT PRESERVATION
──────────────────────────────
When a section must be split across multiple chunks, EVERY child chunk
starts with a breadcrumb so it always carries the structural context:

    [Section: Risk Factors > Customer Concentration]
    <content continued...>

This means the LLM and retriever always know WHERE in the document
a chunk comes from, even if the section heading was on a previous page.

TOKEN COUNTING
──────────────
We use the same tiktoken cl100k_base tokenizer as Phase 1 for
consistency. This means token counts are directly comparable.

CONFIGURABLE SIZES
──────────────────
TARGET_CHUNK_TOKENS   = 500   (same default as Phase 1 for fair comparison)
CHUNK_OVERLAP_TOKENS  = 0     (overlap is less important with semantic boundaries)
MIN_CHUNK_TOKENS      = 50    (tiny chunks are merged or dropped)
HARD_MAX_TOKENS       = 1000  (absolute maximum — triggers a warning)
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

import tiktoken

from src.ingestion.models import (
    ParsedDocument,
    TextBlock,
    StructuredChunk,
    BLOCK_TYPE_HEADING,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_LIST,
    BLOCK_TYPE_PARAGRAPH,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_TABLE,
    CONTENT_TYPE_LIST,
    CONTENT_TYPE_MIXED,
)
from src.ingestion.structure import build_section_hierarchy


# ── Configurable defaults ─────────────────────────────────────────────────────
TARGET_CHUNK_TOKENS  = 500
MIN_CHUNK_TOKENS     = 50
HARD_MAX_TOKENS      = 1000
TOKENIZER_ENCODING   = "cl100k_base"   # same as Phase 1


def _get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TOKENIZER_ENCODING)


def _count_tokens(text: str) -> int:
    enc = _get_tokenizer()
    return len(enc.encode(text))


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Deterministic chunk ID based on document ID + chunk index.
    12-character hex — same length as Phase 1 for consistency.
    """
    raw = f"sa::{document_id}::chunk{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _section_path_prefix(section_path: List[str]) -> str:
    """
    Build the breadcrumb prefix that starts continuation chunks.

    Example: ["Risk Factors", "Supply Chain"]
    → "[Section: Risk Factors > Supply Chain]\n\n"

    This text is prepended to continuation chunks so they always
    carry their structural context without needing the retriever
    to join chunks across boundaries.
    """
    if not section_path:
        return ""
    path_str = " > ".join(section_path)
    return f"[Section: {path_str}]\n\n"


def _infer_content_type(blocks: List[TextBlock]) -> str:
    """
    Determine the dominant content type for a group of blocks.
    A chunk with only tables → "table"
    A chunk with only lists → "list"
    A chunk with only text/headings → "text"
    A chunk with mixed types → "mixed"
    """
    types = {b.block_type for b in blocks}
    if BLOCK_TYPE_TABLE in types and len(types) == 1:
        return CONTENT_TYPE_TABLE
    if BLOCK_TYPE_LIST in types and len(types) <= 2:
        return CONTENT_TYPE_LIST
    if types == {BLOCK_TYPE_PARAGRAPH} or types == {BLOCK_TYPE_HEADING} or \
            types <= {BLOCK_TYPE_HEADING, BLOCK_TYPE_PARAGRAPH}:
        return CONTENT_TYPE_TEXT
    return CONTENT_TYPE_MIXED


def build_structured_chunks(
    document: ParsedDocument,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
    hard_max_tokens: int = HARD_MAX_TOKENS,
) -> List[StructuredChunk]:
    """
    Receives : ParsedDocument with structure-detected blocks
               target_tokens  — aim for chunks of this size
               min_tokens     — drop/merge chunks below this size
               hard_max_tokens — warn if a chunk exceeds this
    Returns  : list of StructuredChunk objects (validated, ordered)

    This is the main entry point called by ingest.py and compare_chunking.py.
    """
    all_blocks = document.all_blocks

    if not all_blocks:
        print(f"  [chunking] WARNING: Document '{document.source_filename}' has no blocks.")
        return []

    # ── Attach section path to every block ────────────────────────────────────
    block_with_path = build_section_hierarchy(all_blocks)

    # ── Accumulate blocks into chunks ─────────────────────────────────────────
    chunks: List[StructuredChunk] = []
    chunk_index = 0

    # Current accumulator state
    acc_blocks: List[TextBlock]    = []
    acc_path: List[str]            = []
    acc_pages: List[int]           = []
    acc_tokens: int                = 0
    acc_has_prefix: bool           = False   # did we already add a section prefix?

    def _seal_chunk(is_continuation: bool = False) -> Optional[StructuredChunk]:
        """Seal the current accumulator into a StructuredChunk."""
        nonlocal chunk_index

        if not acc_blocks:
            return None

        # Build final text: prefix + block texts joined
        text_parts = []
        if is_continuation and acc_path:
            prefix = _section_path_prefix(acc_path)
            text_parts.append(prefix)
        text_parts.extend(b.text for b in acc_blocks)
        full_text = "\n\n".join(p for p in text_parts if p.strip())

        if not full_text.strip():
            return None

        token_count = _count_tokens(full_text)

        # Determine section info
        section      = acc_path[-1] if acc_path else ""
        parent       = acc_path[-2] if len(acc_path) >= 2 else ""
        content_type = _infer_content_type(acc_blocks)

        page_start = min(acc_pages) if acc_pages else 1
        page_end   = max(acc_pages) if acc_pages else 1

        cid = _make_chunk_id(document.document_id, chunk_index)
        chunk_index += 1

        return StructuredChunk(
            chunk_id=cid,
            document_id=document.document_id,
            source_filename=document.source_filename,
            page_start=page_start,
            page_end=page_end,
            section=section,
            section_path=list(acc_path),
            parent_section=parent,
            content_type=content_type,
            text=full_text,
            chunk_token_count=token_count,
            document_title=document.title,
        )

    def _reset_acc(new_path: List[str]) -> None:
        """Reset accumulator for a new chunk, carrying forward the section path."""
        nonlocal acc_blocks, acc_path, acc_pages, acc_tokens, acc_has_prefix
        acc_blocks    = []
        acc_path      = list(new_path)
        acc_pages     = []
        acc_tokens    = 0
        acc_has_prefix = False

    # ── Main walk ─────────────────────────────────────────────────────────────
    for block, section_path in block_with_path:

        # Update running section path
        acc_path = list(section_path)

        # Skip heading-only blocks from accumulation IF they're trivially short
        # and the next block will carry the same path (avoids orphan heading chunks)
        if block.block_type == BLOCK_TYPE_HEADING:
            block_tokens = _count_tokens(block.text)

            # Headings are always added to the current chunk (to keep heading + content together)
            # They don't trigger a split by themselves
            acc_blocks.append(block)
            acc_pages.append(block.page_number)
            acc_tokens += block_tokens
            continue

        block_tokens = _count_tokens(block.text)

        # ── TABLE: keep whole unless massive ──────────────────────────────────
        if block.block_type == BLOCK_TYPE_TABLE:
            if block_tokens > hard_max_tokens:
                # Very large table: seal what we have, then emit table as its own chunk
                print(
                    f"  [chunking] WARNING: Table on page {block.page_number} "
                    f"is {block_tokens} tokens (> hard max {hard_max_tokens}). "
                    "Keeping as single oversized chunk."
                )

            # If table fits in current chunk, add it
            if acc_tokens + block_tokens <= target_tokens:
                acc_blocks.append(block)
                acc_pages.append(block.page_number)
                acc_tokens += block_tokens
            else:
                # Seal current chunk, then start new chunk with the table
                chunk = _seal_chunk(is_continuation=False)
                if chunk:
                    chunks.append(chunk)
                _reset_acc(acc_path)
                acc_blocks.append(block)
                acc_pages.append(block.page_number)
                acc_tokens += block_tokens
            continue

        # ── REGULAR BLOCK (paragraph, list, unknown) ──────────────────────────
        if acc_tokens + block_tokens <= target_tokens:
            # Block fits: just add it
            acc_blocks.append(block)
            acc_pages.append(block.page_number)
            acc_tokens += block_tokens
        elif block_tokens > target_tokens:
            # Single block is larger than target: seal what we have, then
            # split the oversized block into sub-chunks
            if acc_blocks:
                chunk = _seal_chunk(is_continuation=False)
                if chunk:
                    chunks.append(chunk)

            # Split the oversized block token by token
            sub_chunks = _split_oversized_block(
                block, acc_path, document, chunk_index, target_tokens
            )
            for sc in sub_chunks:
                if sc:
                    chunks.append(sc)
                    chunk_index += 1

            _reset_acc(acc_path)
        else:
            # Current chunk is full: seal it and start new chunk
            chunk = _seal_chunk(is_continuation=False)
            if chunk:
                chunks.append(chunk)

            # New chunk starts as a continuation of same section
            _reset_acc(acc_path)
            acc_blocks.append(block)
            acc_pages.append(block.page_number)
            acc_tokens += block_tokens

    # ── Seal final accumulator ────────────────────────────────────────────────
    final_chunk = _seal_chunk(is_continuation=False)
    if final_chunk:
        chunks.append(final_chunk)

    # ── Post-process: merge tiny chunks ───────────────────────────────────────
    chunks = _merge_tiny_chunks(chunks, min_tokens, target_tokens)

    # ── Warn on oversized chunks ──────────────────────────────────────────────
    oversized = [c for c in chunks if c.chunk_token_count > hard_max_tokens]
    if oversized:
        print(
            f"  [chunking] WARNING: {len(oversized)} chunk(s) exceed "
            f"hard max of {hard_max_tokens} tokens."
        )

    print(
        f"  [chunking] Created {len(chunks)} structure-aware chunks "
        f"from '{document.source_filename}'"
    )
    return chunks


def _split_oversized_block(
    block: TextBlock,
    section_path: List[str],
    document: ParsedDocument,
    start_index: int,
    target_tokens: int,
) -> List[Optional[StructuredChunk]]:
    """
    Split a single block that exceeds target_tokens into multiple chunks.
    Used for very long paragraphs that need to be divided.
    Each sub-chunk carries the section_path context.
    """
    enc    = _get_tokenizer()
    tokens = enc.encode(block.text)
    sub_chunks = []
    sub_idx    = 0
    pos        = 0

    while pos < len(tokens):
        end = min(pos + target_tokens, len(tokens))
        chunk_tokens = tokens[pos:end]
        chunk_text   = enc.decode(chunk_tokens)

        if section_path:
            prefix    = _section_path_prefix(section_path)
            full_text = prefix + chunk_text
        else:
            full_text = chunk_text

        full_text = full_text.strip()
        if not full_text:
            pos = end
            continue

        section      = section_path[-1] if section_path else ""
        parent       = section_path[-2] if len(section_path) >= 2 else ""

        cid = _make_chunk_id(document.document_id, start_index + sub_idx)
        sub_idx += 1

        sub_chunks.append(StructuredChunk(
            chunk_id=cid,
            document_id=document.document_id,
            source_filename=document.source_filename,
            page_start=block.page_number,
            page_end=block.page_number,
            section=section,
            section_path=list(section_path),
            parent_section=parent,
            content_type=CONTENT_TYPE_TEXT,
            text=full_text,
            chunk_token_count=_count_tokens(full_text),
            document_title=document.title,
        ))

        pos = end

    return sub_chunks


def _merge_tiny_chunks(
    chunks: List[StructuredChunk],
    min_tokens: int,
    target_tokens: int,
) -> List[StructuredChunk]:
    """
    Merge chunks below min_tokens into the previous chunk (when possible).

    WHY MERGE?
    Tiny chunks (e.g., a single heading with no content) produce poor
    embeddings and waste index slots. A 10-token chunk "Risk Factors" by itself
    is nearly useless for retrieval. Merged with the following paragraph,
    it becomes useful context.

    Merging strategy: merge tiny chunk into its predecessor IF:
    - The predecessor is from the same source/document
    - The merged size stays under target_tokens
    Otherwise: keep the tiny chunk (better tiny than missing).
    """
    if not chunks:
        return chunks

    merged: List[StructuredChunk] = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i]

        if chunk.chunk_token_count < min_tokens and merged:
            prev = merged[-1]
            combined_text = prev.text + "\n\n" + chunk.text
            combined_tokens = _count_tokens(combined_text)

            if combined_tokens <= target_tokens and prev.source_filename == chunk.source_filename:
                # Merge into previous
                import dataclasses
                merged[-1] = dataclasses.replace(
                    prev,
                    text=combined_text,
                    chunk_token_count=combined_tokens,
                    page_end=max(prev.page_end, chunk.page_end),
                )
                i += 1
                continue

        merged.append(chunk)
        i += 1

    return merged
