"""
src/ingestion/validation.py  ─  Pre-embedding chunk validation
═══════════════════════════════════════════════════════════════
Phase 2: Validate chunks before they are embedded and indexed.

WHY VALIDATE?
─────────────
Embedding bad data is worse than having fewer chunks:

1. EMPTY CHUNKS: embed as zero-ish vectors → noisy search results
2. DUPLICATES: inflate the index, bias retrieval toward repeated content
3. MISSING METADATA: source/page info needed for citations and debugging
4. IMPOSSIBLE PAGE RANGES: page_start > page_end indicates a parsing bug
5. OVERSIZED CHUNKS: exceed embedding model limits, degrade quality

DESIGN PRINCIPLE
────────────────
Non-critical issues = WARNINGS (pipeline continues, issue is logged).
Critical issues (empty text, missing source) = chunk is REMOVED.
The pipeline never crashes for content-quality issues.

OUTPUT FORMAT
─────────────
validate_chunks() returns:
  - valid_chunks: list of StructuredChunk that passed validation
  - summary: dict with counts and warning messages

The summary is printed to stdout so the user can see what happened.
"""

from __future__ import annotations

import hashlib
from typing import List, Dict, Tuple, Any

from src.ingestion.models import StructuredChunk, VALID_CONTENT_TYPES


# ── Validation thresholds ─────────────────────────────────────────────────────
MIN_CHUNK_TOKENS = 30      # chunks below this are warnings (not removed)
HARD_MAX_TOKENS  = 1000    # chunks above this trigger a warning
MIN_TEXT_LENGTH  = 10      # characters: below this the chunk is dropped


def validate_chunks(
    chunks: List[StructuredChunk],
    hard_max_tokens: int = HARD_MAX_TOKENS,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> Tuple[List[StructuredChunk], Dict[str, Any]]:
    """
    Receives : list of StructuredChunk from the chunker
    Returns  : (valid_chunks, summary_dict)

    Checks performed:
        1. Empty text → remove
        2. Missing source_filename → remove (cannot cite)
        3. Missing document_id → remove
        4. Impossible page range (start > end) → warn, fix
        5. Duplicate exact text → warn, remove duplicate
        6. Very small chunks (< min_chunk_tokens) → warn, keep
        7. Oversized chunks (> hard_max_tokens) → warn, keep
        8. Invalid content_type → warn, normalize to "text"
        9. Missing section (may be legitimate for intro content)→ note
    """
    warnings: List[str] = []
    removed_count   = 0
    warning_count   = 0

    # Track seen text hashes for duplicate detection
    seen_hashes: Dict[str, str] = {}   # hash → chunk_id

    valid_chunks: List[StructuredChunk] = []

    for i, chunk in enumerate(chunks):
        chunk_label = f"Chunk {i} (id={chunk.chunk_id})"
        keep = True

        # ── 1. Empty text ──────────────────────────────────────────────────────
        if not chunk.text or not chunk.text.strip():
            warnings.append(f"{chunk_label}: REMOVED — empty text")
            removed_count += 1
            keep = False

        # ── 2. Text too short ─────────────────────────────────────────────────
        elif len(chunk.text.strip()) < MIN_TEXT_LENGTH:
            warnings.append(
                f"{chunk_label}: REMOVED — text too short "
                f"({len(chunk.text.strip())} chars)"
            )
            removed_count += 1
            keep = False

        if not keep:
            continue

        # ── 3. Missing source_filename ─────────────────────────────────────────
        if not chunk.source_filename:
            warnings.append(f"{chunk_label}: REMOVED — missing source_filename")
            removed_count += 1
            continue

        # ── 4. Missing document_id ────────────────────────────────────────────
        if not chunk.document_id:
            warnings.append(f"{chunk_label}: REMOVED — missing document_id")
            removed_count += 1
            continue

        # ── 5. Impossible page range ──────────────────────────────────────────
        if chunk.page_start > chunk.page_end:
            warnings.append(
                f"{chunk_label}: WARNING — page_start ({chunk.page_start}) > "
                f"page_end ({chunk.page_end}). Swapping."
            )
            import dataclasses
            chunk = dataclasses.replace(
                chunk,
                page_start=chunk.page_end,
                page_end=chunk.page_start,
            )
            # Also update aliases
            object.__setattr__(chunk, "page_num", chunk.page_start)
            warning_count += 1

        # ── 6. Duplicate detection ────────────────────────────────────────────
        text_hash = hashlib.md5(chunk.text.strip().encode()).hexdigest()
        if text_hash in seen_hashes:
            warnings.append(
                f"{chunk_label}: REMOVED — duplicate of {seen_hashes[text_hash]}"
            )
            removed_count += 1
            continue
        seen_hashes[text_hash] = chunk.chunk_id

        # ── 7. Very small chunk ────────────────────────────────────────────────
        if chunk.chunk_token_count < min_chunk_tokens:
            warnings.append(
                f"{chunk_label}: WARNING — small chunk "
                f"({chunk.chunk_token_count} tokens, min={min_chunk_tokens}). Kept."
            )
            warning_count += 1

        # ── 8. Oversized chunk ────────────────────────────────────────────────
        if chunk.chunk_token_count > hard_max_tokens:
            warnings.append(
                f"{chunk_label}: WARNING — oversized chunk "
                f"({chunk.chunk_token_count} tokens, max={hard_max_tokens}). Kept."
            )
            warning_count += 1

        # ── 9. Invalid content_type ───────────────────────────────────────────
        if chunk.content_type not in VALID_CONTENT_TYPES:
            warnings.append(
                f"{chunk_label}: WARNING — invalid content_type "
                f"'{chunk.content_type}'. Normalizing to 'text'."
            )
            import dataclasses
            chunk = dataclasses.replace(chunk, content_type="text")
            warning_count += 1

        # ── 10. Missing section (informational) ───────────────────────────────
        # Not an error — some documents start with content before any heading.
        # We note it for transparency but don't remove the chunk.
        if not chunk.section:
            # Note: not added to warnings list (too noisy), just counted
            pass

        valid_chunks.append(chunk)

    # ── Build summary ──────────────────────────────────────────────────────────
    no_section_count = sum(1 for c in valid_chunks if not c.section)
    table_count      = sum(1 for c in valid_chunks if c.content_type == "table")
    avg_tokens       = (
        sum(c.chunk_token_count for c in valid_chunks) / len(valid_chunks)
        if valid_chunks else 0
    )

    summary: Dict[str, Any] = {
        "input_chunk_count":  len(chunks),
        "valid_chunk_count":  len(valid_chunks),
        "removed_count":      removed_count,
        "warning_count":      warning_count,
        "no_section_count":   no_section_count,
        "table_chunk_count":  table_count,
        "avg_tokens_per_chunk": round(avg_tokens, 1),
        "warnings":           warnings,
    }

    return valid_chunks, summary


def print_validation_summary(summary: Dict[str, Any]) -> None:
    """
    Pretty-print the validation summary to stdout.
    Called by ingest.py after validation completes.
    """
    bar = "-" * 56

    print(f"\n{bar}")
    print("  CHUNK VALIDATION SUMMARY")
    print(bar)
    print(f"  Input chunks    : {summary['input_chunk_count']}")
    print(f"  Valid chunks    : {summary['valid_chunk_count']}")
    print(f"  Removed         : {summary['removed_count']}")
    print(f"  Warnings        : {summary['warning_count']}")
    print(f"  No section      : {summary['no_section_count']}  (content before first heading)")
    print(f"  Table chunks    : {summary['table_chunk_count']}")
    print(f"  Avg tokens/chunk: {summary['avg_tokens_per_chunk']}")

    if summary["warnings"]:
        print(f"\n  Details ({len(summary['warnings'])} issue(s)):")
        for w in summary["warnings"]:
            print(f"    • {w}")

    print(bar)
