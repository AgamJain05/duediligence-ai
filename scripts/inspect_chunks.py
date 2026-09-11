"""
scripts/inspect_chunks.py  ─  Chunk store inspector
═════════════════════════════════════════════════════
Usage:
    python scripts/inspect_chunks.py                              # Phase 1 chunks
    python scripts/inspect_chunks.py --strategy structure_aware  # Phase 2 chunks
    python scripts/inspect_chunks.py --strategy structure_aware --limit 10
    python scripts/inspect_chunks.py --strategy structure_aware --section "Risk Factors"

Shows all chunks from the index with metadata:
    Chunk ID | Strategy | Section | Page | Tokens | Content Type | Text preview

This is your debugger for the chunking output.
If the chunker is working correctly:
  - Phase 1 chunks will have NO section field (just source + page + text)
  - Phase 2 chunks will have section_path, content_type, document_title
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Fix Windows console encoding
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── ANSI colors ───────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_DIM    = "\033[2m"
_BLUE   = "\033[94m"

_TYPE_COLORS = {
    "table": _YELLOW,
    "list":  _GREEN,
    "text":  _RESET,
    "mixed": _CYAN,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect the chunk store created by ingest.py."
    )
    p.add_argument(
        "--strategy",
        choices=["fixed", "structure_aware"],
        default="fixed",
        help="Which chunk store to inspect (default: fixed / Phase 1)",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory containing chunk store files (default: {DEFAULT_INDEX_DIR})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show only the first N chunks",
    )
    p.add_argument(
        "--section",
        type=str,
        default=None,
        help="Filter: only show chunks where section contains this string",
    )
    p.add_argument(
        "--content-type",
        choices=["text", "table", "list", "mixed"],
        default=None,
        help="Filter: only show chunks with this content_type (Phase 2 only)",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    p.add_argument(
        "--text-length",
        type=int,
        default=300,
        help="Characters of text to show per chunk preview (default: 300)",
    )
    return p.parse_args()


def _color(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{_RESET}"


def _load_chunks(index_dir: Path, strategy: str) -> list:
    """Load the appropriate chunks file based on strategy."""
    if strategy == "fixed":
        path = index_dir / "chunks.json"
    else:
        path = index_dir / "chunks_structure_aware.json"

    if not path.exists():
        print(f"[ERROR] Chunk store not found: {path}")
        print(f"  → Run first: python scripts/ingest.py --strategy {strategy}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    use_color = not args.no_color

    chunks = _load_chunks(args.index_dir, args.strategy)

    # ── Filters ───────────────────────────────────────────────────────────────
    filtered = chunks

    if args.section:
        filtered = [
            c for c in filtered
            if args.section.lower() in (c.get("section", "") or "").lower()
            or any(
                args.section.lower() in seg.lower()
                for seg in (c.get("section_path") or [])
            )
        ]

    if args.content_type:
        filtered = [
            c for c in filtered
            if c.get("content_type") == args.content_type
        ]

    if args.limit:
        filtered = filtered[:args.limit]

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  CHUNK INSPECTOR  —  Strategy: {args.strategy.upper()}")
    print(f"  Total chunks in store : {len(chunks)}")
    print(f"  Showing               : {len(filtered)}")
    if args.section:
        print(f"  Section filter        : '{args.section}'")
    if args.content_type:
        print(f"  Content-type filter   : {args.content_type}")
    print(f"{'═' * 72}\n")

    # ── Chunk summary statistics (Phase 2 only) ───────────────────────────────
    if args.strategy == "structure_aware" and chunks:
        token_counts = [c.get("chunk_token_count", 0) for c in chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        content_types = {}
        for c in chunks:
            ct = c.get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1

        sections_with_content = sum(1 for c in chunks if c.get("section"))

        print(f"  {'STATISTICS':─<60}")
        print(f"  Avg tokens per chunk  : {avg_tokens:.0f}")
        print(f"  Chunks with section   : {sections_with_content} / {len(chunks)}")
        print(f"  Content type breakdown:")
        for ct, count in sorted(content_types.items()):
            print(f"    {ct:8s} : {count}")
        print()

    # ── Per-chunk display ─────────────────────────────────────────────────────
    for i, chunk in enumerate(filtered, start=1):
        chunk_id   = chunk.get("chunk_id", "?")
        source     = chunk.get("source_filename") or chunk.get("source", "?")
        page_start = chunk.get("page_start") or chunk.get("page_num", "?")
        page_end   = chunk.get("page_end", page_start)
        text       = chunk.get("text", "")
        ctype      = chunk.get("content_type", "text")
        section    = chunk.get("section", "")
        section_path = chunk.get("section_path") or []
        tokens     = chunk.get("chunk_token_count", "?")

        # Format page range
        if page_start == page_end:
            page_str = f"p{page_start}"
        else:
            page_str = f"p{page_start}–{page_end}"

        # Color by content type
        type_color = _TYPE_COLORS.get(ctype, _RESET)

        # Chunk header
        print(f"{'─' * 72}")
        print(
            _color(f"  [{i:3d}]  ID: {chunk_id}", _BOLD, use_color) +
            f"  |  {page_str}  |  " +
            _color(f"{ctype:6s}", type_color, use_color) +
            f"  |  {tokens} tok  |  {source}"
        )

        # Section path (Phase 2)
        if section_path:
            path_str = " > ".join(section_path)
            print(_color(f"         Section: {path_str}", _CYAN, use_color))
        elif section:
            print(_color(f"         Section: {section}", _CYAN, use_color))

        # Text preview
        preview = text.strip()[:args.text_length].replace("\n", " ↵ ")
        ellipsis = "..." if len(text.strip()) > args.text_length else ""
        print(f"         {preview}{ellipsis}")
        print()

    print(f"{'═' * 72}\n")
    print(f"  Showed {len(filtered)} of {len(chunks)} chunks.")
    if args.strategy == "fixed":
        print("  Note: Phase 1 chunks have no section/content_type metadata.")
        print("  Run: python scripts/inspect_chunks.py --strategy structure_aware")
    print()


if __name__ == "__main__":
    main()
