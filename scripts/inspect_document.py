"""
scripts/inspect_document.py  -  Document structure inspector
══════════════════════════════════════════════════════════════
Usage:
    python scripts/inspect_document.py data/raw/OrionVault_Systems_Due_Diligence_Dossier.pdf
    python scripts/inspect_document.py data/raw/OrionVault_Systems_Due_Diligence_Dossier.pdf --pages 1 2 3

Shows the parsed structure of a PDF: pages, headings, paragraphs, tables, lists.
This is the primary LEARNING TOOL for understanding how Phase 2 sees documents.

WHY THIS MATTERS
────────────────
Before you run ingestion, you want to know:
  - Did the parser detect the right headings?
  - Are tables preserved as tables or destroyed?
  - Is normalization working (blank lines, artifacts)?
  - What section hierarchy will the chunker see?

This script answers all of those questions without running the full pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding for Unicode output
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.models import (
    BLOCK_TYPE_HEADING,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_LIST,
    BLOCK_TYPE_UNKNOWN,
)
from src.ingestion.structure import build_section_hierarchy


# ── ANSI colors for terminal output ──────────────────────────────────────────
# (safe — just ignored if terminal doesn't support them)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_BLUE   = "\033[94m"
_RED    = "\033[91m"
_DIM    = "\033[2m"

_BLOCK_COLORS = {
    BLOCK_TYPE_HEADING:   _BOLD + _CYAN,
    BLOCK_TYPE_PARAGRAPH: _RESET,
    BLOCK_TYPE_TABLE:     _YELLOW,
    BLOCK_TYPE_LIST:      _GREEN,
    BLOCK_TYPE_UNKNOWN:   _DIM,
}

_BLOCK_LABELS = {
    BLOCK_TYPE_HEADING:   "HEADING",
    BLOCK_TYPE_PARAGRAPH: "PARAGRAPH",
    BLOCK_TYPE_TABLE:     "TABLE",
    BLOCK_TYPE_LIST:      "LIST",
    BLOCK_TYPE_UNKNOWN:   "UNKNOWN",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect the parsed structure of a PDF document.",
    )
    p.add_argument("pdf_path", help="Path to the PDF file to inspect")
    p.add_argument(
        "--pages",
        nargs="*",
        type=int,
        default=None,
        help="Only show specific page numbers (default: all pages)",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    p.add_argument(
        "--show-raw",
        action="store_true",
        help="Also show the raw (pre-normalization) text for each page",
    )
    p.add_argument(
        "--show-section-paths",
        action="store_true",
        help="Show the section path (hierarchy) for each block",
    )
    return p.parse_args()


def _color(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{_RESET}"


def main() -> None:
    args = parse_args()
    use_color = not args.no_color

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  DOCUMENT INSPECTOR  --  Phase 2 Structure View")
    print(f"  File: {pdf_path.name}")
    print(f"{'=' * 70}\n")

    # ── Parse the document ────────────────────────────────────────────────────
    doc = parse_pdf(pdf_path)

    print(f"\n  Document title : {_color(doc.title, _BOLD, use_color)}")
    print(f"  Document ID    : {doc.document_id}")
    print(f"  Total pages    : {doc.total_pages}")

    # Block type summary
    all_blocks = doc.all_blocks
    type_counts = {}
    for b in all_blocks:
        type_counts[b.block_type] = type_counts.get(b.block_type, 0) + 1
    print(f"  Total blocks   : {len(all_blocks)}")
    for btype, count in sorted(type_counts.items()):
        label = _BLOCK_LABELS.get(btype, btype)
        print(f"    {label:12s} : {count}")

    # Section path walk for the whole document
    block_paths = {}
    if args.show_section_paths:
        annotated = build_section_hierarchy(all_blocks)
        for block, path in annotated:
            block_paths[id(block)] = path

    # ── Page-by-page output ───────────────────────────────────────────────────
    page_filter = set(args.pages) if args.pages else None

    for page in doc.pages:
        if page_filter and page.page_number not in page_filter:
            continue

        print(f"\n{'-' * 70}")
        print(
            _color(
                f"  PAGE {page.page_number}  "
                f"({len(page.blocks)} blocks)",
                _BOLD + _BLUE,
                use_color,
            )
        )
        print(f"{'-' * 70}\n")

        if args.show_raw:
            print(_color("  [RAW TEXT]", _DIM, use_color))
            for line in page.raw_text.splitlines()[:8]:
                print(_color(f"  │  {line}", _DIM, use_color))
            print(_color("  │  ...", _DIM, use_color))
            print()

        if not page.blocks:
            print(_color("  (no blocks detected on this page)", _DIM, use_color))
            continue

        for block in page.blocks:
            color   = _BLOCK_COLORS.get(block.block_type, _RESET)
            label   = _BLOCK_LABELS.get(block.block_type, "?")
            conf    = f"{block.confidence:.2f}"

            # Block header line
            header = f"  [{label}]  (confidence: {conf})"
            if block.table_id:
                header += f"  [table_id: {block.table_id}]"
            print(_color(header, color, use_color))

            # Section path if requested
            if args.show_section_paths and id(block) in block_paths:
                path = block_paths[id(block)]
                if path:
                    path_str = " > ".join(path)
                    print(_color(f"  │ section_path: {path_str}", _DIM, use_color))

            # Block text — truncated for readability
            text_preview = block.text.strip()
            lines = text_preview.split("\n")
            MAX_LINES = 6
            if len(lines) > MAX_LINES:
                shown = "\n".join(lines[:MAX_LINES])
                remaining = len(lines) - MAX_LINES
                print(f"  {shown}")
                print(_color(f"  ... ({remaining} more lines)", _DIM, use_color))
            else:
                print(f"  {text_preview}")

            print()

    # ── Section hierarchy summary ─────────────────────────────────────────────
    print(f"\n{'-' * 70}")
    print(_color("  SECTION HIERARCHY (headings only)", _BOLD, use_color))
    print(f"{'-' * 70}\n")

    headings = [
        b for b in all_blocks
        if b.block_type == BLOCK_TYPE_HEADING and b.confidence >= 0.65
    ]
    if not headings:
        print("  (no headings detected)\n")
    else:
        annotated = build_section_hierarchy(all_blocks)
        for block, path in annotated:
            if block.block_type != BLOCK_TYPE_HEADING or block.confidence < 0.65:
                continue
            indent = "  " * max(0, len(path) - 1)
            marker = "→" if len(path) > 1 else "•"
            print(f"  {indent}{marker}  {block.text.strip()}")
        print()

    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
