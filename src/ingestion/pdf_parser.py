"""
src/ingestion/pdf_parser.py  ─  Phase 2 PDF parsing
════════════════════════════════════════════════════
Produces a ParsedDocument tree with structure-detected blocks.

WHY pdfplumber INSTEAD OF pypdf?
──────────────────────────────────
pypdf (Phase 1) just gives us a text blob per page.
pdfplumber gives us:
  • Character-level data: each character's x/y position and font size
  • Native table detection: pdfplumber can detect table grid lines
  • Better text ordering for multi-column layouts

We use pdfplumber for extraction and pass font-size metadata to
structure.py so heading detection can use the "larger font = heading"
signal.

FALLBACK SAFETY
───────────────
pdfplumber wraps pdfminer. If table extraction fails (which can happen
with complex PDFs), we catch the exception and preserve the raw text
as an UNKNOWN block rather than silently losing it.

KNOWN LIMITATIONS (documented, not hidden)
──────────────────────────────────────────
1. Scanned PDFs (image-only): pdfplumber gets no text. We skip those pages.
2. Complex multi-column layouts: text order may be wrong.
3. Table detection is layout-based: decorative lines may be mis-detected.
4. Font size is not always reliable: some PDFs use the same size everywhere.

Pipeline position:
    [PDF file] → parse_pdf() → ParsedDocument → normalization → chunking
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple

# pdfplumber is our new extraction library (Phase 2)
# pypdf remains for Phase 1 compatibility
try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False
    print(
        "[pdf_parser] WARNING: pdfplumber not installed. "
        "Run: pip install pdfplumber>=0.11.0\n"
        "  Falling back to pypdf (reduced structure detection quality)."
    )
    import pypdf  # type: ignore

from src.ingestion.models import (
    ParsedDocument,
    ParsedPage,
    TextBlock,
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_UNKNOWN,
)
from src.ingestion.normalization import normalize_pages
from src.ingestion.structure import detect_blocks


def _make_document_id(filepath: Path) -> str:
    """
    Stable document identifier derived from filename.
    We use the filename (not full path) so the ID is portable.
    12-character hex: same approach as Phase 1 chunk IDs.
    """
    return hashlib.md5(filepath.name.encode()).hexdigest()[:12]


def parse_pdf(
    file_path: str | Path,
    strip_headers_footers: bool = True,
) -> ParsedDocument:
    """
    Receives : path to a PDF file
               strip_headers_footers — whether to remove repeated lines
    Returns  : ParsedDocument with structure-detected blocks

    This is the main entry point for Phase 2 ingestion.

    Steps:
        1. Open PDF with pdfplumber (or pypdf fallback)
        2. Extract raw text + font sizes per page
        3. Detect native tables via pdfplumber
        4. Normalize all pages (header/footer removal, whitespace cleanup)
        5. Run structure detection to classify blocks
        6. Return ParsedDocument tree
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"[pdf_parser] PDF not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"[pdf_parser] Expected .pdf, got: {file_path.suffix}")

    doc_id   = _make_document_id(file_path)
    filename = file_path.name

    print(f"[pdf_parser] Parsing: {filename}")

    if _PDFPLUMBER_AVAILABLE:
        return _parse_with_pdfplumber(
            file_path, doc_id, filename, strip_headers_footers
        )
    else:
        return _parse_with_pypdf(
            file_path, doc_id, filename, strip_headers_footers
        )


# ── pdfplumber implementation ─────────────────────────────────────────────────

def _parse_with_pdfplumber(
    file_path: Path,
    doc_id: str,
    filename: str,
    strip_headers_footers: bool,
) -> ParsedDocument:
    """Parse using pdfplumber for richer layout information."""
    raw_texts: List[str] = []           # one entry per page
    font_infos: List[Optional[List[Tuple[str, float]]]] = []  # character font sizes
    table_regions: List[List[dict]] = []  # tables detected by pdfplumber per page

    page_numbers: List[int] = []

    with pdfplumber.open(str(file_path)) as pdf:
        total_pages = len(pdf.pages)
        print(f"  [pdf_parser] Total pages: {total_pages}")

        for page_obj in pdf.pages:
            page_num = page_obj.page_number  # 1-indexed in pdfplumber

            # ── Extract raw text ─────────────────────────────────────────────
            raw_text = page_obj.extract_text(x_tolerance=3, y_tolerance=3) or ""
            if not raw_text.strip():
                print(f"  [pdf_parser] Page {page_num}: no text (skipped)")
                continue

            raw_texts.append(raw_text)
            page_numbers.append(page_num)

            # ── Extract character-level font sizes (for heading detection) ────
            try:
                chars = page_obj.chars
                font_info = [
                    (c.get("text", ""), c.get("size", 0.0))
                    for c in chars
                    if c.get("text", "").strip()
                ]
                font_infos.append(font_info if font_info else None)
            except Exception:
                font_infos.append(None)

            # ── Extract native tables ─────────────────────────────────────────
            try:
                tables = page_obj.extract_tables() or []
                table_regions.append(tables)
            except Exception as exc:
                print(
                    f"  [pdf_parser] Page {page_num}: table extraction failed "
                    f"({type(exc).__name__}). Preserving as raw text."
                )
                table_regions.append([])

    if not raw_texts:
        print(f"  [pdf_parser] WARNING: No text extracted from {filename}")
        return ParsedDocument(
            document_id=doc_id,
            source_filename=filename,
            title=_infer_title(filename),
        )

    # ── Normalize all pages together (for header/footer detection) ────────────
    normalized_pairs = normalize_pages(raw_texts, strip_headers_footers)

    # ── Build ParsedPage objects ──────────────────────────────────────────────
    parsed_pages: List[ParsedPage] = []
    doc_title = ""

    for idx, (raw, normalized) in enumerate(normalized_pairs):
        page_num    = page_numbers[idx]
        font_info   = font_infos[idx] if idx < len(font_infos) else None
        page_tables = table_regions[idx] if idx < len(table_regions) else []

        # ── Detect structure blocks ───────────────────────────────────────────
        blocks = detect_blocks(normalized, page_num, font_info)

        # ── Inject pdfplumber-detected tables ─────────────────────────────────
        # Tables from pdfplumber may not appear in the text extraction cleanly.
        # We generate a formatted table text and insert it as a TABLE block.
        if page_tables:
            table_blocks = _format_tables(page_tables, page_num)
            # Merge: replace any existing TABLE blocks with pdfplumber versions
            # and insert after last non-table block, or just append
            non_table_blocks = [b for b in blocks if b.block_type != BLOCK_TYPE_TABLE]
            blocks = non_table_blocks + table_blocks

        # ── Infer document title from first heading on page 1 ─────────────────
        if not doc_title and idx == 0:
            for block in blocks:
                from src.ingestion.models import BLOCK_TYPE_HEADING
                if block.block_type == BLOCK_TYPE_HEADING and block.confidence >= 0.70:
                    doc_title = block.text.strip()
                    break

        parsed_pages.append(ParsedPage(
            page_number=page_num,
            document_id=doc_id,
            source_filename=filename,
            raw_text=raw,
            normalized_text=normalized,
            blocks=blocks,
        ))

    title = doc_title or _infer_title(filename)
    print(
        f"  [pdf_parser] Parsed: {len(parsed_pages)} pages, "
        f"title='{title}'"
    )

    return ParsedDocument(
        document_id=doc_id,
        source_filename=filename,
        pages=parsed_pages,
        title=title,
    )


def _format_tables(
    tables: List[list],
    page_num: int,
) -> List[TextBlock]:
    """
    Convert pdfplumber table data (list of rows, each row is a list of cells)
    into TextBlock objects with pipe-separated text.

    pdfplumber table format:
        [
            ["Year", "Revenue", "Margin"],  # header row
            ["2023", "$121M",   "14%"],
            ["2024", "$148M",   "16%"],
        ]

    We convert to:
        "Year | Revenue | Margin\n2023 | $121M | 14%\n2024 | $148M | 16%"
    """
    table_blocks: List[TextBlock] = []

    for t_idx, table in enumerate(tables):
        if not table:
            continue

        rows = []
        for row in table:
            if row is None:
                continue
            # Replace None cells with empty string
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            rows.append(" | ".join(cells))

        if not rows:
            continue

        table_text = "\n".join(rows)

        table_blocks.append(TextBlock(
            block_type=BLOCK_TYPE_TABLE,
            text=table_text,
            raw_text=table_text,
            page_number=page_num,
            confidence=0.92,   # pdfplumber native detection → high confidence
            table_id=f"p{page_num}_t{t_idx}",
        ))

    return table_blocks


# ── pypdf fallback ────────────────────────────────────────────────────────────

def _parse_with_pypdf(
    file_path: Path,
    doc_id: str,
    filename: str,
    strip_headers_footers: bool,
) -> ParsedDocument:
    """
    Fallback parser using pypdf when pdfplumber is not installed.
    Produces less accurate structure detection (no font size info,
    no native table detection) but preserves Phase 1 functionality.
    """
    import pypdf  # noqa: F811

    raw_texts: List[str] = []
    page_numbers: List[int] = []

    with open(file_path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        for idx, page_obj in enumerate(reader.pages):
            raw = page_obj.extract_text() or ""
            if raw.strip():
                raw_texts.append(raw)
                page_numbers.append(idx + 1)

    if not raw_texts:
        return ParsedDocument(
            document_id=doc_id,
            source_filename=filename,
            title=_infer_title(filename),
        )

    normalized_pairs = normalize_pages(raw_texts, strip_headers_footers)
    parsed_pages: List[ParsedPage] = []
    doc_title = ""

    for idx, (raw, normalized) in enumerate(normalized_pairs):
        page_num = page_numbers[idx]
        blocks   = detect_blocks(normalized, page_num, font_size_info=None)

        if not doc_title and idx == 0:
            from src.ingestion.models import BLOCK_TYPE_HEADING
            for block in blocks:
                if block.block_type == BLOCK_TYPE_HEADING and block.confidence >= 0.70:
                    doc_title = block.text.strip()
                    break

        parsed_pages.append(ParsedPage(
            page_number=page_num,
            document_id=doc_id,
            source_filename=filename,
            raw_text=raw,
            normalized_text=normalized,
            blocks=blocks,
        ))

    return ParsedDocument(
        document_id=doc_id,
        source_filename=filename,
        pages=parsed_pages,
        title=doc_title or _infer_title(filename),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_title(filename: str) -> str:
    """Best-effort title from filename: remove extension, replace underscores/hyphens."""
    name = Path(filename).stem
    return name.replace("_", " ").replace("-", " ")
