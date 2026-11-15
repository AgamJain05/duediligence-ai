"""
ingestion.py  ─  Document loading and text extraction
═══════════════════════════════════════════════════════
Phase 1: PDF and Markdown ingestion.
- No OCR (image-only PDF pages are skipped)
- No complex table reconstruction
- No sophisticated metadata system
- Text extracted page-by-page (or virtual-page for .md) to preserve
  page numbers for citation

Supported formats:
  .pdf  — text-based PDFs via pypdf
  .md   — Markdown files, split into virtual pages of ~60 non-blank lines

Pipeline position:
    [PDF / .md files]  →  load_documents_from_directory()
                       →  [list of page dicts]  →  chunking.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict

import pypdf


# ──────────────────────────────────────────────────────────────────────────────
# Type alias for clarity
# A "Page" is just a plain dict so every part of the pipeline can inspect it.
# We intentionally avoid custom classes in Phase 1 to keep the code readable.
# ──────────────────────────────────────────────────────────────────────────────
PageDict = Dict[str, object]   # {source: str, page_num: int, text: str}

# How many non-blank lines to group into one virtual "page" for Markdown files
_MD_LINES_PER_PAGE = 60


def load_pdf(file_path: str | Path) -> List[PageDict]:
    """
    Receives : path to a single PDF file (str or Path)
    Returns  : list of page dicts  [{source, page_num, text}, ...]

    Why it exists: RAG needs plain text.  This is Step 1 of the pipeline.
    We iterate page-by-page so that each extracted unit carries a page number.
    Page numbers are preserved all the way to the final answer so the LLM
    can cite them (e.g. "See OrionVault_Dossier.pdf, page 4").

    Blank pages (no extractable text) are silently skipped.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    source_name = file_path.name   # just the filename, not the full path
    pages: List[PageDict] = []

    with open(file_path, "rb") as fh:
        reader = pypdf.PdfReader(fh)
        total_pages = len(reader.pages)

        for page_idx, page_obj in enumerate(reader.pages):
            page_num = page_idx + 1          # 1-indexed, human-friendly
            raw_text = page_obj.extract_text() or ""
            text = raw_text.strip()

            if not text:
                # Blank or image-only page — skip rather than store empty strings
                continue

            pages.append(
                {
                    "source": source_name,
                    "page_num": page_num,
                    "text": text,
                }
            )

    print(
        f"  [ingestion] {source_name}: "
        f"{len(pages)} text pages extracted (of {total_pages} total)"
    )
    return pages


def load_md(file_path: str | Path) -> List[PageDict]:
    """
    Receives : path to a Markdown (.md) file
    Returns  : list of page dicts [{source, page_num, text}, ...]

    Markdown files have no page boundaries.  We split them into virtual
    pages of _MD_LINES_PER_PAGE non-blank lines so that the rest of the
    Phase 1 pipeline (chunker, retriever, citation) works unchanged.

    Inline Markdown syntax (**, *, ``, []()) is stripped for cleaner text,
    but structural markers (#, |, -) are kept so the downstream chunker
    has signal for structure-aware splitting.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    if file_path.suffix.lower() not in (".md", ".markdown"):
        raise ValueError(f"Expected a .md file, got: {file_path.suffix}")

    source_name = file_path.name
    raw_content = file_path.read_text(encoding="utf-8", errors="replace")

    # Strip inline formatting for cleaner chunk text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", raw_content)   # **bold**
    text = re.sub(r"\*(.+?)\*",     r"\1", text)            # *italic*
    text = re.sub(r"`(.+?)`",        r"\1", text)            # `code`
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)         # [link](url)

    lines = text.splitlines()

    # Group into virtual pages of _MD_LINES_PER_PAGE non-blank lines each
    pages: List[PageDict] = []
    current_lines: List[str] = []
    non_blank = 0
    page_num = 1

    for line in lines:
        current_lines.append(line)
        if line.strip():
            non_blank += 1
        if non_blank >= _MD_LINES_PER_PAGE:
            page_text = "\n".join(current_lines).strip()
            if page_text:
                pages.append({
                    "source": source_name,
                    "page_num": page_num,
                    "text": page_text,
                })
                page_num += 1
            current_lines = []
            non_blank = 0

    # Flush any remaining lines
    if current_lines:
        page_text = "\n".join(current_lines).strip()
        if page_text:
            pages.append({
                "source": source_name,
                "page_num": page_num,
                "text": page_text,
            })

    print(
        f"  [ingestion] {source_name}: "
        f"{len(pages)} virtual pages extracted from Markdown"
    )
    return pages


def load_documents_from_directory(directory: str | Path) -> List[PageDict]:
    """
    Receives : path to a directory containing PDF and/or Markdown files
    Returns  : combined list of page dicts from ALL supported files

    Supported: .pdf (text-based), .md / .markdown
    Image-only PDF pages and empty Markdown files are silently skipped.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    md_files  = sorted(list(directory.glob("*.md")) + list(directory.glob("*.markdown")))
    all_files = pdf_files + md_files

    if not all_files:
        print(f"[WARNING] No PDF or Markdown files found in {directory}")
        return []

    print(f"[ingestion] Found {len(pdf_files)} PDF(s) and {len(md_files)} Markdown file(s) in {directory}")

    all_pages: List[PageDict] = []
    for path in sorted(all_files):
        if path.suffix.lower() == ".pdf":
            pages = load_pdf(path)
        else:
            pages = load_md(path)
        all_pages.extend(pages)

    print(
        f"[ingestion] Total pages loaded: {len(all_pages)} "
        f"from {len(all_files)} document(s)"
    )
    return all_pages
