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


def load_markdown(file_path: str | Path) -> List[PageDict]:
    """
    Load a Markdown (.md) or plain-text (.txt) file as page-like sections.

    Primary splitting: H2 headings (## ...) — each section becomes one "page".
    This preserves document structure (Risk Factors, Financial Results, etc.)
    and gives the downstream chunker meaningful semantic boundaries.

    Fallback: if the file has no H2 headings, split into virtual pages of
    _MD_LINES_PER_PAGE non-blank lines (robust for flat / unstructured docs).

    In both cases, inline Markdown syntax (**, *, ``, []()) is stripped for
    cleaner chunk text; structural markers (#, |, -) are kept.

    page_num is the 1-indexed section/page number for citation display.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source_name = file_path.name
    raw = file_path.read_text(encoding="utf-8", errors="replace")

    # ── Strip inline Markdown formatting for cleaner chunk text ──────────────
    text_clean = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)        # **bold**
    text_clean = re.sub(r"\*(.+?)\*",     r"\1", text_clean)  # *italic*
    text_clean = re.sub(r"`(.+?)`",        r"\1", text_clean)  # `code`
    text_clean = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text_clean)  # [link](url)
    text_clean = re.sub(r"<!--.*?-->", "", text_clean, flags=re.DOTALL)  # HTML comments

    # ── Determine whether H2 splitting applies ───────────────────────────────
    has_h2 = bool(re.search(r"^## ", text_clean, re.MULTILINE))

    if has_h2:
        # --- H2-based splitting (primary path for structured NexaCore docs) ---
        # Extract H1 title for citation context prefix
        h1_title = ""
        for line in text_clean.splitlines():
            if line.startswith("# ") and not line.startswith("## "):
                h1_title = line.lstrip("# ").strip()
                break

        parts = re.split(r"(?=^## )", text_clean, flags=re.MULTILINE)
        pages: List[PageDict] = []
        section_num = 0

        for part in parts:
            text = part.strip()
            if not text:
                continue
            if h1_title and not text.startswith(f"# {h1_title}"):
                text = f"[Document: {h1_title}]\n\n{text}"
            section_num += 1
            pages.append({"source": source_name, "page_num": section_num, "text": text})

        print(f"  [ingestion] {source_name}: {len(pages)} H2 sections extracted")

    else:
        # --- Line-based splitting (fallback for flat / unstructured docs) ---
        lines = text_clean.splitlines()
        pages = []
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
                    pages.append({"source": source_name, "page_num": page_num, "text": page_text})
                    page_num += 1
                current_lines = []
                non_blank = 0

        if current_lines:
            page_text = "\n".join(current_lines).strip()
            if page_text:
                pages.append({"source": source_name, "page_num": page_num, "text": page_text})

        print(f"  [ingestion] {source_name}: {len(pages)} virtual pages extracted from Markdown")

    return pages


def load_documents_from_directory(directory: str | Path) -> List[PageDict]:
    """
    Load all PDF, Markdown (.md), and plain-text (.txt) files from a directory.

    PDFs  → extracted page-by-page via pypdf (load_pdf)
    .md   → split on H2 headings, fallback to line-based (load_markdown)
    .txt  → same as .md via load_markdown

    Files are sorted alphabetically for deterministic ordering across runs.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    supported = {".pdf", ".md", ".markdown", ".txt"}
    doc_files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    )

    if not doc_files:
        print(f"[WARNING] No supported documents (.pdf / .md / .txt) found in {directory}")
        return []

    print(f"[ingestion] Found {len(doc_files)} document(s) in {directory}")

    all_pages: List[PageDict] = []
    for doc_path in doc_files:
        if doc_path.suffix.lower() == ".pdf":
            all_pages.extend(load_pdf(doc_path))
        else:
            all_pages.extend(load_markdown(doc_path))

    print(
        f"[ingestion] Total sections/pages loaded: {len(all_pages)} "
        f"from {len(doc_files)} document(s)"
    )
    return all_pages
