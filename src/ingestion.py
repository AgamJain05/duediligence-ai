"""
ingestion.py  ─  Document loading and text extraction
═══════════════════════════════════════════════════════
Phase 1: Basic PDF ingestion only.
- No OCR (images are skipped)
- No complex table reconstruction
- No sophisticated metadata system
- Text extracted page-by-page to preserve page numbers for citation

Pipeline position:
    [PDF files]  →  load_pdf()  →  [list of page dicts]  →  chunking.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import pypdf


# ──────────────────────────────────────────────────────────────────────────────
# Type alias for clarity
# A "Page" is just a plain dict so every part of the pipeline can inspect it.
# We intentionally avoid custom classes in Phase 1 to keep the code readable.
# ──────────────────────────────────────────────────────────────────────────────
PageDict = Dict[str, object]   # {source: str, page_num: int, text: str}


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


def load_documents_from_directory(directory: str | Path) -> List[PageDict]:
    """
    Receives : path to a directory containing PDF files
    Returns  : combined list of page dicts from ALL PDFs in that directory

    Why it exists: Convenience wrapper so the ingestion script can point at
    a folder rather than list individual files.  In Phase 1 only .pdf files
    are processed; other file types are quietly ignored.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))   # sorted for determinism

    if not pdf_files:
        print(f"[WARNING] No PDF files found in {directory}")
        return []

    print(f"[ingestion] Found {len(pdf_files)} PDF(s) in {directory}")

    all_pages: List[PageDict] = []
    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    print(
        f"[ingestion] Total pages loaded: {len(all_pages)} "
        f"from {len(pdf_files)} document(s)"
    )
    return all_pages
