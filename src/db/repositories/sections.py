"""
src/db/repositories/sections.py  ─  Section hierarchy persistence
══════════════════════════════════════════════════════════════════════
Phase 4: Persist the Phase 2 section tree into the sections table.

WHY PERSIST SECTIONS?
──────────────────────
Phase 2 extracts a section hierarchy from PDF structure:
    Risk Factors → Cloud Infrastructure
    Risk Factors → Customer Concentration

Phase 4 stores this in the sections table so we can:
  1. Filter retrieval by section (SQL WHERE on section title/path)
  2. Navigate the document structure without re-parsing the PDF
  3. Answer "What sections are in this document version?"
  4. Enable future section-level analytics

The sections are identified by (document_version_id, title, section_order)
for idempotency — re-ingesting the same document creates the same section rows.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from src.db.models import Section


def upsert_section(
    db: Session,
    document_version_id: int,
    title: str,
    section_order: int,
    section_path: Optional[List[str]] = None,
    parent_section_id: Optional[int] = None,
) -> Section:
    """
    Insert or return an existing section for this document version.

    Idempotency key: (document_version_id, section_order, title).
    If the same section is ingested twice, the same row is returned.

    Args:
        db:                  Active SQLAlchemy session
        document_version_id: FK to document_versions
        title:               Section heading text
        section_order:       Position in document (0-indexed)
        section_path:        Full path from document root, e.g. ["Risk Factors", "Cloud"]
        parent_section_id:   FK to parent section row (None for top-level)

    Returns:
        Section ORM instance
    """
    existing = (
        db.query(Section)
        .filter_by(
            document_version_id=document_version_id,
            section_order=section_order,
            title=title,
        )
        .first()
    )
    if existing:
        return existing

    section = Section(
        document_version_id=document_version_id,
        parent_section_id=parent_section_id,
        title=title,
        section_path=section_path or [],
        section_order=section_order,
    )
    db.add(section)
    db.flush()
    return section


def build_section_map(
    db: Session,
    document_version_id: int,
    chunk_dicts: List[dict],
) -> Dict[str, int]:
    """
    Build a mapping of section_path_str → section.id for a document version.

    This is called during ingestion to create section rows from Phase 2 chunk
    metadata and return a lookup map so chunks can reference their section.

    The Phase 2 chunk format includes:
        chunk["section"]      - immediate section heading
        chunk["section_path"] - full path list ["Risk Factors", "Customer Concentration"]

    We create one Section row per unique section_path observed.

    Returns:
        Dict mapping section_path as string key → section primary key id
        Example: '["Risk Factors", "Cloud Infrastructure"]' → 42
    """
    import json

    section_map: Dict[str, int] = {}
    section_order = 0

    for chunk in chunk_dicts:
        path = chunk.get("section_path") or []
        if not path:
            continue

        path_key = json.dumps(path, ensure_ascii=False)
        if path_key in section_map:
            continue  # already created

        # Create the section
        title = path[-1] if path else chunk.get("section", "")
        section = upsert_section(
            db=db,
            document_version_id=document_version_id,
            title=title,
            section_order=section_order,
            section_path=path,
        )
        section_map[path_key] = section.id
        section_order += 1

    return section_map


def get_sections_for_version(db: Session, document_version_id: int) -> List[Section]:
    """Return all sections for a document version, ordered by section_order."""
    return (
        db.query(Section)
        .filter_by(document_version_id=document_version_id)
        .order_by(Section.section_order)
        .all()
    )
