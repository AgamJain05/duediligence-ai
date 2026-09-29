"""
src/db/repositories/documents.py  ─  Document & version management
══════════════════════════════════════════════════════════════════════
Phase 4: Core versioning logic for the knowledge base.

VERSION MANAGEMENT POLICY
──────────────────────────
This module implements two key retrieval functions:

1. get_current_document_version(document_id)
   Returns the version where is_current=True.
   Used for: standard queries with no temporal context.

2. get_document_version_at_date(document_id, query_date)
   Returns the version applicable on a specific date.
   Used for: "What was the policy in 2024?" type queries.

   Precedence rules:
     Step 1: Filter versions where effective_date <= query_date
     Step 2: Among those, pick the latest effective_date
     Step 3: Tie-break: pick the latest published_date
     Step 4: Filename ordering is NOT used (unreliable)

WHY NOT USE is_current FOR TEMPORAL QUERIES?
──────────────────────────────────────────────
is_current marks the DEFAULT version for non-temporal queries. But if a
user asks "What was the policy in 2024?" they don't want the current 2026
version — they want the version effective in 2024. The effective_date field
enables this precise temporal retrieval.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from src.db.models import Document, DocumentVersion, Source


# ── Document operations ───────────────────────────────────────────────────────

def get_or_create_document(
    db: Session,
    company_id: int,
    title: str,
    document_type: str,
) -> Document:
    """
    Return existing document by (company_id, title, document_type), or create one.

    Idempotent: same company + title + type always maps to the same Document row.
    """
    doc = (
        db.query(Document)
        .filter_by(company_id=company_id, title=title, document_type=document_type)
        .first()
    )
    if doc:
        return doc

    doc = Document(company_id=company_id, title=title, document_type=document_type)
    db.add(doc)
    db.flush()
    return doc


def get_document_by_id(db: Session, document_id: int) -> Optional[Document]:
    """Return document by primary key."""
    return db.query(Document).filter_by(id=document_id).first()


def list_documents_for_company(db: Session, company_id: int) -> list[Document]:
    """Return all documents for a company."""
    return db.query(Document).filter_by(company_id=company_id).order_by(Document.title).all()


# ── Document version operations ───────────────────────────────────────────────

def compute_file_checksum(file_path: str | Path) -> str:
    """
    Compute SHA-256 checksum of a file.

    WHY SHA-256?
    - Collision-resistant: two different files will not produce the same hash
    - Fast enough for files up to hundreds of MB
    - Hex digest is 64 characters, fits in a VARCHAR(64) column

    This checksum is the primary mechanism for detecting duplicate ingestion:
    if the same file is ingested twice, the checksum matches and we skip.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_version_by_checksum(db: Session, document_id: int, checksum: str) -> Optional[DocumentVersion]:
    """
    Return an existing document version that matches (document_id, checksum).

    If found, this exact file has already been ingested for this document.
    The caller can skip ingestion and log a warning.
    """
    return (
        db.query(DocumentVersion)
        .filter_by(document_id=document_id, checksum=checksum)
        .first()
    )


def create_document_version(
    db: Session,
    document_id: int,
    version_label: str,
    source_filename: str,
    checksum: str,
    effective_date: Optional[date] = None,
    published_date: Optional[date] = None,
    source_uri: Optional[str] = None,
    is_current: bool = False,
    source_id: Optional[int] = None,
) -> DocumentVersion:
    """
    Create a new document version.

    If is_current=True, all other versions of this document are set to
    is_current=False first (only one version can be current at a time).
    """
    if is_current:
        # Unset is_current on all existing versions for this document
        _clear_current_flag(db, document_id)

    version = DocumentVersion(
        document_id=document_id,
        source_id=source_id,
        version_label=version_label,
        effective_date=effective_date,
        published_date=published_date,
        source_filename=source_filename,
        source_uri=source_uri,
        checksum=checksum,
        is_current=is_current,
    )
    db.add(version)
    db.flush()
    return version


def _clear_current_flag(db: Session, document_id: int) -> None:
    """Unset is_current on all versions of a document."""
    db.query(DocumentVersion).filter_by(document_id=document_id, is_current=True).update(
        {"is_current": False}
    )


def mark_current_version(db: Session, version_id: int) -> None:
    """Set a specific version as current (and unset all others for same document)."""
    version = db.query(DocumentVersion).filter_by(id=version_id).first()
    if not version:
        raise ValueError(f"DocumentVersion id={version_id} not found")
    _clear_current_flag(db, version.document_id)
    version.is_current = True
    db.flush()


# ── VERSION QUERY FUNCTIONS ───────────────────────────────────────────────────

def get_current_document_version(
    db: Session, document_id: int
) -> Optional[DocumentVersion]:
    """
    Return the current (default) version of a document.

    "Current" means is_current=True. This is the version to use when the
    user's query has no temporal context.

    Returns None if no version exists or none is marked current.
    """
    return (
        db.query(DocumentVersion)
        .filter_by(document_id=document_id, is_current=True)
        .first()
    )


def get_document_version_at_date(
    db: Session, document_id: int, query_date: date
) -> Optional[DocumentVersion]:
    """
    Return the document version applicable on query_date.

    PRECEDENCE RULES (documented, not guessed):
    ─────────────────────────────────────────────
    1. Candidate pool: versions where effective_date <= query_date
       (versions that had already become effective by that date)

    2. Primary sort: latest effective_date DESC
       (the most recently effective version wins)

    3. Tie-break: latest published_date DESC
       (if two versions share the same effective date, the later published one wins)

    4. Filename ordering: NOT used. Filenames are not reliably sortable.

    5. is_current: NOT used for temporal queries.
       (is_current means "current right now", not "current at query_date")

    Returns None if no version has effective_date <= query_date.
    In that case, the caller should decide whether to fall back to the earliest
    version or return an empty result.
    """
    results = (
        db.query(DocumentVersion)
        .filter(
            DocumentVersion.document_id == document_id,
            DocumentVersion.effective_date <= query_date,
        )
        .order_by(
            DocumentVersion.effective_date.desc(),
            DocumentVersion.published_date.desc().nullslast(),
        )
        .first()
    )
    return results


def list_versions_for_document(db: Session, document_id: int) -> list[DocumentVersion]:
    """Return all versions of a document ordered by effective_date."""
    return (
        db.query(DocumentVersion)
        .filter_by(document_id=document_id)
        .order_by(DocumentVersion.effective_date.asc())
        .all()
    )


# ── Source operations ─────────────────────────────────────────────────────────

def get_or_create_source(
    db: Session,
    source_name: str,
    source_uri: Optional[str] = None,
    source_type: str = "local_file",
    checksum: Optional[str] = None,
) -> Source:
    """
    Return or create a source record.

    For local files, source_uri is the absolute file path.
    For re-ingestion of the same file, we look up by checksum.
    """
    if checksum:
        existing = db.query(Source).filter_by(checksum=checksum).first()
        if existing:
            return existing

    source = Source(
        source_name=source_name,
        source_uri=source_uri,
        source_type=source_type,
        retrieved_at=datetime.utcnow(),
        checksum=checksum,
    )
    db.add(source)
    db.flush()
    return source
