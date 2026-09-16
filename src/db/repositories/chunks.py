"""
src/db/repositories/chunks.py  ─  Chunk persistence and embedding storage
══════════════════════════════════════════════════════════════════════════════
Phase 4: Store Phase 2 chunks with their embeddings in PostgreSQL.

IDEMPOTENCY VIA chunk_hash
────────────────────────────
Every chunk has a chunk_hash (SHA-256 of text). The unique constraint on
chunk_hash means inserting the same text twice raises an IntegrityError,
which we catch and treat as "already exists — skip".

EMBEDDING DIMENSION VALIDATION
────────────────────────────────
Before any insertion, validate that the embedding dimension matches the
configured EMBEDDING_DIM (default 768 for BAAI/bge-base-en-v1.5). A
mismatch causes a clear ValueError rather than a cryptic PostgreSQL error.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.models import Chunk, EMBEDDING_DIM


def compute_chunk_hash(text: str) -> str:
    """
    Compute SHA-256 hash of chunk text.

    WHY HASH THE TEXT?
    The hash uniquely identifies chunk content regardless of chunk_id or
    document source. If the same paragraph appears in two document versions,
    both will produce the same hash — we only store the embedding once.
    This is the idempotency key for chunk insertion.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_embedding_dim(embedding: np.ndarray) -> None:
    """
    Validate that an embedding vector has the expected dimension.

    Raises ValueError with a clear message if there is a mismatch.
    This catches early the case where someone changed EMBEDDING_MODEL
    without updating EMBEDDING_DIM, which would silently insert wrong-sized
    vectors and cause pgvector dimension mismatch errors at search time.
    """
    actual_dim = embedding.shape[-1] if hasattr(embedding, 'shape') else len(embedding)
    if actual_dim != EMBEDDING_DIM:
        raise ValueError(
            f"[chunks] Embedding dimension mismatch: "
            f"expected {EMBEDDING_DIM} (EMBEDDING_DIM), got {actual_dim}.\n"
            f"  Check EMBEDDING_MODEL in .env — current model must produce "
            f"{EMBEDDING_DIM}-dimensional vectors."
        )


def upsert_chunk(
    db: Session,
    document_version_id: int,
    chunk_index: int,
    text: str,
    content_type: str,
    page_start: int,
    page_end: int,
    token_count: int,
    section_id: Optional[int] = None,
    embedding: Optional[np.ndarray] = None,
    embedding_model: Optional[str] = None,
) -> Tuple[Chunk, bool]:
    """
    Insert or return existing chunk by chunk_hash.

    Returns (chunk, created):
        chunk:   Chunk ORM instance
        created: True if newly inserted, False if already existed

    IDEMPOTENCY:
    If chunk_hash already exists in the DB, the existing chunk is returned
    and created=False. The caller can update embedding if it's missing.

    EMBEDDING STORAGE:
    If an embedding is provided, it is validated for dimension and stored.
    If the chunk already exists but has no embedding, the embedding is added.
    """
    chunk_hash = compute_chunk_hash(text)

    # Try to find existing chunk
    existing = db.query(Chunk).filter_by(chunk_hash=chunk_hash).first()
    if existing:
        # If embedding is now available and wasn't before, update it
        if embedding is not None and existing.embedding is None:
            validate_embedding_dim(embedding)
            existing.embedding = embedding.tolist()
            existing.embedding_model = embedding_model
        return existing, False

    # Validate embedding before insertion
    emb_list = None
    if embedding is not None:
        validate_embedding_dim(embedding)
        emb_list = embedding.tolist()

    chunk = Chunk(
        document_version_id=document_version_id,
        section_id=section_id,
        chunk_index=chunk_index,
        text=text,
        content_type=content_type,
        page_start=page_start,
        page_end=page_end,
        token_count=token_count,
        chunk_hash=chunk_hash,
        embedding_model=embedding_model,
        embedding=emb_list,
    )

    try:
        db.add(chunk)
        db.flush()
        return chunk, True
    except IntegrityError:
        db.rollback()
        # Race condition: another process inserted the same chunk_hash
        existing = db.query(Chunk).filter_by(chunk_hash=chunk_hash).first()
        return existing, False


def get_chunks_by_version(
    db: Session, document_version_id: int
) -> List[Chunk]:
    """Return all chunks for a document version, ordered by chunk_index."""
    return (
        db.query(Chunk)
        .filter_by(document_version_id=document_version_id)
        .order_by(Chunk.chunk_index)
        .all()
    )


def count_embedded_chunks(db: Session) -> int:
    """Return count of chunks that have a non-null embedding."""
    return db.query(Chunk).filter(Chunk.embedding.isnot(None)).count()


def count_total_chunks(db: Session) -> int:
    """Return total chunk count."""
    return db.query(Chunk).count()


def bulk_upsert_chunks(
    db: Session,
    document_version_id: int,
    chunk_dicts: List[Dict],
    embeddings: np.ndarray,
    section_map: Dict[str, int],
    embedding_model: str,
) -> Tuple[int, int]:
    """
    Upsert all chunks for a document version with their embeddings.

    Args:
        db:                  Active session
        document_version_id: FK reference
        chunk_dicts:         List of chunk dicts from Phase 2 ingestion
        embeddings:          numpy array shape (N, EMBEDDING_DIM)
        section_map:         Dict from build_section_map() — path_key → section.id
        embedding_model:     Model name string for recording

    Returns:
        (created_count, skipped_count)
    """
    import json

    created = 0
    skipped = 0

    for idx, (chunk_dict, emb_vec) in enumerate(zip(chunk_dicts, embeddings)):
        # Resolve section_id from section_path
        path = chunk_dict.get("section_path") or []
        path_key = json.dumps(path, ensure_ascii=False) if path else ""
        section_id = section_map.get(path_key) if path_key else None

        _, was_created = upsert_chunk(
            db=db,
            document_version_id=document_version_id,
            chunk_index=idx,
            text=chunk_dict["text"],
            content_type=chunk_dict.get("content_type", "text"),
            page_start=chunk_dict.get("page_start") or chunk_dict.get("page_num", 0),
            page_end=chunk_dict.get("page_end") or chunk_dict.get("page_start") or 0,
            token_count=chunk_dict.get("chunk_token_count", 0),
            section_id=section_id,
            embedding=emb_vec,
            embedding_model=embedding_model,
        )
        if was_created:
            created += 1
        else:
            skipped += 1

    return created, skipped
