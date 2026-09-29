"""
src/db/__init__.py  ─  Phase 4 Database Package
════════════════════════════════════════════════
Exports the key symbols used throughout Phase 4.

Phase 4 adds PostgreSQL + pgvector as a persistent knowledge layer alongside
the Phase 1–3 FAISS-based retrieval. Nothing in this package removes Phase 1–3
functionality; it adds a new retrieval path.

Import hierarchy:
    src.db.connection  → engine, Session, get_db, ping_db
    src.db.models      → ORM table classes
    src.db.repositories → query/mutation functions per domain
"""

from src.db.connection import get_db, ping_db, check_pgvector  # noqa: F401

__all__ = ["get_db", "ping_db", "check_pgvector"]
