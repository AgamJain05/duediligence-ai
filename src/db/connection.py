"""
src/db/connection.py  ─  PostgreSQL connection management
══════════════════════════════════════════════════════════
Phase 4: Database connectivity using SQLAlchemy + psycopg2.

WHY SQLALCHEMY ENGINE?
───────────────────────
SQLAlchemy's engine manages a connection pool, meaning multiple queries
can reuse open connections without the overhead of establishing a new TCP
connection each time. For Supabase (a remote host), this matters.

The Session is the unit-of-work for ORM operations:
  - Changes are tracked in-memory
  - session.commit() flushes to the DB atomically
  - session.rollback() discards all pending changes
  - context manager usage ensures cleanup even on errors

SUPABASE SSL
─────────────
Supabase requires SSL. The DATABASE_URL must include ?sslmode=require.
SQLAlchemy passes this to psycopg2 automatically.

HNSW EF_SEARCH
───────────────
pgvector's HNSW index uses a per-session parameter `hnsw.ef_search` that
controls how many candidate vectors are explored during a query. A higher
value gives better recall at the cost of latency. We set it on every new
connection via a SQLAlchemy event listener so all sessions share the same
tuned value without callers needing to know about it.

Usage:
    from src.db.connection import get_db

    with get_db() as session:
        results = session.execute(...)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
_DATABASE_URL = os.getenv("DATABASE_URL", "")
_HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))

if not _DATABASE_URL:
    raise EnvironmentError(
        "[db/connection] DATABASE_URL is not set.\n"
        "  Add it to your .env file:\n"
        "  DATABASE_URL=postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres?sslmode=require"
    )

# ── Engine ────────────────────────────────────────────────────────────────────
# pool_pre_ping=True: checks connection health before using a pooled connection.
# This handles Supabase's idle connection timeout gracefully.
engine = create_engine(
    _DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,   # Set True for SQL query logging during debugging
)

# ── HNSW ef_search — applied to every new connection ─────────────────────────
@event.listens_for(engine, "connect")
def _set_hnsw_ef_search(dbapi_conn, connection_record):  # noqa: ANN001
    """
    Set hnsw.ef_search on every new DB connection.

    WHY HERE?
    pgvector's HNSW search quality is controlled by `hnsw.ef_search`.
    The default (40) is conservative. We increase it to improve recall
    at the cost of slightly higher latency. Setting it on the connection
    event means every session automatically uses the configured value.
    """
    with dbapi_conn.cursor() as cursor:
        cursor.execute(f"SET hnsw.ef_search = {_HNSW_EF_SEARCH};")


# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy Session. Commits on success, rolls back on any exception.

    Usage:
        with get_db() as db:
            db.add(some_object)
            # commit happens automatically on exit
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Health checks ─────────────────────────────────────────────────────────────

def ping_db() -> bool:
    """
    Return True if the database is reachable, False otherwise.

    WHY: Provides a simple connectivity test that can be run before
    ingestion or retrieval to give a clear error message instead of a
    cryptic SQLAlchemy traceback.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError as exc:
        print(f"[db/connection] Database unreachable: {exc}")
        return False


def check_pgvector() -> bool:
    """
    Return True if the pgvector extension is installed in the connected database.

    Supabase has pgvector pre-installed, but this check is explicit documentation
    that the project depends on it. If it's missing, the error message is clear.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
            row = result.fetchone()
            if row:
                return True
            print(
                "[db/connection] pgvector extension not found.\n"
                "  Run: CREATE EXTENSION IF NOT EXISTS vector;\n"
                "  (Supabase enables this by default via the dashboard.)"
            )
            return False
    except (OperationalError, ProgrammingError) as exc:
        print(f"[db/connection] pgvector check failed: {exc}")
        return False


def get_engine():
    """Return the shared SQLAlchemy engine (useful for Alembic env.py)."""
    return engine
