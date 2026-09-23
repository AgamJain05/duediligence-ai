"""
tests/db/conftest.py  ─  Shared fixtures for Phase 4 database tests
══════════════════════════════════════════════════════════════════════

ISOLATION STRATEGY
──────────────────
Tests use transaction rollback isolation:
  1. Each test gets a fresh database session
  2. All changes are wrapped in a SAVEPOINT
  3. On teardown, the SAVEPOINT is rolled back
  4. The production data is untouched

This is faster than creating/dropping tables per test and safer than
running against a separate test database (though TEST_DATABASE_URL is
supported if set).

SKIP POLICY
────────────
If TEST_DATABASE_URL is empty AND the default DATABASE_URL cannot be
pinged, all tests in tests/db/ are skipped. This prevents CI failures
when the database is not configured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ── Database availability check ───────────────────────────────────────────────

def _get_test_db_url() -> str:
    """
    Return the test database URL.
    Prefers TEST_DATABASE_URL; falls back to DATABASE_URL.
    """
    return os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL", "")


def _db_available() -> bool:
    """Check if the test database is reachable."""
    url = _get_test_db_url()
    if not url or url.startswith("postgresql://postgres:[YOUR-PASSWORD]"):
        return False
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


# Compute once at collection time
DB_AVAILABLE = _db_available()

requires_db = pytest.mark.skipif(
    not DB_AVAILABLE,
    reason=(
        "Database not available. "
        "Set DATABASE_URL or TEST_DATABASE_URL in .env to run Phase 4 DB tests."
    ),
)


# ── Session fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_engine():
    """
    Create a single engine for all tests in the session.
    Disposed after all tests complete.
    """
    if not DB_AVAILABLE:
        pytest.skip("Database not available.")

    url = _get_test_db_url()
    from sqlalchemy import create_engine
    import pgvector.sqlalchemy  # noqa: F401 — register vector type

    engine = create_engine(url, pool_pre_ping=True, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """
    Provide a transactional database session for one test.

    Uses SAVEPOINT to wrap all test writes. On teardown, rolls back to the
    savepoint so the database is clean for the next test.

    This is the recommended pytest pattern for SQLAlchemy integration tests:
    https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction
    """
    from sqlalchemy.orm import Session
    import pgvector.sqlalchemy  # noqa: F401

    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    # Nested transaction (SAVEPOINT) for test isolation
    session.begin_nested()

    yield session

    # Teardown: roll back everything the test wrote
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="session")
def embedding_dim() -> int:
    """Return the expected embedding dimension from the configured model."""
    from src.db.models import EMBEDDING_DIM
    return EMBEDDING_DIM


@pytest.fixture
def sample_company_data():
    return {"name": "OrionVault Systems", "expected_slug": "orionvault-systems"}


@pytest.fixture
def sample_chunk_text():
    return (
        "OrionVault's top three customers collectively represented 41% of total "
        "subscription revenue in FY2025, creating material concentration risk. "
        "The company has taken steps to diversify its customer base through "
        "channel partnerships and expanded enterprise sales in the US market."
    )
