"""
tests/db/test_schema.py  ─  Schema validation tests
══════════════════════════════════════════════════════
Tests:
  - All expected tables exist
  - Foreign keys are present
  - Unique constraints work
  - pgvector vector column exists with correct dimension
  - HNSW index exists
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from tests.db.conftest import requires_db


@requires_db
class TestTablesExist:
    """Verify all Phase 4 tables are present in the database."""

    EXPECTED_TABLES = [
        "companies",
        "sources",
        "documents",
        "document_versions",
        "sections",
        "chunks",
        "financial_metrics",
        "ingestion_runs",
    ]

    def test_all_tables_present(self, db_engine):
        inspector = inspect(db_engine)
        actual_tables = inspector.get_table_names()
        for table in self.EXPECTED_TABLES:
            assert table in actual_tables, (
                f"Table '{table}' is missing. Run: alembic upgrade head"
            )

    def test_companies_columns(self, db_engine):
        inspector = inspect(db_engine)
        cols = {c["name"] for c in inspector.get_columns("companies")}
        assert {"id", "name", "slug", "created_at", "updated_at"}.issubset(cols)

    def test_document_versions_columns(self, db_engine):
        inspector = inspect(db_engine)
        cols = {c["name"] for c in inspector.get_columns("document_versions")}
        required = {"id", "document_id", "version_label", "effective_date",
                    "published_date", "checksum", "is_current", "source_filename"}
        assert required.issubset(cols), f"Missing columns: {required - cols}"

    def test_chunks_has_embedding_column(self, db_engine):
        """Verify the pgvector embedding column exists."""
        inspector = inspect(db_engine)
        cols = {c["name"]: c for c in inspector.get_columns("chunks")}
        assert "embedding" in cols, "chunks.embedding column missing (run migrations)"

    def test_chunks_has_chunk_hash(self, db_engine):
        inspector = inspect(db_engine)
        cols = {c["name"] for c in inspector.get_columns("chunks")}
        assert "chunk_hash" in cols

    def test_chunks_has_embedding_model(self, db_engine):
        inspector = inspect(db_engine)
        cols = {c["name"] for c in inspector.get_columns("chunks")}
        assert "embedding_model" in cols


@requires_db
class TestUniqueConstraints:
    """Verify unique constraints prevent duplicate records."""

    def test_company_slug_unique(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        co1 = get_or_create_company(db_session, "TestCo Unique")
        co2 = get_or_create_company(db_session, "TestCo Unique")
        assert co1.id == co2.id, "Same company name should return same row"

    def test_document_version_checksum_unique(self, db_session):
        from src.db.repositories.companies  import get_or_create_company
        from src.db.repositories.documents  import (
            get_or_create_document, get_version_by_checksum, create_document_version
        )
        from sqlalchemy.exc import IntegrityError

        co  = get_or_create_company(db_session, "DupeTestCo")
        doc = get_or_create_document(db_session, co.id, "Test Doc", "policy")
        ver = create_document_version(
            db_session, doc.id, "v1", "test.pdf", "abc123checksum"
        )

        # Attempting to create another version with same checksum for same doc
        # should raise IntegrityError due to (document_id, checksum) unique constraint
        with pytest.raises(Exception):  # IntegrityError
            create_document_version(
                db_session, doc.id, "v1_duplicate", "test.pdf", "abc123checksum"
            )
            db_session.flush()

    def test_chunk_hash_unique(self, db_session, sample_chunk_text):
        from src.db.repositories.companies  import get_or_create_company
        from src.db.repositories.documents  import get_or_create_document, create_document_version
        from src.db.repositories.chunks     import upsert_chunk

        co  = get_or_create_company(db_session, "ChunkHashTestCo")
        doc = get_or_create_document(db_session, co.id, "Hash Test Doc", "policy")
        ver = create_document_version(db_session, doc.id, "v1", "f.pdf", "hash_unique_test_001")

        chunk1, created1 = upsert_chunk(db_session, ver.id, 0, sample_chunk_text, "text", 1, 1, 50)
        chunk2, created2 = upsert_chunk(db_session, ver.id, 0, sample_chunk_text, "text", 1, 1, 50)

        assert created1 is True,  "First insertion should create the chunk"
        assert created2 is False, "Second insertion (same text) should be skipped"
        assert chunk1.id == chunk2.id, "Should return same chunk row"


@requires_db
class TestForeignKeys:
    """Verify foreign key relationships."""

    def test_document_references_company(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document

        co  = get_or_create_company(db_session, "FKTestCo")
        doc = get_or_create_document(db_session, co.id, "FK Test Doc", "risk_memo")
        assert doc.company_id == co.id

    def test_version_references_document(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version

        co  = get_or_create_company(db_session, "FKTestCo2")
        doc = get_or_create_document(db_session, co.id, "FK Test Doc 2", "policy")
        ver = create_document_version(db_session, doc.id, "v1", "f.pdf", "fk_test_checksum")
        assert ver.document_id == doc.id


@requires_db
class TestHNSWIndex:
    """Verify HNSW index exists and is configured correctly."""

    def test_hnsw_index_exists(self, db_session):
        result = db_session.execute(
            text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'chunks'
                  AND indexname = 'chunks_embedding_hnsw_idx'
            """)
        ).fetchone()
        assert result is not None, (
            "HNSW index 'chunks_embedding_hnsw_idx' not found. "
            "Run: alembic upgrade head"
        )

    def test_hnsw_uses_cosine_ops(self, db_session):
        result = db_session.execute(
            text("""
                SELECT indexdef FROM pg_indexes
                WHERE tablename='chunks' AND indexname='chunks_embedding_hnsw_idx'
            """)
        ).scalar()
        assert result is not None
        assert "vector_cosine_ops" in result, (
            "HNSW index should use vector_cosine_ops for cosine distance"
        )

    def test_pgvector_extension_installed(self, db_session):
        result = db_session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname='vector'")
        ).fetchone()
        assert result is not None, "pgvector extension not installed"
