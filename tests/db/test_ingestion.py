"""
tests/db/test_ingestion.py  ─  Ingestion idempotency tests
════════════════════════════════════════════════════════════
Tests the four idempotency cases:
  Case 1: First ingestion — all records created
  Case 2: Same file re-ingested — no duplicates
  Case 3: New version (different checksum) — new version created, old preserved
  Case 4: Different document type — separate document row

Also tests:
  - Embedding dimension validation
  - checksum-based duplicate detection
  - is_current flag management
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.db.conftest import requires_db


@requires_db
class TestFirstIngestion:
    """Case 1: First ingestion creates all records correctly."""

    def test_company_created(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        co = get_or_create_company(db_session, "IngestTestCo")
        assert co.id is not None
        assert co.slug == "ingesttestco"

    def test_document_created(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document
        co  = get_or_create_company(db_session, "IngestDocCo")
        doc = get_or_create_document(db_session, co.id, "Annual Review 2024", "annual_review")
        assert doc.id is not None
        assert doc.document_type == "annual_review"
        assert doc.company_id == co.id

    def test_version_created_with_checksum(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        co  = get_or_create_company(db_session, "VersionCo1")
        doc = get_or_create_document(db_session, co.id, "Test Doc", "policy")
        ver = create_document_version(
            db_session, doc.id, "v1", "test.pdf", "deadbeef00001111"
        )
        assert ver.id is not None
        assert ver.version_label == "v1"
        assert ver.checksum == "deadbeef00001111"
        assert ver.document_id == doc.id

    def test_chunk_created_with_embedding(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.chunks    import upsert_chunk

        co  = get_or_create_company(db_session, "ChunkEmb Co")
        doc = get_or_create_document(db_session, co.id, "ChunkDoc", "risk_memo")
        ver = create_document_version(db_session, doc.id, "v1", "f.pdf", "emb_test_001")

        text = "This is a test chunk about cloud concentration risk."
        emb  = np.random.rand(embedding_dim).astype(np.float32)
        emb  = emb / np.linalg.norm(emb)  # normalize

        chunk, created = upsert_chunk(
            db_session, ver.id, 0, text, "text", 1, 2, 15,
            embedding=emb.reshape(1, embedding_dim),
            embedding_model="BAAI/bge-base-en-v1.5",
        )
        assert created is True
        assert chunk.id is not None
        assert chunk.embedding is not None
        assert len(chunk.embedding) == embedding_dim


@requires_db
class TestDuplicateDetection:
    """Case 2: Re-ingesting the same file produces no duplicates."""

    def test_same_checksum_detected(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import (
            get_or_create_document, create_document_version, get_version_by_checksum
        )

        co  = get_or_create_company(db_session, "DupeDetectCo")
        doc = get_or_create_document(db_session, co.id, "DupeDoc", "policy")
        _   = create_document_version(db_session, doc.id, "v1", "f.pdf", "dupe_checksum_aaa")

        # Simulate second ingestion: check checksum first
        existing = get_version_by_checksum(db_session, doc.id, "dupe_checksum_aaa")
        assert existing is not None, "Should detect duplicate checksum"
        assert existing.version_label == "v1"

    def test_same_chunk_text_not_duplicated(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.chunks    import upsert_chunk

        co  = get_or_create_company(db_session, "ChunkDupeCo")
        doc = get_or_create_document(db_session, co.id, "ChunkDupeDoc", "risk_memo")
        ver = create_document_version(db_session, doc.id, "v1", "f.pdf", "chunkdupe001")

        text = "Identical chunk text that should only appear once in the database."
        emb  = np.random.rand(embedding_dim).astype(np.float32)

        _, c1 = upsert_chunk(db_session, ver.id, 0, text, "text", 1, 1, 12,
                              embedding=emb.reshape(1, embedding_dim))
        _, c2 = upsert_chunk(db_session, ver.id, 0, text, "text", 1, 1, 12,
                              embedding=emb.reshape(1, embedding_dim))

        assert c1 is False or c2 is False, "One insertion should be skipped"


@requires_db
class TestVersionManagement:
    """Case 3 + 4: New version and different document types."""

    def test_new_version_different_checksum(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import (
            get_or_create_document, create_document_version,
            list_versions_for_document
        )

        co  = get_or_create_company(db_session, "NewVerCo")
        doc = get_or_create_document(db_session, co.id, "Versioned Doc", "annual_review")

        v1 = create_document_version(db_session, doc.id, "v1", "doc_v1.pdf", "checksum_v1_xxx")
        v2 = create_document_version(db_session, doc.id, "v2", "doc_v2.pdf", "checksum_v2_yyy",
                                      is_current=True)

        versions = list_versions_for_document(db_session, doc.id)
        assert len(versions) == 2, "Both versions should be stored"
        assert v1.id != v2.id

    def test_is_current_only_one(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import (
            get_or_create_document, create_document_version,
            list_versions_for_document
        )
        from src.db.models import DocumentVersion

        co  = get_or_create_company(db_session, "CurrentFlagCo")
        doc = get_or_create_document(db_session, co.id, "Current Flag Doc", "policy")

        _  = create_document_version(db_session, doc.id, "v1", "v1.pdf", "cs_v1",
                                      is_current=True)
        v2 = create_document_version(db_session, doc.id, "v2", "v2.pdf", "cs_v2",
                                      is_current=True)

        # After creating v2 as current, v1 should no longer be current
        versions = list_versions_for_document(db_session, doc.id)
        current_versions = [v for v in versions if v.is_current]
        assert len(current_versions) == 1, (
            f"Only one version should be current, found {len(current_versions)}"
        )
        assert current_versions[0].version_label == "v2"

    def test_different_document_type_separate_rows(self, db_session):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document

        co    = get_or_create_company(db_session, "DocTypeCo")
        doc1  = get_or_create_document(db_session, co.id, "OrionVault Review", "annual_review")
        doc2  = get_or_create_document(db_session, co.id, "OrionVault Risk Memo", "risk_memo")

        assert doc1.id != doc2.id, "Different document types should be separate rows"
        assert doc1.document_type == "annual_review"
        assert doc2.document_type == "risk_memo"


@requires_db
class TestEmbeddingDimensionValidation:
    """Verify embedding dimension mismatch is caught before insertion."""

    def test_wrong_dimension_raises(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.chunks    import upsert_chunk

        co  = get_or_create_company(db_session, "DimValidCo")
        doc = get_or_create_document(db_session, co.id, "DimDoc", "policy")
        ver = create_document_version(db_session, doc.id, "v1", "f.pdf", "dim_test_cs")

        wrong_emb = np.random.rand(embedding_dim - 1).astype(np.float32)  # wrong dim

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            upsert_chunk(
                db_session, ver.id, 0, "test text", "text", 1, 1, 5,
                embedding=wrong_emb.reshape(1, embedding_dim - 1),
            )

    def test_correct_dimension_passes(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.chunks    import upsert_chunk

        co  = get_or_create_company(db_session, "DimValidCo2")
        doc = get_or_create_document(db_session, co.id, "DimDoc2", "policy")
        ver = create_document_version(db_session, doc.id, "v1", "f.pdf", "dim_ok_cs")

        correct_emb = np.random.rand(embedding_dim).astype(np.float32)

        chunk, created = upsert_chunk(
            db_session, ver.id, 0, "dimension test chunk text", "text", 1, 1, 5,
            embedding=correct_emb.reshape(1, embedding_dim),
        )
        assert created is True
        assert chunk.embedding is not None
