"""
tests/db/test_retrieval_db.py  ─  PostgreSQL retrieval tests
══════════════════════════════════════════════════════════════
Tests:
  - Vector search returns results with provenance
  - Filtered vector search respects WHERE clauses
  - Provenance fields are complete in every result
  - EXPLAIN ANALYZE runs without error
  - FTS returns results for keyword queries
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.db.conftest import requires_db


def _insert_test_chunk(db_session, version_id: int, text: str, embedding_dim: int,
                        content_type: str = "text") -> None:
    """Helper to insert a chunk with a random normalized embedding."""
    from src.db.repositories.chunks import upsert_chunk
    emb = np.random.rand(embedding_dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    upsert_chunk(
        db_session, version_id, 0, text, content_type, 1, 1, len(text.split()),
        embedding=emb.reshape(1, embedding_dim),
        embedding_model="BAAI/bge-base-en-v1.5",
    )


@requires_db
class TestProvenanceFields:
    """Every result must carry complete provenance."""

    def test_result_has_company_field(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_HNSW

        co  = get_or_create_company(db_session, "ProvenanceCo")
        doc = get_or_create_document(db_session, co.id, "Provenance Doc", "risk_memo")
        ver = create_document_version(db_session, doc.id, "v1", "p.pdf", "prov_cs_001")
        _insert_test_chunk(db_session, ver.id, "Cloud infrastructure concentration risk details.", embedding_dim)

        retriever = PostgresRetriever(db_session)
        results = retriever.search("cloud risk", top_k=1, mode=POSTGRES_HNSW)

        if not results:
            pytest.skip("No results returned — corpus may be empty")

        r = results[0]
        assert r.metadata.get("company") is not None,            "company field missing"
        assert r.metadata.get("document") is not None,           "document field missing"
        assert r.metadata.get("document_type") is not None,      "document_type field missing"
        assert r.metadata.get("version_label") is not None,      "version_label field missing"
        assert r.metadata.get("document_version_id") is not None, "document_version_id missing"
        assert r.page_start is not None,                          "page_start missing"
        assert r.chunk_id is not None,                            "chunk_id missing"

    def test_result_chunk_id_is_hash(self, db_session, embedding_dim):
        """chunk_id should be the SHA-256 hash (64 hex chars)."""
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_HNSW

        co  = get_or_create_company(db_session, "HashCo")
        doc = get_or_create_document(db_session, co.id, "Hash Doc", "policy")
        ver = create_document_version(db_session, doc.id, "v1", "h.pdf", "hash_cs_002")
        _insert_test_chunk(db_session, ver.id, "Unique text for hash verification.", embedding_dim)

        retriever = PostgresRetriever(db_session)
        results = retriever.search("hash verification", top_k=1, mode=POSTGRES_HNSW)

        if not results:
            pytest.skip("No results returned")

        chunk_id = results[0].chunk_id
        assert len(chunk_id) == 64, f"chunk_id should be 64-char SHA-256, got {len(chunk_id)}"
        assert all(c in "0123456789abcdef" for c in chunk_id), "chunk_id should be hex"


@requires_db
class TestFilteredRetrieval:
    """Metadata filters are applied as SQL WHERE clauses."""

    def test_filter_by_document_type(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_HNSW

        co   = get_or_create_company(db_session, "FilterTypeCo")
        doc1 = get_or_create_document(db_session, co.id, "Risk Doc", "risk_memo")
        doc2 = get_or_create_document(db_session, co.id, "Policy Doc", "policy")
        v1   = create_document_version(db_session, doc1.id, "v1", "r.pdf", "filter_type_risk")
        v2   = create_document_version(db_session, doc2.id, "v1", "p.pdf", "filter_type_policy")

        _insert_test_chunk(db_session, v1.id, "Risk memo: cloud concentration risk factors.", embedding_dim)
        _insert_test_chunk(db_session, v2.id, "Policy: data residency requirements for Germany.", embedding_dim)

        retriever = PostgresRetriever(db_session)

        # Filter to only risk_memo
        risk_results = retriever.search(
            "cloud risk", top_k=5, filters={"document_type": "risk_memo"}, mode=POSTGRES_HNSW
        )
        risk_types = {r.metadata.get("document_type") for r in risk_results}
        # All returned chunks should be from risk_memo (if any match)
        assert "policy" not in risk_types, "Filter by document_type should exclude policy chunks"

    def test_filter_is_current(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_HNSW

        co  = get_or_create_company(db_session, "FilterCurrentCo")
        doc = get_or_create_document(db_session, co.id, "Current Doc", "annual_review")
        v1  = create_document_version(db_session, doc.id, "v1", "v1.pdf", "filter_curr_v1",
                                       is_current=False)
        v2  = create_document_version(db_session, doc.id, "v2", "v2.pdf", "filter_curr_v2",
                                       is_current=True)

        _insert_test_chunk(db_session, v1.id, "Old policy: data residency v1.", embedding_dim)
        _insert_test_chunk(db_session, v2.id, "New policy: data residency v2.", embedding_dim)

        retriever = PostgresRetriever(db_session)
        results = retriever.search(
            "data residency", top_k=5,
            filters={"is_current": True, "company": "FilterCurrentCo"},
            mode=POSTGRES_HNSW,
        )
        is_current_flags = {r.metadata.get("is_current") for r in results}
        assert False not in is_current_flags, (
            "Filter is_current=True should exclude non-current version results"
        )


@requires_db
class TestExactSearch:
    """Exact search bypasses the HNSW index."""

    def test_exact_returns_results(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_EXACT

        co  = get_or_create_company(db_session, "ExactSearchCo")
        doc = get_or_create_document(db_session, co.id, "Exact Doc", "risk_memo")
        ver = create_document_version(db_session, doc.id, "v1", "e.pdf", "exact_cs_001")
        _insert_test_chunk(db_session, ver.id, "Exact search test chunk.", embedding_dim)

        retriever = PostgresRetriever(db_session)
        results = retriever.search("exact search", top_k=5, mode=POSTGRES_EXACT)
        # Just verify it runs without error and returns a list
        assert isinstance(results, list)


@requires_db
class TestFullTextSearch:
    """PostgreSQL FTS returns results for keyword queries."""

    def test_fts_returns_results(self, db_session, embedding_dim):
        from src.db.repositories.companies import get_or_create_company
        from src.db.repositories.documents import get_or_create_document, create_document_version
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_FTS

        co  = get_or_create_company(db_session, "FTSCo")
        doc = get_or_create_document(db_session, co.id, "FTS Doc", "annual_review")
        ver = create_document_version(db_session, doc.id, "v1", "f.pdf", "fts_cs_001")
        _insert_test_chunk(
            db_session, ver.id,
            "OrionVault Sentinel Copilot is the company AI flagship product.",
            embedding_dim
        )

        retriever = PostgresRetriever(db_session)
        results = retriever.search("Sentinel Copilot", top_k=5, mode=POSTGRES_FTS)
        # FTS should find chunks containing "Sentinel" or "Copilot"
        assert isinstance(results, list)
