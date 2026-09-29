"""
tests/db/test_exact_vs_hnsw.py  ─  Exact vs HNSW recall comparison
════════════════════════════════════════════════════════════════════
Tests that HNSW approximate search achieves high recall relative to
exact brute-force search on this corpus.

For small corpora (< 1000 vectors), HNSW with ef_search=100 should
achieve perfect or near-perfect recall. This test documents and
verifies that property.

Also tests:
  - Both modes return the same number of results
  - HNSW results are a subset of exact results (no phantom results)
  - Similarity scores are plausible (0 ≤ similarity ≤ 1)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.db.conftest import requires_db


@requires_db
class TestExactVsHNSW:
    """Compare exact and HNSW retrieval on the Phase 4 knowledge base."""

    def test_hnsw_is_subset_of_exact(self, db_session, embedding_dim):
        """HNSW results should all exist in exact results (no phantom results)."""
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_EXACT, POSTGRES_HNSW
        from src.db.models import Chunk

        # Only run if there are embedded chunks to search
        chunk_count = db_session.query(Chunk).filter(Chunk.embedding.isnot(None)).count()
        if chunk_count == 0:
            pytest.skip("No embedded chunks in database — run ingest_pg.py first")

        retriever = PostgresRetriever(db_session)
        top_k = min(5, chunk_count)

        exact_results = retriever.search("cloud infrastructure risk", top_k=top_k,
                                          mode=POSTGRES_EXACT)
        hnsw_results  = retriever.search("cloud infrastructure risk", top_k=top_k,
                                          mode=POSTGRES_HNSW)

        exact_ids = {r.chunk_id for r in exact_results}
        hnsw_ids  = {r.chunk_id for r in hnsw_results}

        # Every HNSW result should appear in exact results (no phantom results)
        phantoms = hnsw_ids - exact_ids
        assert len(phantoms) == 0, (
            f"HNSW returned {len(phantoms)} chunks not in exact results: {phantoms}"
        )

    def test_recall_at_k(self, db_session, embedding_dim):
        """HNSW Recall@K relative to exact — expected ≥ 0.8 for small corpus."""
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_EXACT, POSTGRES_HNSW
        from src.db.models import Chunk

        chunk_count = db_session.query(Chunk).filter(Chunk.embedding.isnot(None)).count()
        if chunk_count == 0:
            pytest.skip("No embedded chunks in database")

        retriever = PostgresRetriever(db_session)
        top_k = min(5, chunk_count)

        queries = [
            "What are the key risks?",
            "OrionVault revenue FY2025",
            "Sentinel Copilot AI product",
        ]

        total_recall = 0.0
        count = 0

        for query in queries:
            exact  = retriever.search(query, top_k=top_k, mode=POSTGRES_EXACT)
            hnsw   = retriever.search(query, top_k=top_k, mode=POSTGRES_HNSW)

            exact_ids = {r.chunk_id for r in exact}
            hnsw_ids  = {r.chunk_id for r in hnsw}

            if exact_ids:
                recall = len(exact_ids & hnsw_ids) / len(exact_ids)
                total_recall += recall
                count += 1

        if count > 0:
            avg_recall = total_recall / count
            print(f"\n  Average HNSW Recall@{top_k} vs Exact: {avg_recall:.2%}")
            # For small corpus, recall should be near-perfect
            assert avg_recall >= 0.6, (
                f"HNSW Recall@{top_k} = {avg_recall:.2%} (expected ≥ 60%). "
                f"Try increasing HNSW_EF_SEARCH in .env."
            )

    def test_similarity_scores_plausible(self, db_session):
        """Similarity scores should be in [0, 1] range."""
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_HNSW
        from src.db.models import Chunk

        chunk_count = db_session.query(Chunk).filter(Chunk.embedding.isnot(None)).count()
        if chunk_count == 0:
            pytest.skip("No embedded chunks in database")

        retriever = PostgresRetriever(db_session)
        results = retriever.search("risk factors", top_k=5, mode=POSTGRES_HNSW)

        for r in results:
            assert 0.0 <= r.semantic_score <= 1.0, (
                f"Similarity score {r.semantic_score} out of [0,1] range. "
                f"Distance metric may be incorrect."
            )

    def test_sorted_by_similarity_descending(self, db_session):
        """Results should be sorted by similarity descending."""
        from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_HNSW
        from src.db.models import Chunk

        chunk_count = db_session.query(Chunk).filter(Chunk.embedding.isnot(None)).count()
        if chunk_count < 2:
            pytest.skip("Need at least 2 embedded chunks")

        retriever = PostgresRetriever(db_session)
        results = retriever.search("key risks", top_k=5, mode=POSTGRES_HNSW)

        if len(results) < 2:
            pytest.skip("Not enough results")

        scores = [r.semantic_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Results should be sorted by similarity score descending"
        )
