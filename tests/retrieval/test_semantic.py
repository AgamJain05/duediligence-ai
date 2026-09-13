"""
tests/retrieval/test_semantic.py  -  Unit tests for SemanticRetriever
=====================================================================
Tests that don't require a real FAISS index use mocks.
Tests that verify result structure use a fixture that builds a tiny in-memory index.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.models import RetrievalResult


# ── Helper: build a tiny real FAISS index in a temp directory ─────────────────

def _build_tiny_index(tmp_dir: Path, n_chunks: int = 4, dim: int = 4):
    """
    Build a tiny FAISS index + chunks.json in tmp_dir.
    Uses dim=4 for speed (real model uses 768).
    """
    import faiss

    embeddings = np.random.randn(n_chunks, dim).astype(np.float32)
    # L2 normalize (same as embed_texts does)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path  = tmp_dir / "faiss_structure_aware.index"
    chunks_path = tmp_dir / "chunks_structure_aware.json"

    faiss.write_index(index, str(index_path))

    chunks = [
        {
            "chunk_id":      f"chunk_{i:03d}",
            "text":          f"Sample text for chunk {i}",
            "source":        "test.pdf",
            "source_filename": "test.pdf",
            "page_start":    i + 1,
            "page_end":      i + 1,
            "section":       "Test Section",
            "section_path":  ["Test Section"],
            "content_type":  "text",
        }
        for i in range(n_chunks)
    ]
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)

    return embeddings, chunks


@pytest.fixture
def tiny_index(tmp_path):
    """Provides (tmp_dir, embeddings, chunks) for a 4-chunk, dim=4 index."""
    embeddings, chunks = _build_tiny_index(tmp_path, n_chunks=4, dim=4)
    return tmp_path, embeddings, chunks


# ── Import SemanticRetriever after fixture setup ──────────────────────────────
# We import late to avoid loading the real embedding model at test collection time.

class TestSemanticRetrieverConstruction:
    def test_loads_index_and_chunks(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, _, chunks = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        assert retriever.chunk_count == 4

    def test_missing_index_raises(self, tmp_path):
        from src.retrieval.semantic import SemanticRetriever
        with pytest.raises(FileNotFoundError):
            SemanticRetriever(index_dir=tmp_path)

    def test_all_chunks_accessible(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, _, chunks = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        all_c = retriever.all_chunks()
        assert len(all_c) == 4
        assert all(isinstance(c, dict) for c in all_c)


class TestSemanticRetrieverSearch:
    def test_search_returns_list(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)

        # Patch embed_query to return the first chunk's embedding (should rank #1)
        query_vec = embeddings[0:1].copy()   # shape (1, 4)
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            results = retriever.search("test query", top_k=3)

        assert isinstance(results, list)
        assert len(results) == 3

    def test_results_are_retrieval_results(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        query_vec = embeddings[0:1].copy()
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            results = retriever.search("test", top_k=2)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_semantic_score_set(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        query_vec = embeddings[0:1].copy()
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            results = retriever.search("test", top_k=4)
        for r in results:
            assert r.semantic_score is not None
            assert isinstance(r.semantic_score, float)

    def test_semantic_rank_ascending(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        query_vec = embeddings[0:1].copy()
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            results = retriever.search("test", top_k=4)
        ranks = [r.semantic_rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_scores_sorted_descending(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        query_vec = embeddings[0:1].copy()
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            results = retriever.search("test", top_k=4)
        scores = [r.semantic_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        query_vec = embeddings[0:1].copy()
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            results = retriever.search("test", top_k=2)
        assert len(results) == 2

    def test_empty_query_returns_empty(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, _, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        results = retriever.search("   ", top_k=5)
        assert results == []

    def test_top_k_larger_than_index(self, tiny_index):
        """Asking for more results than index size returns all available."""
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        query_vec = embeddings[0:1].copy()
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            results = retriever.search("test", top_k=1000)
        assert len(results) == 4   # only 4 chunks exist

    def test_keyword_score_not_set(self, tiny_index):
        """SemanticRetriever does NOT set keyword scores."""
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        query_vec = embeddings[0:1].copy()
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            results = retriever.search("test", top_k=2)
        for r in results:
            assert r.keyword_score is None
            assert r.keyword_rank is None

    def test_latency_recorded(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, embeddings, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        query_vec = embeddings[0:1].copy()
        with patch("src.retrieval.semantic.embed_query", return_value=query_vec):
            retriever.search("test", top_k=2)
        assert retriever.last_embedding_ms >= 0.0
        assert retriever.last_search_ms >= 0.0

    def test_get_chunk_by_id(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, _, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        chunk = retriever.get_chunk_by_id("chunk_000")
        assert chunk is not None
        assert chunk["chunk_id"] == "chunk_000"

    def test_get_chunk_by_id_missing(self, tiny_index):
        from src.retrieval.semantic import SemanticRetriever
        tmp_dir, _, _ = tiny_index
        retriever = SemanticRetriever(index_dir=tmp_dir)
        chunk = retriever.get_chunk_by_id("nonexistent_id")
        assert chunk is None
