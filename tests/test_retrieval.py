"""
tests/test_retrieval.py  ─  Unit tests for vector store, retrieval, and RAG pipeline
══════════════════════════════════════════════════════════════════════════════════════
Tests:
  Vector Store:
    1. build_index() succeeds with valid embeddings
    2. build_index() raises on empty input
    3. search() returns exactly top_k results
    4. search() results are (int, float) tuples
    5. Scores are in cosine similarity range [-1, 1]
    6. top_k is clamped when larger than index size

  Prompt Construction:
    7. Query appears in the built prompt
    8. Chunk text appears in the built prompt
    9. Source and page appear in the built prompt
    10. Prompt has correct role structure [system, user]

  RAG Pipeline (mocked LLM):
    11. Retrieved chunk text is injected into the LLM prompt
    12. Zero retrieved chunks triggers no-context response
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.vector_store import build_index, search
from src.generator    import build_prompt
from src.rag_pipeline import run_pipeline

# ── Shared test constants ─────────────────────────────────────────────────────
DIM = 768   # BAAI/bge-base-en-v1.5 output dimension
N   = 20    # number of synthetic vectors in the test index


def _make_normalized_embeddings(n: int = N, dim: int = DIM, seed: int = 42) -> np.ndarray:
    """
    Generate L2-normalized random float32 embeddings for testing.
    Normalization ensures inner product == cosine similarity, matching
    the assumptions of build_index() and embed_texts().
    """
    rng  = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_chunks(n: int = N) -> list:
    """Synthetic chunk metadata matching the format written by ingest.py."""
    return [
        {
            "chunk_id": f"chunk_{i:04d}",
            "source":   "orion_dossier.pdf",
            "page_num": (i % 10) + 1,
            "text":     f"OrionVault risk factor number {i}: supply chain exposure.",
        }
        for i in range(n)
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Vector Store Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildIndex:
    def test_build_succeeds_with_valid_embeddings(self):
        """build_index() must add all vectors to the FAISS index."""
        emb   = _make_normalized_embeddings(N)
        index = build_index(emb)
        assert index.ntotal == N

    def test_build_raises_on_empty_array(self):
        """build_index() must raise ValueError when given an empty array."""
        empty = np.empty((0, DIM), dtype=np.float32)
        with pytest.raises(ValueError):
            build_index(empty)

    def test_build_raises_on_1d_array(self):
        """build_index() must raise ValueError for a 1-D array."""
        bad = np.ones(DIM, dtype=np.float32)
        with pytest.raises(ValueError):
            build_index(bad)

    def test_index_dimension_matches_embeddings(self):
        """The FAISS index dimension must equal the embedding dimension."""
        emb   = _make_normalized_embeddings(5, dim=DIM)
        index = build_index(emb)
        assert index.d == DIM


class TestSearch:
    def setup_method(self):
        """Build a fresh index before each test."""
        self.embeddings = _make_normalized_embeddings(N)
        self.index      = build_index(self.embeddings)
        self.query_vec  = _make_normalized_embeddings(1, seed=99)

    def test_returns_exactly_top_k(self):
        """search() must return exactly top_k results when index is large enough."""
        results = search(self.index, self.query_vec, top_k=5)
        assert len(results) == 5

    def test_returns_tuples(self):
        """Each result must be a (int, float) tuple."""
        results = search(self.index, self.query_vec, top_k=3)
        for item in results:
            assert isinstance(item, tuple) and len(item) == 2
            idx, score = item
            assert isinstance(idx,   int)
            assert isinstance(score, float)

    def test_scores_in_cosine_range(self):
        """With L2-normalized vectors, inner-product scores must be in [-1, 1]."""
        results = search(self.index, self.query_vec, top_k=N)
        for _, score in results:
            assert -1.0 - 1e-5 <= score <= 1.0 + 1e-5, (
                f"Score {score:.6f} is outside cosine similarity range [-1, 1]"
            )

    def test_results_sorted_descending(self):
        """Results must be sorted from highest to lowest similarity score."""
        results = search(self.index, self.query_vec, top_k=10)
        scores  = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), (
            "Results are not sorted by descending similarity score"
        )

    def test_top_k_clamped_to_index_size(self):
        """Requesting more results than vectors must not raise; returns at most ntotal."""
        small_emb   = _make_normalized_embeddings(3)
        small_index = build_index(small_emb)
        results = search(small_index, self.query_vec, top_k=1000)
        assert len(results) <= 3

    def test_1d_query_is_accepted(self):
        """search() must accept a 1-D query array and reshape it internally."""
        query_1d = self.query_vec.flatten()   # shape (768,) not (1, 768)
        results  = search(self.index, query_1d, top_k=5)
        assert len(results) == 5

    def test_indices_are_valid(self):
        """Returned indices must be valid offsets into the embedding array."""
        results = search(self.index, self.query_vec, top_k=5)
        for idx, _ in results:
            assert 0 <= idx < N, f"Index {idx} is out of bounds for N={N}"


# ═════════════════════════════════════════════════════════════════════════════
# Prompt Construction Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildPrompt:
    CHUNK_1 = {
        "source":   "orion_dossier.pdf",
        "page_num": 4,
        "text":     "Supply chain concentration is a primary risk.",
        "score":    0.91,
    }
    CHUNK_2 = {
        "source":   "orion_financials.pdf",
        "page_num": 12,
        "text":     "Revenue grew 22% year-over-year in FY2024.",
        "score":    0.83,
    }

    def test_query_appears_in_user_message(self):
        """The user's question must appear verbatim in the user message."""
        msgs = build_prompt("What are the supply chain risks?", [self.CHUNK_1])
        user = msgs[1]["content"]
        assert "What are the supply chain risks?" in user

    def test_chunk_text_appears_in_user_message(self):
        """Each retrieved chunk's text must be present in the user message."""
        msgs = build_prompt("Supply chain?", [self.CHUNK_1, self.CHUNK_2])
        user = msgs[1]["content"]
        assert "Supply chain concentration is a primary risk." in user
        assert "Revenue grew 22% year-over-year in FY2024." in user

    def test_source_in_prompt(self):
        """Source filenames must appear in the user message for citation."""
        msgs = build_prompt("Question?", [self.CHUNK_1, self.CHUNK_2])
        user = msgs[1]["content"]
        assert "orion_dossier.pdf" in user
        assert "orion_financials.pdf" in user

    def test_page_number_in_prompt(self):
        """Page numbers must appear in the user message for citation."""
        msgs = build_prompt("Question?", [self.CHUNK_1])
        user = msgs[1]["content"]
        assert "4" in user   # page_num=4

    def test_prompt_has_system_and_user_roles(self):
        """build_prompt must return exactly [system, user] messages."""
        msgs = build_prompt("Q?", [self.CHUNK_1])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_message_contains_citation_rule(self):
        """System message must instruct the LLM to cite sources."""
        msgs = build_prompt("Q?", [self.CHUNK_1])
        sys_content = msgs[0]["content"].lower()
        assert "cite" in sys_content or "source" in sys_content

    def test_empty_chunks_prompt_still_valid(self):
        """build_prompt with no chunks must not crash."""
        msgs = build_prompt("Any question?", [])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"


# ═════════════════════════════════════════════════════════════════════════════
# RAG Pipeline Integration Tests  (LLM is mocked — no real API call)
# ═════════════════════════════════════════════════════════════════════════════

class TestRunPipeline:
    """
    Tests for run_pipeline().  We mock:
      - embed_query()   so we don't need the sentence-transformer model
      - OpenAI client   so we don't make real API calls

    The vector store and chunking run for real.
    """

    def setup_method(self):
        self.embeddings = _make_normalized_embeddings(N)
        self.index      = build_index(self.embeddings)
        self.chunks     = _make_chunks(N)

    @patch("src.retriever.embed_query")
    @patch("src.generator.OpenAI")
    def test_retrieved_chunks_in_prompt(self, mock_openai_cls, mock_embed_query):
        """
        The text of retrieved chunks must appear in the LLM prompt messages.
        We mock embed_query to return a valid vector and mock OpenAI to avoid
        real API calls.
        """
        # embed_query returns a valid normalized vector
        mock_embed_query.return_value = _make_normalized_embeddings(1, seed=7)

        # Mock the OpenAI client chain: client.chat.completions.create(...)
        mock_client   = MagicMock()
        mock_choice   = MagicMock()
        mock_choice.message.content = "Mocked LLM answer about supply chain."
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        mock_openai_cls.return_value = mock_client

        result = run_pipeline(
            query="What are OrionVault's risks?",
            index=self.index,
            chunks=self.chunks,
            top_k=3,
        )

        # Check trace dict structure
        assert "query"            in result
        assert "retrieved_chunks" in result
        assert "prompt_messages"  in result
        assert "answer"           in result
        assert "model_used"       in result

        # Check that retrieved chunks appear in the prompt
        user_content = result["prompt_messages"][1]["content"]
        retrieved_texts = [c["text"] for c in result["retrieved_chunks"]]
        for text in retrieved_texts:
            assert text in user_content, (
                f"Retrieved chunk text not found in prompt:\n  {text[:80]}"
            )

    @patch("src.retriever.embed_query")
    def test_zero_results_returns_no_context_message(self, mock_embed_query):
        """
        When the retriever finds nothing, run_pipeline must return a
        graceful no-context message without calling the LLM.
        """
        # Return a zero vector — FAISS will return results but with low scores.
        # To test the empty-results path, we use a tiny 0-vector index.
        import faiss as _faiss
        empty_index = _faiss.IndexFlatIP(DIM)
        # Index with 0 vectors
        mock_embed_query.return_value = _make_normalized_embeddings(1, seed=1)

        result = run_pipeline(
            query="Completely irrelevant question",
            index=empty_index,
            chunks=[],
            top_k=5,
        )

        assert result["retrieved_chunks"] == []
        assert "No relevant context" in result["answer"] or "no" in result["answer"].lower()
        assert result["model_used"].startswith("N/A")
