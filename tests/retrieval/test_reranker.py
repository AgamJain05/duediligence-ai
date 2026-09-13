"""
tests/retrieval/test_reranker.py  -  Unit tests for reranker interface
======================================================================
Tests verify the interface contract (input/output), not the model's quality.
The CrossEncoderReranker's model is mocked to avoid loading 85 MB weights.
NullReranker tests run without mocking (no model involved).
"""

import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.models import RetrievalResult
from src.retrieval.reranker import CrossEncoderReranker, NullReranker


def make_result(chunk_id: str, text: str = "Sample text.", fusion_score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id      = chunk_id,
        text          = text,
        source        = "test.pdf",
        page_start    = 1,
        page_end      = 1,
        section       = "Section",
        section_path  = ["Section"],
        content_type  = "text",
        metadata      = {},
        fusion_score  = fusion_score,
    )


# ── NullReranker ──────────────────────────────────────────────────────────────

class TestNullReranker:
    def setup_method(self):
        self.reranker = NullReranker()

    def test_returns_candidates_unchanged(self):
        candidates = [make_result("A"), make_result("B"), make_result("C")]
        result = self.reranker.rerank("query", candidates)
        ids = [r.chunk_id for r in result]
        assert ids == ["A", "B", "C"]

    def test_top_k_respected(self):
        candidates = [make_result("A"), make_result("B"), make_result("C")]
        result = self.reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2

    def test_empty_candidates(self):
        result = self.reranker.rerank("query", [])
        assert result == []

    def test_rerank_score_not_set(self):
        """NullReranker does not set rerank_score."""
        candidates = [make_result("A")]
        result = self.reranker.rerank("query", candidates)
        assert result[0].rerank_score is None

    def test_latency_zero(self):
        self.reranker.rerank("query", [make_result("A")])
        assert self.reranker.last_rerank_ms == 0.0

    def test_explain_movement_returns_string(self):
        before = [make_result("A"), make_result("B")]
        after  = [make_result("B"), make_result("A")]
        s = self.reranker.explain_movement(before, after)
        assert isinstance(s, str)
        assert "NullReranker" in s


# ── CrossEncoderReranker (mocked) ─────────────────────────────────────────────

class TestCrossEncoderReranker:
    """
    The cross-encoder model is mocked in all these tests.
    We test:
      1. The reranker calls predict() with correct (query, text) pairs
      2. Results are sorted by rerank_score descending
      3. rerank_score is set on every result
      4. Original scores (fusion, semantic) are preserved
      5. top_k limits the output
    """

    def _make_reranker_with_mock_model(self, mock_scores):
        """Create a CrossEncoderReranker with a mocked predict() method."""
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "mock-model"
        reranker.last_rerank_ms = 0.0
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores
        reranker._model = mock_model
        return reranker

    def test_rerank_sets_rerank_score(self):
        reranker = self._make_reranker_with_mock_model([3.0, 1.0, 2.0])
        candidates = [make_result("A"), make_result("B"), make_result("C")]
        result = reranker.rerank("query", candidates)
        for r in result:
            assert r.rerank_score is not None

    def test_reranked_sorted_by_score_descending(self):
        # A gets 1.0, B gets 3.0, C gets 2.0 -> reranked: B, C, A
        reranker = self._make_reranker_with_mock_model([1.0, 3.0, 2.0])
        candidates = [make_result("A"), make_result("B"), make_result("C")]
        result = reranker.rerank("query", candidates)
        assert [r.chunk_id for r in result] == ["B", "C", "A"]

    def test_top_k_limits_output(self):
        reranker = self._make_reranker_with_mock_model([1.0, 3.0, 2.0])
        candidates = [make_result("A"), make_result("B"), make_result("C")]
        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0].chunk_id == "B"
        assert result[1].chunk_id == "C"

    def test_predict_called_with_pairs(self):
        reranker = self._make_reranker_with_mock_model([1.0])
        candidates = [make_result("A", text="Some text")]
        reranker.rerank("my query", candidates)
        call_args = reranker._model.predict.call_args[0][0]
        assert call_args == [("my query", "Some text")]

    def test_fusion_score_preserved(self):
        reranker = self._make_reranker_with_mock_model([5.0])
        candidates = [make_result("A", fusion_score=0.0328)]
        result = reranker.rerank("query", candidates)
        assert result[0].fusion_score == pytest.approx(0.0328)

    def test_empty_candidates_returns_empty(self):
        reranker = self._make_reranker_with_mock_model([])
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_score_is_float(self):
        reranker = self._make_reranker_with_mock_model([3.14])
        candidates = [make_result("A")]
        result = reranker.rerank("query", candidates)
        assert isinstance(result[0].rerank_score, float)

    def test_explain_movement_up(self):
        reranker = self._make_reranker_with_mock_model([2.0, 5.0])
        candidates = [make_result("A"), make_result("B")]
        before = candidates
        after  = reranker.rerank("query", candidates)
        # B should move UP (was rank 2, now rank 1)
        explanation = reranker.explain_movement(before, after)
        assert "UP" in explanation

    def test_latency_recorded(self):
        reranker = self._make_reranker_with_mock_model([1.0, 2.0])
        candidates = [make_result("A"), make_result("B")]
        reranker.rerank("query", candidates)
        assert reranker.last_rerank_ms >= 0.0

    def test_all_original_fields_preserved(self):
        """semantic_score, keyword_score, etc. from input must survive reranking."""
        reranker = self._make_reranker_with_mock_model([1.0])
        r = make_result("A")
        r.semantic_score = 0.87
        r.semantic_rank  = 2
        r.keyword_score  = 12.3
        r.keyword_rank   = 1
        result = reranker.rerank("query", [r])
        assert result[0].semantic_score == 0.87
        assert result[0].semantic_rank  == 2
        assert result[0].keyword_score  == 12.3
        assert result[0].keyword_rank   == 1
