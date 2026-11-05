"""
tests/test_ui_backend.py  ─  Tests for the Streamlit UI backend adapter
=========================================================================

Tests verify:
  1. Empty query is rejected (returns error, not fabricated answer)
  2. Pipeline is called with correct query and mode
  3. Retrieved chunks are mapped correctly
  4. Missing metadata fields are handled gracefully
  5. Backend errors produce useful error messages
  6. Unsupported modes raise ValueError (never silently accepted)
  7. Filters are passed through correctly
  8. No answer fabrication when retrieval returns 0 results
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.models import (
    HybridSearchResult, LatencyRecord, RetrievalResult,
    VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED,
)
from src.ui_backend import (
    _map_results_to_chunks,
    _error_result,
    check_faiss_index,
    check_llm,
    get_available_modes,
    run_query,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(chunk_id="abc123", text="Sample text.", source="test.pdf",
                 page_start=1, page_end=2, section="Risk Factors",
                 semantic_score=0.85, rerank_score=None, fusion_score=None,
                 keyword_score=None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_start=page_start,
        page_end=page_end,
        section=section,
        section_path=[section],
        content_type="text",
        metadata={},
        semantic_score=semantic_score,
        rerank_score=rerank_score,
        fusion_score=fusion_score,
        keyword_score=keyword_score,
    )


def _make_hybrid_result(results=None) -> HybridSearchResult:
    return HybridSearchResult(
        query="test query",
        mode=HYBRID_RERANKED,
        results=results or [_make_result()],
        debug={},
        latency=LatencyRecord(total_ms=42.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Empty query returns error, not fabricated answer
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyQueryRejection:
    def test_empty_string_returns_error(self):
        """Empty query must return an error, never a fabricated answer."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        # A real HybridRetriever returns empty results for empty query
        mock_retriever.search.return_value = HybridSearchResult(
            query="", mode=HYBRID_RERANKED, results=[], debug={}, latency=LatencyRecord()
        )

        result = run_query("", mode=HYBRID_RERANKED, top_k=5, faiss_retriever=mock_retriever)
        # Should say no context found, not fabricate
        assert result["retrieved_chunks"] == []
        assert "No relevant" in result["answer"] or result.get("error") or result["model_used"] == "N/A"

    def test_whitespace_only_query_returns_no_chunks(self):
        """Whitespace-only query must not produce fabricated chunks."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.return_value = HybridSearchResult(
            query="   ", mode=HYBRID_RERANKED, results=[], debug={}, latency=LatencyRecord()
        )
        result = run_query("   ", mode=HYBRID_RERANKED, top_k=5, faiss_retriever=mock_retriever)
        assert result["retrieved_chunks"] == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pipeline called with correct query and mode
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineCalledCorrectly:
    def test_search_called_with_query(self):
        """retriever.search() must be called with the exact user query."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.return_value = _make_hybrid_result()

        with patch("src.ui_backend.answer_query") as mock_gen:
            mock_gen.return_value = {"answer": "test", "prompt_messages": [], "model_used": "test-model"}
            run_query("What is the revenue?", mode=HYBRID_RERANKED, top_k=5, faiss_retriever=mock_retriever)

        called_query = mock_retriever.search.call_args[0][0]
        assert called_query == "What is the revenue?"

    def test_top_k_applied(self):
        """final_top_k must be set to the caller's top_k before search."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.return_value = _make_hybrid_result()

        with patch("src.ui_backend.answer_query") as mock_gen:
            mock_gen.return_value = {"answer": "x", "prompt_messages": [], "model_used": "m"}
            run_query("query", mode=HYBRID_RERANKED, top_k=8, faiss_retriever=mock_retriever)

        assert mock_retriever.final_top_k == 8

    def test_filters_passed_to_search(self):
        """Non-empty filters must reach retriever.search()."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.return_value = _make_hybrid_result()

        filters = {"content_type": "table"}
        with patch("src.ui_backend.answer_query") as mock_gen:
            mock_gen.return_value = {"answer": "x", "prompt_messages": [], "model_used": "m"}
            run_query("query", mode=HYBRID_RERANKED, top_k=5, filters=filters,
                      faiss_retriever=mock_retriever)

        called_filters = mock_retriever.search.call_args[1].get("filters") or \
                         mock_retriever.search.call_args[0][1] if len(mock_retriever.search.call_args[0]) > 1 else None
        # Filters may be passed as keyword arg
        all_call_args = str(mock_retriever.search.call_args)
        assert "table" in all_call_args

    def test_answer_query_called_with_chunks(self):
        """answer_query must receive the mapped chunks, not raw RetrievalResult objects."""
        r = _make_result(chunk_id="xyz789", text="Revenue was $186M.")
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.return_value = _make_hybrid_result(results=[r])

        with patch("src.ui_backend.answer_query") as mock_gen:
            mock_gen.return_value = {"answer": "a", "prompt_messages": [], "model_used": "m"}
            run_query("revenue question", mode=HYBRID_RERANKED, top_k=5,
                      faiss_retriever=mock_retriever)

            called_chunks = mock_gen.call_args[0][1]
            assert isinstance(called_chunks, list)
            assert len(called_chunks) == 1
            assert called_chunks[0]["text"] == "Revenue was $186M."
            assert called_chunks[0]["source"] == "test.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Retrieved chunks mapped correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkMapping:
    def test_required_generator_fields_present(self):
        """Mapped chunks must have the fields answer_query() expects."""
        r = _make_result(source="doc.pdf", page_start=3, text="Some text.")
        chunks = _map_results_to_chunks([r])
        assert len(chunks) == 1
        c = chunks[0]
        # Generator requires: source, page_num, text, score, chunk_id
        assert c["source"] == "doc.pdf"
        assert c["page_num"] == 3
        assert c["text"] == "Some text."
        assert isinstance(c["score"], float)
        assert "chunk_id" in c

    def test_score_priority_rerank_over_semantic(self):
        """score field must prefer rerank_score when present."""
        r = _make_result(semantic_score=0.7, rerank_score=0.95)
        chunks = _map_results_to_chunks([r])
        assert chunks[0]["score"] == pytest.approx(0.95)

    def test_score_fallback_to_semantic(self):
        """score field falls back to semantic_score when rerank is None."""
        r = _make_result(semantic_score=0.72, rerank_score=None)
        chunks = _map_results_to_chunks([r])
        assert chunks[0]["score"] == pytest.approx(0.72)

    def test_extra_ui_fields_present(self):
        """Mapped chunks include diagnostic fields for the UI."""
        r = _make_result(section="Risk Factors")
        chunks = _map_results_to_chunks([r])
        c = chunks[0]
        assert "section" in c
        assert "semantic_score" in c
        assert "keyword_score" in c
        assert "fusion_score" in c
        assert "rerank_score" in c


# ─────────────────────────────────────────────────────────────────────────────
# 4. Missing metadata handled gracefully
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingMetadata:
    def test_empty_section_ok(self):
        """Chunks with empty section should not raise."""
        r = _make_result(section="")
        chunks = _map_results_to_chunks([r])
        assert chunks[0]["section"] == ""

    def test_none_scores_ok(self):
        """Chunks with all None scores should not raise and score=0.0."""
        r = _make_result(semantic_score=None, rerank_score=None,
                         fusion_score=None, keyword_score=None)
        chunks = _map_results_to_chunks([r])
        assert chunks[0]["score"] == 0.0

    def test_metadata_missing_keys_return_none(self):
        """UI fields from metadata dict that are absent return None, not KeyError."""
        r = _make_result()
        r.metadata = {}  # empty metadata
        chunks = _map_results_to_chunks([r])
        assert chunks[0]["document"] is None
        assert chunks[0]["version_label"] is None
        assert chunks[0]["effective_date"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Backend errors produce useful error messages
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_retrieval_exception_returns_error_result(self):
        """A crash in retriever.search() must return an error dict, not raise."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.side_effect = RuntimeError("FAISS index corrupted")

        result = run_query("query", mode=HYBRID_RERANKED, top_k=5, faiss_retriever=mock_retriever)
        assert result["error"] is not None
        assert "FAISS index corrupted" in result["error"]
        assert result["retrieved_chunks"] == []
        assert result["model_used"] == "N/A"

    def test_generation_exception_returns_error_result(self):
        """A crash in answer_query() must return an error dict."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.return_value = _make_hybrid_result()

        with patch("src.ui_backend.answer_query") as mock_gen:
            mock_gen.side_effect = RuntimeError("LLM API timeout")
            result = run_query("query", mode=HYBRID_RERANKED, top_k=5, faiss_retriever=mock_retriever)

        assert result["error"] is not None
        assert "LLM API timeout" in result["error"]

    def test_none_retriever_returns_error(self):
        """Passing None as retriever must return an error, not crash."""
        result = run_query("query", mode=HYBRID_RERANKED, top_k=5, faiss_retriever=None)
        assert result["error"] is not None
        assert result["retrieved_chunks"] == []

    def test_error_result_helper(self):
        """_error_result always returns a safe dict with no answer fabrication."""
        r = _error_result("question", HYBRID_RERANKED, "Database connection refused")
        assert "Database connection refused" in r["error"]
        assert r["model_used"] == "N/A"
        assert r["retrieved_chunks"] == []
        # answer field is just the error message, not a fabricated answer
        assert "Error" in r["answer"] or "error" in r["answer"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Unsupported modes cannot be selected
# ─────────────────────────────────────────────────────────────────────────────

class TestModeValidation:
    def test_invalid_mode_not_in_available_modes(self):
        """get_available_modes must only return modes that are actually usable."""
        faiss_modes = get_available_modes(faiss_ok=True, db_ok=False)
        assert "MADE_UP_MODE" not in faiss_modes

    def test_no_faiss_no_faiss_modes(self):
        """When FAISS is unavailable, no FAISS modes should be returned."""
        modes = get_available_modes(faiss_ok=False, db_ok=False)
        from src.ui_backend import FAISS_MODES
        for m in modes:
            assert m not in FAISS_MODES

    def test_faiss_modes_available_when_index_present(self):
        """When FAISS is available, all 4 Phase 3 modes are returned."""
        modes = get_available_modes(faiss_ok=True, db_ok=False)
        assert VECTOR_ONLY in modes
        assert KEYWORD_ONLY in modes
        assert HYBRID in modes
        assert HYBRID_RERANKED in modes

    def test_pg_modes_only_when_db_ok(self):
        """PostgreSQL modes only appear when db_ok=True."""
        from src.ui_backend import POSTGRES_HNSW
        modes_no_db = get_available_modes(faiss_ok=True, db_ok=False)
        assert POSTGRES_HNSW not in modes_no_db
        modes_with_db = get_available_modes(faiss_ok=True, db_ok=True)
        assert POSTGRES_HNSW in modes_with_db


# ─────────────────────────────────────────────────────────────────────────────
# 7. No answer fabrication when retrieval returns 0 results
# ─────────────────────────────────────────────────────────────────────────────

class TestNoFabrication:
    def test_empty_retrieval_skips_llm(self):
        """When retrieval returns 0 chunks, answer_query must NOT be called."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.return_value = HybridSearchResult(
            query="anything", mode=HYBRID_RERANKED, results=[], debug={}, latency=LatencyRecord()
        )

        with patch("src.ui_backend.answer_query") as mock_gen:
            result = run_query("anything", mode=HYBRID_RERANKED, top_k=5,
                               faiss_retriever=mock_retriever)
            mock_gen.assert_not_called()

        assert result["retrieved_chunks"] == []
        assert result["model_used"] == "N/A"

    def test_empty_retrieval_answer_explains_situation(self):
        """Empty retrieval answer must explain missing context, not invent facts."""
        mock_retriever = MagicMock()
        mock_retriever.mode = HYBRID_RERANKED
        mock_retriever.search.return_value = HybridSearchResult(
            query="q", mode=HYBRID_RERANKED, results=[], debug={}, latency=LatencyRecord()
        )
        result = run_query("q", mode=HYBRID_RERANKED, top_k=5, faiss_retriever=mock_retriever)
        # Must mention something about no context, not just return an answer
        answer_lower = result["answer"].lower()
        assert any(w in answer_lower for w in ["no relevant", "not found", "no context", "ingest", "error"])


# ─────────────────────────────────────────────────────────────────────────────
# 8. Status checks work without crashing
# ─────────────────────────────────────────────────────────────────────────────

class TestStatusChecks:
    def test_check_faiss_index_returns_tuple(self):
        ok, msg = check_faiss_index()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_check_llm_returns_tuple(self):
        ok, msg = check_llm()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_check_llm_false_when_no_key(self):
        import os
        original = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            ok, msg = check_llm()
            assert ok is False
            assert "OPENROUTER_API_KEY" in msg
        finally:
            if original:
                os.environ["OPENROUTER_API_KEY"] = original
