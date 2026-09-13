"""
tests/retrieval/test_keyword.py  -  Unit tests for BM25 KeywordRetriever
=========================================================================
All tests use deterministic in-memory data. No disk I/O. No external calls.
"""

import pytest
from src.retrieval.keyword import KeywordRetriever, _tokenize


# ── Tokenizer tests ────────────────────────────────────────────────────────────

class TestTokenize:
    def test_lowercase(self):
        assert _tokenize("Revenue FY2024") == ["revenue", "fy2024"]

    def test_strips_punctuation(self):
        assert _tokenize("Revenue (USD m) | 121") == ["revenue", "usd", "m", "121"]

    def test_handles_numbers(self):
        assert "154" in _tokenize("Revenue 154 million")

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_punctuation(self):
        assert _tokenize("| | | ---") == []

    def test_hyphenated_words(self):
        # Hyphen is non-alphanumeric, so "mid-market" becomes ["mid", "market"]
        tokens = _tokenize("mid-market customers")
        assert "mid" in tokens
        assert "market" in tokens

    def test_special_chars_ignored(self):
        assert _tokenize("VaultGuard®") == ["vaultguard"]


# ── KeywordRetriever construction ─────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {"chunk_id": "A", "text": "Revenue FY2024 was 154 million USD.", "source": "test.pdf",
     "page_start": 1, "page_end": 1, "section": "Financial", "section_path": ["Financial"],
     "content_type": "text", "source_filename": "test.pdf"},
    {"chunk_id": "B", "text": "VaultGuard manages API keys and credentials.",
     "source": "test.pdf", "page_start": 2, "page_end": 2,
     "section": "Products", "section_path": ["Products"], "content_type": "text",
     "source_filename": "test.pdf"},
    {"chunk_id": "C", "text": "Key risks include cloud infrastructure concentration.",
     "source": "test.pdf", "page_start": 3, "page_end": 3,
     "section": "Risks", "section_path": ["Risks"], "content_type": "text",
     "source_filename": "test.pdf"},
    {"chunk_id": "D", "text": "OrionVault was founded in Bengaluru in 2018.",
     "source": "test.pdf", "page_start": 1, "page_end": 1,
     "section": "Overview", "section_path": ["Overview"], "content_type": "text",
     "source_filename": "test.pdf"},
]


class TestKeywordRetriever:

    def setup_method(self):
        self.retriever = KeywordRetriever(SAMPLE_CHUNKS)

    # ── Basic retrieval ───────────────────────────────────────────────────────

    def test_returns_list(self):
        results = self.retriever.search("revenue")
        assert isinstance(results, list)

    def test_exact_match_found(self):
        results = self.retriever.search("VaultGuard credentials")
        ids = [r.chunk_id for r in results]
        assert "B" in ids

    def test_exact_number_match(self):
        """BM25 should find the chunk containing '154'."""
        results = self.retriever.search("154")
        assert len(results) > 0
        assert results[0].chunk_id == "A"

    def test_no_overlap_returns_empty(self):
        """Query with no token overlap -> empty results."""
        results = self.retriever.search("xylophone purple")
        assert results == []

    def test_top_k_respected(self):
        results = self.retriever.search("the a in", top_k=2)
        assert len(results) <= 2

    def test_zero_score_filtered(self):
        """All results must have keyword_score > 0."""
        results = self.retriever.search("VaultGuard")
        for r in results:
            assert r.keyword_score > 0.0

    # ── Result structure ──────────────────────────────────────────────────────

    def test_result_has_required_fields(self):
        results = self.retriever.search("revenue FY2024")
        assert len(results) > 0
        r = results[0]
        assert r.chunk_id
        assert r.text
        assert r.keyword_score is not None
        assert r.keyword_rank is not None
        assert r.semantic_score is None   # not set by keyword retriever

    def test_ranks_are_ascending(self):
        """Rank 1 = best, rank n = worst."""
        results = self.retriever.search("VaultGuard credentials API")
        for i, r in enumerate(results, start=1):
            assert r.keyword_rank == i

    def test_sorted_by_score_descending(self):
        results = self.retriever.search("OrionVault cloud infrastructure risk")
        scores = [r.keyword_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_section_preserved(self):
        results = self.retriever.search("VaultGuard API keys")
        assert any(r.section == "Products" for r in results)

    def test_content_type_preserved(self):
        results = self.retriever.search("revenue 154")
        for r in results:
            assert r.content_type in {"text", "table", "list", "mixed"}

    # ── Query variants ────────────────────────────────────────────────────────

    def test_empty_query_returns_empty(self):
        assert self.retriever.search("") == []

    def test_single_word_query(self):
        results = self.retriever.search("Bengaluru")
        assert len(results) >= 1
        assert results[0].chunk_id == "D"

    def test_multi_word_query_higher_score(self):
        """More query words matching -> higher score."""
        r_multi = self.retriever.search("cloud infrastructure concentration risk")
        r_single = self.retriever.search("cloud")
        # The chunk matching more terms should score higher in multi-word query
        # (This is what BM25 is designed to do)
        assert len(r_multi) > 0 and len(r_single) > 0
        multi_top_score  = r_multi[0].keyword_score
        single_top_score = r_single[0].keyword_score
        # multi-word query should score the same chunk higher (more term overlap)
        # This is not guaranteed by BM25 in all cases, but holds for our test data
        assert multi_top_score >= single_top_score

    # ── get_chunk_scores ──────────────────────────────────────────────────────

    def test_get_chunk_scores_returns_dict(self):
        scores = self.retriever.get_chunk_scores("revenue")
        assert isinstance(scores, dict)
        assert "A" in scores   # "revenue" appears in chunk A

    def test_get_chunk_scores_zero_for_no_match(self):
        scores = self.retriever.get_chunk_scores("xylophone purple")
        assert all(v == 0.0 for v in scores.values())

    # ── explain ───────────────────────────────────────────────────────────────

    def test_explain_found_chunk(self):
        explanation = self.retriever.explain("VaultGuard credentials", "B")
        assert "B" in explanation
        assert "Matched tokens" in explanation

    def test_explain_missing_chunk(self):
        result = self.retriever.explain("query", "nonexistent_id")
        assert result is None

    # ── chunk_count ───────────────────────────────────────────────────────────

    def test_chunk_count(self):
        assert self.retriever.chunk_count == len(SAMPLE_CHUNKS)

    # ── Build latency recorded ────────────────────────────────────────────────

    def test_build_latency_recorded(self):
        assert self.retriever.build_latency_ms >= 0.0

    def test_search_latency_recorded(self):
        self.retriever.search("revenue")
        assert self.retriever.last_search_ms >= 0.0
