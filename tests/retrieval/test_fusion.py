"""
tests/retrieval/test_fusion.py  -  Unit tests for RRF fusion
=============================================================
Tests the RRF formula, merging behavior, and score ordering.
All tests use synthetic deterministic data. No I/O. No external calls.
"""

import pytest
from src.retrieval.models import RetrievalResult
from src.retrieval.fusion import ReciprocalRankFusion


def make_result(chunk_id: str, sem_rank: int = None, kw_rank: int = None,
                sem_score: float = None, kw_score: float = None) -> RetrievalResult:
    """Helper: build a minimal RetrievalResult for fusion tests."""
    return RetrievalResult(
        chunk_id      = chunk_id,
        text          = f"Text for {chunk_id}",
        source        = "test.pdf",
        page_start    = 1,
        page_end      = 1,
        section       = "Test Section",
        section_path  = ["Test Section"],
        content_type  = "text",
        metadata      = {},
        semantic_score = sem_score,
        semantic_rank  = sem_rank,
        keyword_score  = kw_score,
        keyword_rank   = kw_rank,
    )


class TestReciprocalRankFusion:

    def setup_method(self):
        self.rrf = ReciprocalRankFusion(k=60)

    # ── Basic formula verification ────────────────────────────────────────────

    def test_rrf_score_formula_rank1(self):
        """1 / (60 + 1) = 0.01639..."""
        sem = [make_result("A", sem_rank=1)]
        kw  = []
        fused = self.rrf.fuse(sem, kw)
        assert len(fused) == 1
        expected = 1.0 / (60 + 1)
        assert abs(fused[0].fusion_score - expected) < 1e-7

    def test_rrf_score_double_contribution(self):
        """A chunk in both lists gets sum of both contributions."""
        # A at sem_rank=1, A at kw_rank=1
        sem = [make_result("A", sem_rank=1)]
        kw  = [make_result("A", kw_rank=1)]
        fused = self.rrf.fuse(sem, kw)
        assert len(fused) == 1
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert abs(fused[0].fusion_score - expected) < 1e-7

    def test_chunk_in_both_lists_ranks_higher(self):
        """
        A chunk appearing in both lists should outrank a chunk in only one list
        (assuming similar ranks).
        """
        # A in both at rank 1; B in semantic only at rank 2
        sem = [make_result("A", sem_rank=1), make_result("B", sem_rank=2)]
        kw  = [make_result("A", kw_rank=1)]
        fused = self.rrf.fuse(sem, kw)
        # A: 1/61 + 1/61 = 0.03279; B: 1/62 = 0.01613
        assert fused[0].chunk_id == "A"
        assert fused[1].chunk_id == "B"

    def test_keyword_only_chunk_appears_in_fused(self):
        """A chunk ONLY in keyword list should appear in fused results."""
        sem = [make_result("A", sem_rank=1)]
        kw  = [make_result("B", kw_rank=1)]
        fused = self.rrf.fuse(sem, kw)
        ids = [r.chunk_id for r in fused]
        assert "A" in ids
        assert "B" in ids
        assert len(fused) == 2

    def test_sorted_descending(self):
        """Fused results must be in descending fusion_score order."""
        sem = [make_result("A", sem_rank=1), make_result("B", sem_rank=2), make_result("C", sem_rank=3)]
        kw  = [make_result("C", kw_rank=1), make_result("A", kw_rank=2)]
        fused = self.rrf.fuse(sem, kw)
        scores = [r.fusion_score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_empty_semantic(self):
        """Only keyword list — should return keyword results."""
        kw = [make_result("A", kw_rank=1), make_result("B", kw_rank=2)]
        fused = self.rrf.fuse([], kw)
        assert len(fused) == 2

    def test_empty_keyword(self):
        """Only semantic list — should return semantic results."""
        sem = [make_result("A", sem_rank=1), make_result("B", sem_rank=2)]
        fused = self.rrf.fuse(sem, [])
        assert len(fused) == 2

    def test_both_empty(self):
        fused = self.rrf.fuse([], [])
        assert fused == []

    def test_no_duplicates_in_output(self):
        """Each chunk_id should appear exactly once in fused output."""
        sem = [make_result("A", sem_rank=1), make_result("B", sem_rank=2)]
        kw  = [make_result("A", kw_rank=1), make_result("B", kw_rank=2)]
        fused = self.rrf.fuse(sem, kw)
        ids = [r.chunk_id for r in fused]
        assert len(ids) == len(set(ids))

    def test_semantic_scores_preserved(self):
        """Semantic scores from the input should be preserved in fused output."""
        sem = [make_result("A", sem_rank=1, sem_score=0.95)]
        kw  = []
        fused = self.rrf.fuse(sem, kw)
        assert fused[0].semantic_score == 0.95
        assert fused[0].semantic_rank == 1

    def test_keyword_scores_merged(self):
        """Keyword scores should be merged into the fused result for a dual-list chunk."""
        sem = [make_result("A", sem_rank=1, sem_score=0.90)]
        kw  = [make_result("A", kw_rank=1, kw_score=12.5)]
        fused = self.rrf.fuse(sem, kw)
        assert fused[0].semantic_score == 0.90
        assert fused[0].keyword_score  == 12.5
        assert fused[0].semantic_rank  == 1
        assert fused[0].keyword_rank   == 1

    # ── RRF explain ───────────────────────────────────────────────────────────

    def test_explain_both_lists(self):
        sem = [make_result("A", sem_rank=1), make_result("B", sem_rank=2)]
        kw  = [make_result("A", kw_rank=3)]
        explanation = self.rrf.explain(sem, kw, "A")
        assert "Semantic rank" in explanation
        assert "Keyword rank" in explanation
        assert "RRF total" in explanation

    def test_explain_missing_chunk(self):
        sem = [make_result("A", sem_rank=1)]
        kw  = []
        explanation = self.rrf.explain(sem, kw, "Z")
        # Z not in either list — both contributions 0
        assert "NOT IN LIST" in explanation

    # ── Configurable k ────────────────────────────────────────────────────────

    def test_lower_k_increases_top_rank_advantage(self):
        """Lower k gives more advantage to rank-1 over rank-2."""
        rrf_low_k  = ReciprocalRankFusion(k=1)
        rrf_high_k = ReciprocalRankFusion(k=1000)

        sem = [make_result("A", sem_rank=1), make_result("B", sem_rank=2)]
        fused_low  = rrf_low_k.fuse(sem, [])
        fused_high = rrf_high_k.fuse(sem, [])

        # With k=1: 1/2 vs 1/3 = ratio 1.5
        # With k=1000: 1/1001 vs 1/1002 = ratio ~1.001 (nearly equal)
        ratio_low  = fused_low[0].fusion_score  / fused_low[1].fusion_score
        ratio_high = fused_high[0].fusion_score / fused_high[1].fusion_score
        assert ratio_low > ratio_high  # low k amplifies rank differences
