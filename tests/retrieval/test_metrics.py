"""
tests/retrieval/test_metrics.py  -  Unit tests for Recall@K, Precision@K, MRR
===============================================================================
All tests use deterministic data. No external API calls. No file I/O.
"""

import pytest
from src.retrieval.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    mrr,
    evaluate_mode,
    format_metrics_table,
)


# ── Recall@K ─────────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_perfect_recall_k1(self):
        """Single relevant doc at rank 1."""
        assert recall_at_k(["A", "B", "C"], ["A"], k=1) == 1.0

    def test_zero_recall_k1(self):
        """Relevant doc not in top-1."""
        assert recall_at_k(["B", "C", "A"], ["A"], k=1) == 0.0

    def test_recall_k5_all_found(self):
        """All 3 relevant in top-5."""
        assert recall_at_k(["A", "B", "C", "D", "E"], ["A", "B", "C"], k=5) == 1.0

    def test_recall_k5_partial(self):
        """2 of 3 relevant in top-5."""
        result = recall_at_k(["A", "B", "D", "E", "F"], ["A", "B", "C"], k=5)
        assert abs(result - 2/3) < 1e-9

    def test_recall_increases_with_k(self):
        """Recall@K can only increase as K grows."""
        retrieved = ["D", "A", "E", "B", "F", "C"]
        relevant  = ["A", "B", "C"]
        r1 = recall_at_k(retrieved, relevant, k=1)
        r3 = recall_at_k(retrieved, relevant, k=3)
        r5 = recall_at_k(retrieved, relevant, k=5)
        r6 = recall_at_k(retrieved, relevant, k=6)
        assert r1 <= r3 <= r5 <= r6

    def test_empty_relevant_returns_one(self):
        """Vacuously true: nothing to find."""
        assert recall_at_k(["A", "B"], [], k=5) == 1.0

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], ["A"], k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        """K larger than the retrieved list — uses all available results."""
        assert recall_at_k(["A"], ["A", "B"], k=100) == 0.5

    def test_no_overlap(self):
        assert recall_at_k(["X", "Y", "Z"], ["A", "B"], k=5) == 0.0

    def test_duplicate_in_retrieved(self):
        """Duplicates in retrieved list — only unique ids count."""
        # Standard implementation uses set intersection
        result = recall_at_k(["A", "A", "A"], ["A"], k=3)
        assert result == 1.0  # A found in top-3 set


# ── Precision@K ──────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["A", "B"], ["A", "B", "C"], k=2) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["X", "Y"], ["A", "B"], k=2) == 0.0

    def test_half_precision(self):
        result = precision_at_k(["A", "X"], ["A"], k=2)
        assert abs(result - 0.5) < 1e-9

    def test_precision_decreases_with_k(self):
        """Adding irrelevant results at lower ranks decreases precision."""
        retrieved = ["A", "X", "Y", "Z"]
        relevant  = ["A"]
        p1 = precision_at_k(retrieved, relevant, k=1)
        p2 = precision_at_k(retrieved, relevant, k=2)
        p4 = precision_at_k(retrieved, relevant, k=4)
        assert p1 >= p2 >= p4

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["A"], ["A"], k=0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["A"], k=5) == 0.0

    def test_all_relevant(self):
        assert precision_at_k(["A", "B", "C"], ["A", "B", "C", "D"], k=3) == 1.0


# ── Reciprocal Rank ───────────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_rank_1(self):
        assert reciprocal_rank(["A", "B", "C"], ["A"]) == 1.0

    def test_rank_2(self):
        assert abs(reciprocal_rank(["B", "A", "C"], ["A"]) - 0.5) < 1e-9

    def test_rank_3(self):
        assert abs(reciprocal_rank(["B", "C", "A"], ["A"]) - 1/3) < 1e-9

    def test_not_found(self):
        assert reciprocal_rank(["X", "Y", "Z"], ["A"]) == 0.0

    def test_multiple_relevant_picks_first(self):
        """RR uses the FIRST relevant chunk encountered."""
        result = reciprocal_rank(["X", "B", "A"], ["A", "B"])
        assert abs(result - 0.5) < 1e-9  # B is at rank 2, A is at rank 3

    def test_empty_retrieved(self):
        assert reciprocal_rank([], ["A"]) == 0.0


# ── MRR ───────────────────────────────────────────────────────────────────────

class TestMRR:
    def test_single_query_rank1(self):
        assert mrr([["A", "B"]], [["A"]]) == 1.0

    def test_single_query_rank2(self):
        assert abs(mrr([["B", "A"]], [["A"]]) - 0.5) < 1e-9

    def test_three_queries(self):
        all_retrieved = [["A", "B"], ["X", "A"], ["B", "X", "A"]]
        all_relevant  = [["A"],      ["A"],      ["A"]]
        # RRs: 1.0, 0.5, 1/3
        expected = (1.0 + 0.5 + 1/3) / 3
        assert abs(mrr(all_retrieved, all_relevant) - expected) < 1e-9

    def test_no_relevant_found(self):
        assert mrr([["X", "Y"]], [["A"]]) == 0.0

    def test_empty_lists(self):
        assert mrr([], []) == 0.0

    def test_mixed_found_not_found(self):
        all_retrieved = [["A"], ["X"]]   # first found, second not
        all_relevant  = [["A"], ["A"]]
        # RRs: 1.0, 0.0 -> MRR = 0.5
        assert abs(mrr(all_retrieved, all_relevant) - 0.5) < 1e-9


# ── evaluate_mode ─────────────────────────────────────────────────────────────

class TestEvaluateMode:
    def test_basic_output_keys(self):
        retrieved = [["A", "B", "C"]]
        relevant  = [["A"]]
        result = evaluate_mode(retrieved, relevant, k_values=[1, 5])
        assert "Recall@1" in result
        assert "Recall@5" in result
        assert "Precision@1" in result
        assert "MRR" in result

    def test_perfect_results(self):
        retrieved = [["A", "B"], ["C", "D"]]
        relevant  = [["A"],     ["C"]]
        result = evaluate_mode(retrieved, relevant, k_values=[1])
        assert result["Recall@1"] == 1.0
        assert result["Precision@1"] == 1.0
        assert result["MRR"] == 1.0

    def test_empty_input(self):
        result = evaluate_mode([], [], k_values=[5])
        assert result == {}

    def test_averaged_over_queries(self):
        # Q1: found at rank 1, Q2: not found
        retrieved = [["A"], ["X"]]
        relevant  = [["A"], ["A"]]
        result = evaluate_mode(retrieved, relevant, k_values=[1])
        assert abs(result["Recall@1"] - 0.5) < 1e-9
        assert abs(result["MRR"] - 0.5) < 1e-9


# ── format_metrics_table ──────────────────────────────────────────────────────

class TestFormatMetricsTable:
    def test_returns_string(self):
        data = {
            "VECTOR_ONLY": {"Recall@5": 0.8, "MRR": 0.7},
            "HYBRID":      {"Recall@5": 0.9, "MRR": 0.8},
        }
        table = format_metrics_table(data, k_values=[5])
        assert isinstance(table, str)
        assert "VECTOR_ONLY" in table
        assert "HYBRID" in table

    def test_contains_metric_names(self):
        data = {"MODE_A": {"Recall@1": 0.5, "MRR": 0.6}}
        table = format_metrics_table(data, k_values=[1])
        assert "Recall@1" in table
        assert "MRR" in table
