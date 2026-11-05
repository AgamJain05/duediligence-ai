"""
tests/planning/test_analyzer.py  ─  QueryAnalyzer tests
══════════════════════════════════════════════════════════
Tests:
  - SIMPLE classification for clear factual questions
  - COMPARISON detection for compare/vs/contrast queries
  - TEMPORAL detection for date-range / change-over-time queries
  - MULTI_HOP detection for causal chain queries
  - CROSS_DOCUMENT detection for management vs disclosure queries
  - Simple queries do NOT have requires_decomposition=True
  - Complex queries DO have requires_decomposition=True
  - Entity extraction
  - Temporal constraint extraction
  - LLM fallback disabled (rules-only mode)
"""

from __future__ import annotations

import pytest
from src.planning.analyzer import QueryAnalyzer
from src.planning.models import PlanningConfig, QueryType

# Use rules-only config for all tests (no LLM API calls)
RULES_CONFIG = PlanningConfig(allow_llm_classifier=False, allow_llm_planner=False)


@pytest.fixture
def analyzer():
    return QueryAnalyzer(config=RULES_CONFIG)


class TestSimpleClassification:
    """Simple factual questions must NOT be decomposed."""

    def test_revenue_query(self, analyzer):
        a = analyzer.analyze("What is OrionVault's FY2025 revenue?")
        assert a.query_type == QueryType.SIMPLE
        assert a.requires_decomposition is False

    def test_gross_margin_query(self, analyzer):
        a = analyzer.analyze("What is OrionVault's gross margin?")
        assert a.query_type == QueryType.SIMPLE
        assert a.requires_decomposition is False

    def test_who_is_ceo(self, analyzer):
        a = analyzer.analyze("Who is OrionVault's CEO?")
        assert a.query_type == QueryType.SIMPLE
        assert a.requires_decomposition is False

    def test_customer_count(self, analyzer):
        a = analyzer.analyze("How many customers does OrionVault have?")
        assert a.query_type == QueryType.SIMPLE
        assert a.requires_decomposition is False

    def test_what_is_product(self, analyzer):
        a = analyzer.analyze("What is Sentinel Copilot?")
        assert a.query_type == QueryType.SIMPLE
        assert a.requires_decomposition is False


class TestComparisonClassification:
    """Compare/vs/contrast queries → COMPARISON."""

    def test_compare_explicit(self, analyzer):
        a = analyzer.analyze("Compare OrionVault's US and European expansion strategies.")
        assert a.query_type == QueryType.COMPARISON
        assert a.requires_decomposition is True

    def test_versus(self, analyzer):
        a = analyzer.analyze("OrionVault Sentinel Copilot vs VaultGuard — which has more growth potential?")
        assert a.query_type == QueryType.COMPARISON
        assert a.requires_decomposition is True

    def test_difference_between(self, analyzer):
        a = analyzer.analyze("What is the difference between OrionVault's enterprise and SMB strategies?")
        assert a.query_type == QueryType.COMPARISON
        assert a.requires_decomposition is True


class TestTemporalClassification:
    """Queries spanning two time periods → TEMPORAL."""

    def test_from_to(self, analyzer):
        a = analyzer.analyze("How did OrionVault's risk profile change from 2024 to 2026?")
        assert a.query_type == QueryType.TEMPORAL
        assert a.requires_decomposition is True

    def test_change_over_time(self, analyzer):
        a = analyzer.analyze("How has OrionVault's gross margin changed over time?")
        assert a.query_type == QueryType.TEMPORAL
        assert a.requires_decomposition is True

    def test_temporal_constraints_extracted(self, analyzer):
        a = analyzer.analyze("What changed between FY2023 and FY2025?")
        assert a.query_type in {QueryType.TEMPORAL, QueryType.COMPARISON}
        # Should extract years
        temporal = " ".join(a.temporal_constraints)
        assert "2023" in temporal or "2025" in temporal


class TestMultiHopClassification:
    """Causal chain queries → MULTI_HOP."""

    def test_what_caused_and_what_risks(self, analyzer):
        a = analyzer.analyze(
            "What factors caused the margin improvement and what risks could reverse it?"
        )
        assert a.query_type == QueryType.MULTI_HOP
        assert a.requires_decomposition is True


class TestCrossDocumentClassification:
    """Management commentary vs formal disclosures → CROSS_DOCUMENT."""

    def test_management_vs_risk(self, analyzer):
        a = analyzer.analyze(
            "Compare management's commentary on Sentinel Copilot with the formal risk disclosures."
        )
        assert a.query_type in {QueryType.CROSS_DOCUMENT, QueryType.COMPARISON}
        assert a.requires_decomposition is True


class TestEntityExtraction:
    """Named entities are extracted from the query."""

    def test_company_extracted(self, analyzer):
        a = analyzer.analyze("What is OrionVault's revenue?")
        assert "OrionVault" in a.entities

    def test_product_extracted(self, analyzer):
        a = analyzer.analyze("Tell me about Sentinel Copilot.")
        assert "Sentinel" in a.entities or "Sentinel Copilot" in " ".join(a.entities)

    def test_geography_extracted(self, analyzer):
        a = analyzer.analyze("What is OrionVault's European strategy?")
        assert "Europe" in a.entities


class TestTopicExtraction:
    """Topic areas are extracted from the query."""

    def test_financial_topic(self, analyzer):
        a = analyzer.analyze("What is OrionVault's FY2025 revenue?")
        assert "financial performance" in a.topics

    def test_strategy_topic(self, analyzer):
        a = analyzer.analyze("Describe OrionVault's expansion strategy.")
        assert "growth strategy" in a.topics

    def test_risk_topic(self, analyzer):
        a = analyzer.analyze("What are the key risks facing OrionVault?")
        assert "risk analysis" in a.topics


class TestDecompositionFlag:
    """SIMPLE → False, everything else → True."""

    @pytest.mark.parametrize("query", [
        "What is FY2025 revenue?",
        "Who is the CEO?",
        "How many employees are there?",
    ])
    def test_simple_no_decompose(self, analyzer, query):
        a = analyzer.analyze(query)
        assert a.requires_decomposition is False

    @pytest.mark.parametrize("query", [
        "Compare US and European strategies.",
        "How did the risk profile change from 2024 to 2026?",
    ])
    def test_complex_requires_decompose(self, analyzer, query):
        a = analyzer.analyze(query)
        assert a.requires_decomposition is True


class TestAnalysisModel:
    """QueryAnalysis model enforces constraints."""

    def test_simple_type_forces_no_decompose(self, analyzer):
        a = analyzer.analyze("What is OrionVault's FY2025 revenue?")
        # Even if rules were wrong, the model validator enforces SIMPLE → no decompose
        assert a.query_type == QueryType.SIMPLE
        assert a.requires_decomposition is False
        assert a.expected_subquery_count == 1

    def test_expected_subquery_count_ge_1(self, analyzer):
        a = analyzer.analyze("Any question at all.")
        assert a.expected_subquery_count >= 1

    def test_confidence_in_range(self, analyzer):
        a = analyzer.analyze("What is OrionVault's FY2025 revenue?")
        assert 0.0 <= a.confidence <= 1.0
