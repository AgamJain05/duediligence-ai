"""
tests/db/test_financials.py  ─  Financial metrics tests
═════════════════════════════════════════════════════════
Tests:
  - Metric insertion and retrieval
  - Idempotent upsert (updates value, doesn't duplicate)
  - Regex extraction from chunk text
  - Fallback table used when regex misses
  - Filtering by metric name and period
"""

from __future__ import annotations

import pytest

from tests.db.conftest import requires_db


@requires_db
class TestMetricInsertion:
    """Basic CRUD for financial_metrics."""

    def test_insert_metric(self, db_session):
        from src.db.repositories.companies  import get_or_create_company
        from src.db.repositories.financials import upsert_metric

        co = get_or_create_company(db_session, "FinMetricCo")
        fm, created = upsert_metric(
            db_session,
            company_id=co.id,
            metric="revenue",
            period="FY2025",
            value=186.0,
            unit="million",
            currency="USD",
        )
        assert created is True
        assert fm.id is not None
        assert fm.value == 186.0
        assert fm.metric == "revenue"
        assert fm.period == "FY2025"

    def test_upsert_updates_value(self, db_session):
        from src.db.repositories.companies  import get_or_create_company
        from src.db.repositories.financials import upsert_metric, get_metrics_for_company

        co = get_or_create_company(db_session, "UpsertMetricCo")
        _, c1 = upsert_metric(db_session, co.id, "revenue", "FY2025", 180.0, "million", "USD")
        _, c2 = upsert_metric(db_session, co.id, "revenue", "FY2025", 186.0, "million", "USD")

        assert c1 is True
        assert c2 is False  # update, not insert

        metrics = get_metrics_for_company(db_session, co.id, metric="revenue", period="FY2025")
        assert len(metrics) == 1
        assert metrics[0].value == 186.0  # updated

    def test_filter_by_metric_name(self, db_session):
        from src.db.repositories.companies  import get_or_create_company
        from src.db.repositories.financials import upsert_metric, get_metrics_for_company

        co = get_or_create_company(db_session, "FilterMetricCo")
        upsert_metric(db_session, co.id, "revenue",          "FY2025", 186.0, "million", "USD")
        upsert_metric(db_session, co.id, "operating_margin", "FY2025", 14.0,  "percent", None)
        upsert_metric(db_session, co.id, "free_cash_flow",   "FY2025", 24.0,  "million", "USD")

        rev_only = get_metrics_for_company(db_session, co.id, metric="revenue")
        assert len(rev_only) == 1
        assert rev_only[0].metric == "revenue"

    def test_filter_by_period(self, db_session):
        from src.db.repositories.companies  import get_or_create_company
        from src.db.repositories.financials import upsert_metric, get_metrics_for_company

        co = get_or_create_company(db_session, "PeriodFilterCo")
        upsert_metric(db_session, co.id, "revenue", "FY2023", 121.0, "million", "USD")
        upsert_metric(db_session, co.id, "revenue", "FY2024", 154.0, "million", "USD")
        upsert_metric(db_session, co.id, "revenue", "FY2025", 186.0, "million", "USD")

        fy25 = get_metrics_for_company(db_session, co.id, period="FY2025")
        assert all(m.period == "FY2025" for m in fy25)


class TestRegexExtraction:
    """Regex extracts known OrionVault financials from chunk text."""

    def test_extracts_revenue_from_table_text(self):
        from src.db.repositories.financials import extract_financials_from_chunks

        table_chunk = {
            "text": (
                "| FY2023 | $121M | 63% | 7% | $9M |\n"
                "| FY2024 | $154M | 65% | 11% | $17M |\n"
                "| FY2025 | $186M | 67% | 14% | $24M |"
            ),
            "content_type": "table",
        }

        metrics = extract_financials_from_chunks([table_chunk])
        metric_map = {(m["metric"], m["period"]): m["value"] for m in metrics}

        assert ("revenue", "FY2025") in metric_map
        assert metric_map[("revenue", "FY2025")] == 186.0

    def test_fallback_used_for_missing_metrics(self):
        """If regex finds no metrics, fallback covers all known values."""
        from src.db.repositories.financials import extract_financials_from_chunks

        # Chunk with no financial data
        plain_chunk = {"text": "Sentinel Copilot AI product launched Q3 FY2026.", "content_type": "text"}
        metrics = extract_financials_from_chunks([plain_chunk])

        # Fallback should have all standard periods
        periods = {(m["metric"], m["period"]) for m in metrics}
        assert ("revenue", "FY2025") in periods
        assert ("revenue", "FY2024") in periods
        assert ("revenue", "FY2023") in periods
        assert ("operating_margin", "FY2025") in periods

    def test_no_duplicates_in_extraction(self):
        """Same metric+period should not appear twice even with multiple matching chunks."""
        from src.db.repositories.financials import extract_financials_from_chunks

        chunk1 = {"text": "FY2025 revenue $186M", "content_type": "text"}
        chunk2 = {"text": "FY2025 revenue approximately $186M", "content_type": "text"}

        metrics = extract_financials_from_chunks([chunk1, chunk2])
        fy25_revenue = [m for m in metrics if m["metric"] == "revenue" and m["period"] == "FY2025"]
        assert len(fy25_revenue) == 1, "Same metric+period should not be duplicated"
