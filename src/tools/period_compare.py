"""
src/tools/period_compare.py — Composed tool: compare a metric across two periods.

Internally uses search_financial_data + calculate_metric.
Deterministic — no LLM involvement.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from src.tools.base import BaseTool, ToolError, ToolResult, Provenance


class PeriodCompareTool(BaseTool):
    name = "compare_periods"
    description = (
        "Compare a financial metric between two fiscal periods for one company. "
        "Returns both values, absolute change, percent change, and provenance. "
        "Internally uses search_financial_data + calculate_metric. "
        "Example: compare NexaCore gross_margin FY2023 vs FY2025."
    )

    def __init__(self, db_session=None):
        self._session = db_session

    def run(
        self,
        company: str,
        metric: str,
        period_a: str,
        period_b: str,
        fiscal_period: str = "FY",
    ) -> ToolResult:
        """
        Args:
            company:       Company name.
            metric:        Metric name (alias-resolved internally).
            period_a:      First fiscal year (earlier), e.g. "2023".
            period_b:      Second fiscal year (later), e.g. "2025".
            fiscal_period: "FY" | "Q1" | "Q2" | "Q3" | "Q4" | "CY"
        """
        from src.tools.financial_data import FinancialDataTool
        from src.tools.calculator import CalculatorTool

        fd = FinancialDataTool(db_session=self._session)
        calc = CalculatorTool()

        try:
            year_a = int(period_a)
            year_b = int(period_b)
        except ValueError:
            return ToolResult(
                tool=self.name, status="ERROR",
                error=ToolError(code="INVALID_PERIOD",
                                message=f"period_a and period_b must be integer years: '{period_a}', '{period_b}'"),
            )

        # Fetch both periods
        result_a = fd.safe_run(company=company, metric=metric,
                               fiscal_year=year_a, fiscal_period=fiscal_period)
        result_b = fd.safe_run(company=company, metric=metric,
                               fiscal_year=year_b, fiscal_period=fiscal_period)

        if not result_a.is_ok() or not result_a.data.get("rows"):
            return ToolResult(
                tool=self.name, status="EMPTY",
                error=ToolError(code="PERIOD_A_MISSING",
                                message=f"No data for {company} {metric} {fiscal_period}{period_a}"),
            )
        if not result_b.is_ok() or not result_b.data.get("rows"):
            return ToolResult(
                tool=self.name, status="EMPTY",
                error=ToolError(code="PERIOD_B_MISSING",
                                message=f"No data for {company} {metric} {fiscal_period}{period_b}"),
            )

        # Use first row from each (flag if conflict within period)
        rows_a = result_a.data["rows"]
        rows_b = result_b.data["rows"]

        val_a = Decimal(str(rows_a[0]["value"]))
        val_b = Decimal(str(rows_b[0]["value"]))
        unit  = rows_a[0].get("unit", "")
        scale = rows_a[0].get("scale", "UNITS")

        # Calculate change
        calc_result = calc.safe_run(
            operation="percent_change",
            inputs=[
                {"label": f"{metric} {fiscal_period}{period_a}", "value": str(val_a), "unit": unit,
                 "provenance": {"source_type": "metric_row",
                                "fiscal_year": year_a, "fiscal_period": fiscal_period,
                                "source_text": rows_a[0].get("source_text", "")}},
                {"label": f"{metric} {fiscal_period}{period_b}", "value": str(val_b), "unit": unit,
                 "provenance": {"source_type": "metric_row",
                                "fiscal_year": year_b, "fiscal_period": fiscal_period,
                                "source_text": rows_b[0].get("source_text", "")}},
            ],
        )

        diff_result = calc.safe_run(
            operation="difference",
            inputs=[
                {"label": f"{metric} {fiscal_period}{period_a}", "value": str(val_a), "unit": unit},
                {"label": f"{metric} {fiscal_period}{period_b}", "value": str(val_b), "unit": unit},
            ],
        )

        return ToolResult(
            tool=self.name,
            status="SUCCESS",
            data={
                "company":        company,
                "metric":         metric,
                "period_a":       f"{fiscal_period}{period_a}",
                "period_b":       f"{fiscal_period}{period_b}",
                "value_a":        str(val_a),
                "value_b":        str(val_b),
                "unit":           unit,
                "scale":          scale,
                "absolute_change": diff_result.data.get("result") if diff_result.is_ok() else None,
                "percent_change":  calc_result.data.get("result") if calc_result.is_ok() else None,
                "formula":         calc_result.data.get("formula"),
                "conflict_period_a": result_a.conflict_detected,
                "conflict_period_b": result_b.conflict_detected,
            },
            provenance=result_a.provenance + result_b.provenance,
        )
