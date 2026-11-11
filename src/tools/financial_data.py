"""
src/tools/financial_data.py — Parameterized financial metric lookup tool.

Whitelisted lookup — NOT free-form SQL.
Returns ALL matching rows; surfaces conflicts explicitly.
Never picks a winner when sources disagree.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool, ToolError, ToolResult, Provenance
from src.financial.aliases import resolve_alias, suggest_canonical


class FinancialDataTool(BaseTool):
    name = "search_financial_data"
    description = (
        "Look up structured financial metrics from the database. "
        "Accepts company name, metric name (alias-resolved), fiscal year, fiscal period, "
        "and optional dimension filters. Returns ALL matching rows with provenance. "
        "If multiple sources report different values for the same metric, "
        "conflict_detected=true and all values are returned — do NOT resolve conflicts here."
    )

    def __init__(self, db_session=None):
        self._session = db_session

    def run(
        self,
        company: str,
        metric: str,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        dimension_type: Optional[str] = None,
        dimension_value: Optional[str] = None,
    ) -> ToolResult:
        if self._session is None:
            return ToolResult(
                tool=self.name, status="ERROR",
                error=ToolError(code="NO_DB", message="Database session not available."),
            )

        # Alias resolution
        canonical = resolve_alias(metric)
        if canonical is None:
            suggestions = suggest_canonical(metric)
            return ToolResult(
                tool=self.name, status="EMPTY",
                error=ToolError(
                    code="UNKNOWN_METRIC",
                    message=(
                        f"Metric '{metric}' not recognised. "
                        f"Did you mean: {suggestions}? "
                        f"Use scripts/inspect_tools.py to list canonical names."
                    ),
                ),
                data={"suggestions": suggestions},
            )

        # Company lookup
        from src.financial.repository import get_company_id_by_name, list_known_companies
        company_id = get_company_id_by_name(self._session, company)
        if company_id is None:
            known = list_known_companies(self._session)
            return ToolResult(
                tool=self.name, status="ERROR",
                error=ToolError(
                    code="UNKNOWN_COMPANY",
                    message=f"Company '{company}' not found. Known companies: {known}",
                ),
                data={"known_companies": known},
            )

        # Metric lookup
        from src.financial.repository import get_metrics, detect_conflicts
        rows = get_metrics(
            self._session, company_id, metric,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            dimension_type=dimension_type,
            dimension_value=dimension_value,
        )

        if not rows:
            return ToolResult(
                tool=self.name, status="EMPTY",
                data={
                    "company":       company,
                    "metric":        canonical,
                    "fiscal_year":   fiscal_year,
                    "fiscal_period": fiscal_period,
                    "rows":          [],
                },
            )

        conflict = detect_conflicts(rows)
        rows_data = [r.to_dict() for r in rows]
        provenance = [
            Provenance(
                source_type="metric_row",
                metric_id=r.id,
                fiscal_year=r.fiscal_year,
                fiscal_period=r.fiscal_period,
                source_text=r.source_text,
            )
            for r in rows
        ]

        return ToolResult(
            tool=self.name,
            status="CONFLICT" if conflict else "SUCCESS",
            data={
                "company":           company,
                "metric":            canonical,
                "fiscal_year":       fiscal_year,
                "fiscal_period":     fiscal_period,
                "rows":              rows_data,
                "conflict_detected": conflict,
                "conflict_note": (
                    "Multiple sources report different values. "
                    "All values shown with provenance. Do NOT resolve — surface to user."
                ) if conflict else None,
            },
            provenance=provenance,
            conflict_detected=conflict,
        )
