"""
src/financial/aliases.py — Metric alias resolution and canonical vocabulary.
"""
from __future__ import annotations

from typing import Dict, List, Optional

CANONICAL_METRICS: List[str] = [
    "revenue", "gross_profit", "gross_margin", "rd_expense",
    "operating_income", "operating_margin", "net_income",
    "eps_diluted", "capex", "free_cash_flow", "cash_and_equivalents",
    "total_debt", "diluted_shares", "employees",
    "segment_revenue", "regional_revenue", "regional_revenue_share",
    "market_share", "customer_concentration_pct",
]

_ALIAS_MAP: Dict[str, str] = {
    "sales": "revenue", "total revenue": "revenue", "net revenue": "revenue",
    "total sales": "revenue", "net sales": "revenue", "revenues": "revenue",
    "gross profit": "gross_profit", "gp": "gross_profit",
    "gross margin": "gross_margin", "gm": "gross_margin",
    "gross margin %": "gross_margin", "gross margin pct": "gross_margin",
    "r&d": "rd_expense", "research and development": "rd_expense",
    "rd": "rd_expense", "r and d": "rd_expense",
    "research & development": "rd_expense", "r&d expense": "rd_expense",
    "operating income": "operating_income", "ebit": "operating_income",
    "operating profit": "operating_income",
    "operating margin": "operating_margin", "ebit margin": "operating_margin",
    "op margin": "operating_margin",
    "net income": "net_income", "profit": "net_income",
    "net profit": "net_income", "net earnings": "net_income",
    "eps": "eps_diluted", "diluted eps": "eps_diluted",
    "earnings per share": "eps_diluted",
    "diluted earnings per share": "eps_diluted",
    "capital expenditures": "capex", "capital expenditure": "capex",
    "cap ex": "capex",
    "fcf": "free_cash_flow", "free cash flow": "free_cash_flow",
    "free cashflow": "free_cash_flow",
    "cash": "cash_and_equivalents",
    "cash and equivalents": "cash_and_equivalents",
    "cash & equivalents": "cash_and_equivalents",
    "cash equivalents": "cash_and_equivalents",
    "debt": "total_debt", "total debt": "total_debt",
    "long-term debt": "total_debt",
    "shares": "diluted_shares", "diluted shares": "diluted_shares",
    "shares outstanding": "diluted_shares",
    "diluted shares outstanding": "diluted_shares",
    "headcount": "employees", "employee count": "employees", "staff": "employees",
    "segment revenue": "segment_revenue",
    "regional revenue": "regional_revenue", "geographic revenue": "regional_revenue",
    "regional share": "regional_revenue_share",
    "geographic share": "regional_revenue_share",
    "market share": "market_share",
    "customer concentration": "customer_concentration_pct",
}


def resolve_alias(metric: str) -> Optional[str]:
    """Resolve metric name or alias → canonical. Returns None if not found."""
    key = metric.strip().lower()
    if key in CANONICAL_METRICS:
        return key
    return _ALIAS_MAP.get(key)


def suggest_canonical(metric: str, max_suggestions: int = 3) -> List[str]:
    """Return closest canonical names for unknown metric (substring match)."""
    key = metric.strip().lower()
    return [c for c in CANONICAL_METRICS if key in c or c in key][:max_suggestions]


def get_all_aliases() -> Dict[str, str]:
    return dict(_ALIAS_MAP)
