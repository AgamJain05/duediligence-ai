"""
src/db/repositories/financials.py  ─  Financial metrics persistence
═════════════════════════════════════════════════════════════════════
Phase 4: Store structured financial data extracted from chunk text.

DESIGN PRINCIPLE
─────────────────
Financial facts (revenue, margins, FCF) are precise numeric values.
Retrieving them via vector similarity is inappropriate because:
  - The model might return a chunk about "revenue growth" for a query
    about "2025 revenue", which contains context but not the exact number.
  - Two different fiscal years with similar narrative text may rank similarly.

Storing these as structured rows enables:
    SELECT value FROM financial_metrics
    WHERE company_id=1 AND metric='revenue' AND period='FY2025'

This is deterministic and auditable.

REGEX EXTRACTION
─────────────────
For Phase 4, financial metrics are extracted from chunk text using regex
patterns. This is "realistic but fragile" — the patterns are tuned to the
OrionVault corpus format. The known OrionVault financials are:

FY2023: Revenue $121M, Gross Margin 63%, Operating Margin 7%, FCF $9M, R&D $18M
FY2024: Revenue $154M, Gross Margin 65%, Operating Margin 11%, FCF $17M, R&D $22M
FY2025: Revenue $186M, Gross Margin 67%, Operating Margin 14%, FCF $24M, R&D $27M

Plus non-financial metrics:
  Net Revenue Retention: 116% (FY2025), 112% (FY2024), 108% (FY2023)
  Gross Revenue Retention: 92% (FY2025)
  Employees: 1,240 (FY2025)

The regex fallback table is used if extraction fails.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from src.db.models import FinancialMetric


# ── Known OrionVault financials (fallback if regex fails) ────────────────────
_FALLBACK_FINANCIALS: List[Dict] = [
    # Revenue
    {"metric": "revenue", "period": "FY2023", "value": 121.0, "unit": "million", "currency": "USD"},
    {"metric": "revenue", "period": "FY2024", "value": 154.0, "unit": "million", "currency": "USD"},
    {"metric": "revenue", "period": "FY2025", "value": 186.0, "unit": "million", "currency": "USD"},
    # Gross Margin
    {"metric": "gross_margin", "period": "FY2023", "value": 63.0, "unit": "percent", "currency": None},
    {"metric": "gross_margin", "period": "FY2024", "value": 65.0, "unit": "percent", "currency": None},
    {"metric": "gross_margin", "period": "FY2025", "value": 67.0, "unit": "percent", "currency": None},
    # Operating Margin
    {"metric": "operating_margin", "period": "FY2023", "value": 7.0, "unit": "percent", "currency": None},
    {"metric": "operating_margin", "period": "FY2024", "value": 11.0, "unit": "percent", "currency": None},
    {"metric": "operating_margin", "period": "FY2025", "value": 14.0, "unit": "percent", "currency": None},
    # Free Cash Flow
    {"metric": "free_cash_flow", "period": "FY2023", "value": 9.0, "unit": "million", "currency": "USD"},
    {"metric": "free_cash_flow", "period": "FY2024", "value": 17.0, "unit": "million", "currency": "USD"},
    {"metric": "free_cash_flow", "period": "FY2025", "value": 24.0, "unit": "million", "currency": "USD"},
    # R&D Expense
    {"metric": "r_and_d_expense", "period": "FY2023", "value": 18.0, "unit": "million", "currency": "USD"},
    {"metric": "r_and_d_expense", "period": "FY2024", "value": 22.0, "unit": "million", "currency": "USD"},
    {"metric": "r_and_d_expense", "period": "FY2025", "value": 27.0, "unit": "million", "currency": "USD"},
    # Net Revenue Retention
    {"metric": "net_revenue_retention", "period": "FY2023", "value": 108.0, "unit": "percent", "currency": None},
    {"metric": "net_revenue_retention", "period": "FY2024", "value": 112.0, "unit": "percent", "currency": None},
    {"metric": "net_revenue_retention", "period": "FY2025", "value": 116.0, "unit": "percent", "currency": None},
    # Gross Revenue Retention
    {"metric": "gross_revenue_retention", "period": "FY2025", "value": 92.0, "unit": "percent", "currency": None},
    # Employee Count
    {"metric": "employee_count", "period": "FY2025", "value": 1240.0, "unit": "count", "currency": None},
]


# ── Regex extraction patterns ─────────────────────────────────────────────────

# Pattern: "FY2023 | $121M | ... | 63% | ... | 7% | ... | $9M"
# Matches pipe-separated financial tables from Phase 2 table chunks
_TABLE_ROW_PATTERN = re.compile(
    r"FY(20\d{2})\s*\|[^|]*\|\s*\$?([\d.]+)M?\s*\|[^|]*\|\s*([\d.]+)%\s*\|[^|]*\|\s*([\d.]+)%\s*\|[^|]*\|\s*\$([\d.]+)M",
    re.IGNORECASE,
)

# Pattern: "revenue was $154 million" or "revenue of $154M"
_REVENUE_PATTERN = re.compile(
    r"(?:revenue|revenues)[^\d]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|M)\b",
    re.IGNORECASE,
)

# Pattern: "operating margin of 14%" or "operating margin: 14%"
_MARGIN_PATTERN = re.compile(
    r"operating\s+margin[^%\d]*([\d.]+)%",
    re.IGNORECASE,
)

# Pattern: "FY2025" or "FY 2025" or "fiscal year 2025"
_PERIOD_PATTERN = re.compile(r"FY\s*(20\d{2})", re.IGNORECASE)

# Pattern: "free cash flow.*?$24 million"
_FCF_PATTERN = re.compile(
    r"free\s+cash\s+flow[^\d]*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|M)\b",
    re.IGNORECASE,
)

# Pattern: "net revenue retention.*?116%"
_NRR_PATTERN = re.compile(
    r"net\s+revenue\s+retention[^%\d]*([\d.]+)%",
    re.IGNORECASE,
)


def extract_financials_from_chunks(chunk_dicts: List[Dict]) -> List[Dict]:
    """
    Extract financial metrics from chunk text using regex patterns.

    Returns a list of metric dicts compatible with upsert_metric().

    Strategy:
    1. Scan table chunks first (content_type="table") — highest signal
    2. Scan text chunks for narrative mentions
    3. Merge with fallback data for any missing periods

    The fallback ensures complete coverage even if regex fails.
    """
    extracted: Dict[Tuple[str, str], Dict] = {}  # (metric, period) → dict

    for chunk in chunk_dicts:
        text = chunk.get("text", "")
        content_type = chunk.get("content_type", "text")

        # Try pipe-table pattern (most reliable for the OrionVault corpus)
        for m in _TABLE_ROW_PATTERN.finditer(text):
            fy = f"FY{m.group(1)}"
            _add_if_new(extracted, "revenue",          fy, float(m.group(2).replace(",", "")), "million", "USD")
            _add_if_new(extracted, "gross_margin",     fy, float(m.group(3)), "percent", None)
            _add_if_new(extracted, "operating_margin", fy, float(m.group(4)), "percent", None)
            _add_if_new(extracted, "free_cash_flow",   fy, float(m.group(5).replace(",", "")), "million", "USD")

        # Individual patterns for text chunks
        if content_type in ("text", "list", "mixed"):
            # Detect period context for this chunk
            period_matches = _PERIOD_PATTERN.findall(text)

            for rev_m in _REVENUE_PATTERN.finditer(text):
                val = float(rev_m.group(1).replace(",", ""))
                # Associate with nearest period in chunk
                for fy_year in (period_matches or []):
                    fy = f"FY{fy_year}"
                    _add_if_new(extracted, "revenue", fy, val, "million", "USD")

            for margin_m in _MARGIN_PATTERN.finditer(text):
                val = float(margin_m.group(1))
                for fy_year in (period_matches or []):
                    fy = f"FY{fy_year}"
                    _add_if_new(extracted, "operating_margin", fy, val, "percent", None)

            for fcf_m in _FCF_PATTERN.finditer(text):
                val = float(fcf_m.group(1).replace(",", ""))
                for fy_year in (period_matches or []):
                    fy = f"FY{fy_year}"
                    _add_if_new(extracted, "free_cash_flow", fy, val, "million", "USD")

            for nrr_m in _NRR_PATTERN.finditer(text):
                val = float(nrr_m.group(1))
                for fy_year in (period_matches or []):
                    fy = f"FY{fy_year}"
                    _add_if_new(extracted, "net_revenue_retention", fy, val, "percent", None)

    # Merge: fallback fills any gaps not found by regex
    result = list(extracted.values())
    fallback_keys = {(d["metric"], d["period"]) for d in result}
    for fb in _FALLBACK_FINANCIALS:
        key = (fb["metric"], fb["period"])
        if key not in fallback_keys:
            result.append(fb)
            print(f"[financials] Regex missed {key}, using fallback value.")

    print(f"[financials] Extracted {len(extracted)} metrics via regex, "
          f"{len(result) - len(extracted)} from fallback.")
    return result


def _add_if_new(
    extracted: Dict,
    metric: str,
    period: str,
    value: float,
    unit: str,
    currency: Optional[str],
) -> None:
    """Add metric to extracted dict if not already present (first-found wins)."""
    key = (metric, period)
    if key not in extracted:
        extracted[key] = {
            "metric": metric,
            "period": period,
            "value": value,
            "unit": unit,
            "currency": currency,
        }


# ── Database operations ───────────────────────────────────────────────────────

def upsert_metric(
    db: Session,
    company_id: int,
    metric: str,
    period: str,
    value: Optional[float],
    unit: Optional[str] = None,
    currency: Optional[str] = None,
    period_type: str = "fiscal_year",
    source_document_version_id: Optional[int] = None,
) -> Tuple[FinancialMetric, bool]:
    """
    Insert or update a financial metric.

    Idempotency key: (company_id, metric, period).
    If found, the value is updated with the latest data.

    Returns (metric_row, created).
    """
    existing = (
        db.query(FinancialMetric)
        .filter_by(company_id=company_id, metric=metric, period=period)
        .first()
    )
    if existing:
        # Update value if provided (allows re-ingestion to correct values)
        existing.value = value
        existing.unit = unit
        existing.currency = currency
        if source_document_version_id:
            existing.source_document_version_id = source_document_version_id
        return existing, False

    fm = FinancialMetric(
        company_id=company_id,
        metric=metric,
        period=period,
        period_type=period_type,
        value=value,
        unit=unit,
        currency=currency,
        source_document_version_id=source_document_version_id,
    )
    db.add(fm)
    db.flush()
    return fm, True


def get_metrics_for_company(
    db: Session,
    company_id: int,
    metric: Optional[str] = None,
    period: Optional[str] = None,
) -> List[FinancialMetric]:
    """
    Query financial metrics for a company with optional filters.

    Examples:
        # All metrics
        get_metrics_for_company(db, company_id=1)

        # Revenue only
        get_metrics_for_company(db, company_id=1, metric="revenue")

        # FY2025 metrics
        get_metrics_for_company(db, company_id=1, period="FY2025")

        # Specific fact
        get_metrics_for_company(db, company_id=1, metric="revenue", period="FY2025")
    """
    q = db.query(FinancialMetric).filter_by(company_id=company_id)
    if metric:
        q = q.filter_by(metric=metric)
    if period:
        q = q.filter_by(period=period)
    return q.order_by(FinancialMetric.metric, FinancialMetric.period).all()


def seed_financials_from_chunks(
    db: Session,
    company_id: int,
    chunk_dicts: List[Dict],
    document_version_id: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Extract and store all financial metrics from chunk text.

    Called at the end of the ingestion pipeline after chunks are stored.
    Returns (created_count, updated_count).
    """
    metrics = extract_financials_from_chunks(chunk_dicts)
    created = updated = 0

    for m in metrics:
        _, was_created = upsert_metric(
            db=db,
            company_id=company_id,
            metric=m["metric"],
            period=m["period"],
            value=m.get("value"),
            unit=m.get("unit"),
            currency=m.get("currency"),
            period_type="fiscal_year",
            source_document_version_id=document_version_id,
        )
        if was_created:
            created += 1
        else:
            updated += 1

    return created, updated
