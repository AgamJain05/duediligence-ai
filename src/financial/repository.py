"""
src/financial/repository.py — DB operations for financial_metrics_p6 and metric_aliases.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.financial.aliases import resolve_alias
from src.financial.models import FinancialMetricP6, MetricAlias


def get_metrics(
    session: Session,
    company_id: int,
    metric_name: str,
    fiscal_year: Optional[int] = None,
    fiscal_period: Optional[str] = None,
    dimension_type: Optional[str] = None,
    dimension_value: Optional[str] = None,
) -> List[FinancialMetricP6]:
    """
    Return ALL matching rows — multiple rows mean conflict (different sources, same metric).
    Callers must NOT pick a winner; they must surface all values.
    """
    canonical = resolve_alias(metric_name)
    if canonical is None:
        return []

    q = select(FinancialMetricP6).where(
        FinancialMetricP6.company_id == company_id,
        FinancialMetricP6.metric_name == canonical,
    )
    if fiscal_year is not None:
        q = q.where(FinancialMetricP6.fiscal_year == fiscal_year)
    if fiscal_period is not None:
        q = q.where(FinancialMetricP6.fiscal_period == fiscal_period.upper())
    if dimension_type is not None:
        q = q.where(FinancialMetricP6.dimension_type == dimension_type)
    if dimension_value is not None:
        q = q.where(FinancialMetricP6.dimension_value == dimension_value)

    return list(session.execute(q).scalars().all())


def detect_conflicts(rows: List[FinancialMetricP6]) -> bool:
    """True if rows contain different values for the same metric."""
    if len(rows) <= 1:
        return False
    return len({str(r.value) for r in rows}) > 1


def get_company_id_by_name(session: Session, name: str) -> Optional[int]:
    from src.db.models import Company
    result = session.execute(
        select(Company.id).where(Company.name.ilike(f"%{name}%"))
    ).scalar_one_or_none()
    return result


def list_known_companies(session: Session) -> List[str]:
    from src.db.models import Company
    return list(session.execute(select(Company.name)).scalars().all())


def upsert_metric(session: Session, row: Dict[str, Any]) -> None:
    """
    Insert a metric row. On conflict (same source+metric+period), do nothing (idempotent).
    Uses SQLAlchemy core insert with on_conflict_do_nothing.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    stmt = pg_insert(FinancialMetricP6).values(**row).on_conflict_do_nothing()
    session.execute(stmt)


def seed_aliases(session: Session, alias_map: Dict[str, str]) -> int:
    """Insert aliases; skip existing. Returns count inserted."""
    count = 0
    for alias, canonical in alias_map.items():
        if session.get(MetricAlias, alias) is None:
            session.add(MetricAlias(alias=alias, canonical_name=canonical))
            count += 1
    return count
