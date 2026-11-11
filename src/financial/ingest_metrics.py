"""
src/financial/ingest_metrics.py — Populate financial_metrics_p6 from seed file.

Idempotent: re-running does not create duplicates.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from sqlalchemy.orm import Session

from src.financial.aliases import resolve_alias, get_all_aliases
from src.financial.parsing import parse_value
from src.financial.repository import get_company_id_by_name, upsert_metric, seed_aliases

logger = logging.getLogger(__name__)

SEED_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "financial" / "manual_seed_metrics.yaml"


def load_seed(path: Path = SEED_FILE) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def ingest_from_seed(session: Session, seed_data: Optional[Dict] = None) -> Dict[str, int]:
    """
    Ingest all metrics from seed YAML.
    Returns {"inserted": N, "skipped": N, "failed": N}
    """
    if seed_data is None:
        seed_data = load_seed()

    n_aliases = seed_aliases(session, get_all_aliases())
    logger.info(f"Seeded {n_aliases} metric aliases")

    stats = {"inserted": 0, "skipped": 0, "failed": 0}

    for entry in seed_data.get("metrics", []):
        try:
            row = _build_row(session, entry)
            if row is None:
                stats["skipped"] += 1
                continue
            upsert_metric(session, row)
            stats["inserted"] += 1
            logger.debug(f"Upserted: {row['metric_name']} FY{row['fiscal_year']}/{row['fiscal_period']}")
        except Exception as exc:
            logger.warning(f"Failed entry {entry.get('metric','?')}: {exc}")
            stats["failed"] += 1

    session.commit()
    return stats


def _build_row(session: Session, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a DB-ready dict from a seed YAML entry. Returns None to skip."""
    company_name = entry.get("company", "")
    company_id = get_company_id_by_name(session, company_name)
    if company_id is None:
        logger.warning(f"Company not found: '{company_name}' — skipping")
        return None

    metric_raw = entry.get("metric", "")
    canonical = resolve_alias(metric_raw)
    if canonical is None:
        logger.warning(f"Unknown metric '{metric_raw}' — skipping")
        return None

    raw_value = str(entry.get("value_raw", ""))
    parsed = parse_value(raw_value)
    if parsed is None:
        logger.warning(f"Cannot parse value '{raw_value}' for {metric_raw} — skipping")
        return None

    return {
        "company_id":          company_id,
        "fiscal_year":         int(entry["fiscal_year"]),
        "fiscal_period":       str(entry.get("fiscal_period", "FY")).upper(),
        "period_end_date":     entry.get("period_end_date"),
        "metric_name":         canonical,
        "value":               parsed.value,
        "unit":                entry.get("unit", parsed.unit),
        "scale":               entry.get("scale", parsed.scale),
        "dimension_type":      entry.get("dimension_type"),
        "dimension_value":     entry.get("dimension_value"),
        "is_derived":          bool(entry.get("is_derived", False)),
        "document_version_id": entry.get("document_version_id"),
        "page":                entry.get("page"),
        "extraction_method":   entry.get("extraction_method", "MANUAL_SEED"),
        "source_text":         entry.get("source_text", raw_value),
    }
