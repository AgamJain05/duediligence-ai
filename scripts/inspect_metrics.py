#!/usr/bin/env python3
"""
scripts/inspect_metrics.py — Inspect financial_metrics_p6 table.

Usage:
    python scripts/inspect_metrics.py --company NexaCore --metric revenue
    python scripts/inspect_metrics.py --company NexaCore --metric gross_margin --all-years
    python scripts/inspect_metrics.py --company NexaCore --metric regional_revenue_share --dim-type REGION --dim-value China
    python scripts/inspect_metrics.py --list-companies
    python scripts/inspect_metrics.py --list-metrics
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Inspect financial_metrics_p6")
    parser.add_argument("--company",    default=None)
    parser.add_argument("--metric",     default=None)
    parser.add_argument("--year",       type=int, default=None)
    parser.add_argument("--period",     default=None, help="FY, Q4, etc.")
    parser.add_argument("--dim-type",   default=None, dest="dim_type")
    parser.add_argument("--dim-value",  default=None, dest="dim_value")
    parser.add_argument("--all-years",  action="store_true")
    parser.add_argument("--list-companies", action="store_true")
    parser.add_argument("--list-metrics",   action="store_true")
    args = parser.parse_args()

    try:
        from src.db.connection import get_db
    except EnvironmentError as e:
        print(f"DB not configured: {e}")
        sys.exit(1)

    with get_db() as session:
        if args.list_companies:
            from src.financial.repository import list_known_companies
            companies = list_known_companies(session)
            print("Known companies:")
            for c in companies:
                print(f"  {c}")
            return

        if args.list_metrics:
            from src.financial.aliases import CANONICAL_METRICS
            print("Canonical metrics:")
            for m in CANONICAL_METRICS:
                print(f"  {m}")
            return

        if not args.company or not args.metric:
            parser.print_help()
            return

        from src.financial.repository import get_company_id_by_name, get_metrics, detect_conflicts
        company_id = get_company_id_by_name(session, args.company)
        if company_id is None:
            print(f"Company not found: {args.company}")
            return

        rows = get_metrics(
            session, company_id, args.metric,
            fiscal_year=args.year if not args.all_years else None,
            fiscal_period=args.period,
            dimension_type=args.dim_type,
            dimension_value=args.dim_value,
        )

        if not rows:
            print(f"No rows found for {args.company} / {args.metric}")
            return

        conflict = detect_conflicts(rows)
        if conflict:
            print(f"⚠  CONFLICT DETECTED — {len(rows)} rows with different values")

        print(f"\n{'FY':<6} {'Period':<6} {'Value':<12} {'Unit':<18} {'Scale':<10} {'DimType':<10} {'DimValue':<18} {'Src'}")
        print("-" * 100)
        for r in rows:
            dim_t = r.dimension_type or ""
            dim_v = r.dimension_value or ""
            src = r.source_text[:40]
            print(f"{r.fiscal_year:<6} {r.fiscal_period:<6} {str(r.value):<12} {r.unit:<18} {r.scale:<10} {dim_t:<10} {dim_v:<18} {src}")


if __name__ == "__main__":
    main()
