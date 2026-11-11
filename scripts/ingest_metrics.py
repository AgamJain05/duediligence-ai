#!/usr/bin/env python3
"""
scripts/ingest_metrics.py — Populate financial_metrics_p6 from the seed YAML.

Usage:
    python scripts/ingest_metrics.py
    python scripts/ingest_metrics.py --seed data/financial/manual_seed_metrics.yaml
    python scripts/ingest_metrics.py --dry-run

Idempotent: re-running does not create duplicates.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest financial metrics from seed YAML")
    parser.add_argument("--seed", default=None, help="Path to seed YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not write to DB")
    args = parser.parse_args()

    if args.dry_run:
        from src.financial.ingest_metrics import load_seed, SEED_FILE
        from src.financial.parsing import parse_value
        from src.financial.aliases import resolve_alias
        path = Path(args.seed) if args.seed else SEED_FILE
        data = load_seed(path)
        entries = data.get("metrics", [])
        ok = failed = 0
        for e in entries:
            metric = e.get("metric", "")
            raw = str(e.get("value_raw", ""))
            canonical = resolve_alias(metric)
            parsed = parse_value(raw)
            if canonical and parsed:
                ok += 1
            else:
                failed += 1
                logger.warning(f"Would skip: metric='{metric}' value='{raw}'")
        print(f"Dry run: {ok} OK, {failed} would be skipped out of {len(entries)} entries")
        return

    try:
        from src.db.connection import get_db
    except EnvironmentError as e:
        logger.error(f"Database not configured: {e}")
        sys.exit(1)

    from src.financial.ingest_metrics import load_seed, ingest_from_seed, SEED_FILE
    path = Path(args.seed) if args.seed else SEED_FILE
    seed_data = load_seed(path)

    with get_db() as session:
        stats = ingest_from_seed(session, seed_data)

    print(f"Ingestion complete: {stats}")


if __name__ == "__main__":
    main()
