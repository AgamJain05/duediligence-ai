"""
scripts/inspect_query.py  ─  Phase 4 query inspector
═══════════════════════════════════════════════════════
Runs a query against the PostgreSQL knowledge base and shows detailed results.

Includes:
  • SQL filters being applied
  • Candidate chunk count before vector search
  • Retrieved results with similarity scores, provenance, section, page
  • EXPLAIN ANALYZE output (with --explain flag)
  • Comparison of exact vs HNSW results

Usage:
    python scripts/inspect_query.py "What are the key risks?"
    python scripts/inspect_query.py "data residency requirement" --top-k 5
    python scripts/inspect_query.py "FY2025 revenue" --filter-current
    python scripts/inspect_query.py "risks" --company "OrionVault" --doc-type annual_review
    python scripts/inspect_query.py "key risks" --explain
    python scripts/inspect_query.py "risks" --compare-exact-hnsw
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.db.connection import get_db
from src.db.repositories.retrieval import (
    PostgresRetriever,
    POSTGRES_EXACT,
    POSTGRES_HNSW,
    POSTGRES_FILTERED,
    POSTGRES_FTS,
)
from src.retrieval.models import RetrievalResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Phase 4 PostgreSQL query results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--doc-type", type=str, default=None)
    parser.add_argument("--filter-current", action="store_true",
                        help="Only retrieve from is_current=True versions")
    parser.add_argument("--effective-date-lte", type=str, default=None,
                        help="YYYY-MM-DD: only versions effective by this date")
    parser.add_argument("--content-type", type=str, default=None,
                        help="Filter by content_type: text|table|list|mixed")
    parser.add_argument("--mode", choices=["HNSW", "EXACT", "FILTERED", "FTS"],
                        default="HNSW")
    parser.add_argument("--explain", action="store_true",
                        help="Show EXPLAIN ANALYZE for this query")
    parser.add_argument("--compare-exact-hnsw", action="store_true",
                        help="Run both exact and HNSW and compare results")
    parser.add_argument("--ef-search", type=int, default=None,
                        help="Override hnsw.ef_search for this query")
    return parser.parse_args()


def format_result(r: RetrievalResult, idx: int) -> None:
    m = r.metadata
    print(f"\n  [{idx}] similarity={r.semantic_score:.4f}  distance={1-r.semantic_score:.4f}")
    print(f"       Company  : {m.get('company', 'N/A')}")
    print(f"       Document : {m.get('document', 'N/A')}  [{m.get('document_type', 'N/A')}]")
    print(f"       Version  : {m.get('version_label', 'N/A')}  "
          f"effective={m.get('effective_date', 'N/A')}  "
          f"is_current={m.get('is_current', 'N/A')}")
    print(f"       Section  : {m.get('section', 'N/A')}")
    print(f"       Pages    : {m.get('page_start', '?')}–{m.get('page_end', '?')}")
    print(f"       Type     : {m.get('content_type', 'N/A')}  "
          f"tokens={m.get('token_count', '?')}")
    print(f"       Source   : {m.get('source_filename', 'N/A')}")
    print(f"       chunk_id : {r.chunk_id[:16]}...")
    print(f"       Text     : {r.text[:120].replace(chr(10), ' ')}...")


def main() -> None:
    args = parse_args()
    bar = "=" * 68

    # ── Build filters ─────────────────────────────────────────────────────────
    filters = {}
    if args.company:
        filters["company"] = args.company
    if args.doc_type:
        filters["document_type"] = args.doc_type
    if args.filter_current:
        filters["is_current"] = True
    if args.effective_date_lte:
        from datetime import datetime
        filters["effective_date_lte"] = datetime.strptime(args.effective_date_lte, "%Y-%m-%d").date()
    if args.content_type:
        filters["content_type"] = args.content_type

    mode_map = {
        "HNSW": POSTGRES_HNSW,
        "EXACT": POSTGRES_EXACT,
        "FILTERED": POSTGRES_FILTERED,
        "FTS": POSTGRES_FTS,
    }
    mode = mode_map[args.mode]

    print(f"\n{bar}")
    print("  DueDiligence-AI  |  Phase 4 Query Inspector")
    print(bar)
    print(f"  Query   : {args.query!r}")
    print(f"  Top-K   : {args.top_k}")
    print(f"  Mode    : {args.mode}")
    print(f"  Filters : {filters or '(none)'}")
    print(bar)

    with get_db() as db:
        retriever = PostgresRetriever(db)

        # ── Candidate count ────────────────────────────────────────────────────
        candidate_count = retriever.count_candidate_chunks(filters)
        print(f"\n  Candidate pool: {candidate_count} embedded chunks match filters")

        if candidate_count == 0:
            print("  [WARNING] No candidates — check filters or run ingest_pg.py first")

        # ── EXPLAIN ANALYZE ────────────────────────────────────────────────────
        if args.explain:
            print(f"\n{'─' * 68}")
            print("  EXPLAIN ANALYZE:")
            print('─' * 68)
            plan = retriever.explain_query(args.query, args.top_k, filters, mode)
            print(plan)
            print('─' * 68)

        # ── Compare exact vs HNSW ─────────────────────────────────────────────
        if args.compare_exact_hnsw:
            print(f"\n{'─' * 68}")
            print("  Experiment: Exact vs HNSW Comparison")
            print('─' * 68)

            t0 = time.perf_counter()
            exact_results = retriever.search(
                args.query, top_k=args.top_k, filters=filters, mode=POSTGRES_EXACT
            )
            exact_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            hnsw_results = retriever.search(
                args.query, top_k=args.top_k, filters=filters, mode=POSTGRES_HNSW,
                ef_search=args.ef_search,
            )
            hnsw_ms = (time.perf_counter() - t0) * 1000

            exact_ids  = {r.chunk_id for r in exact_results}
            hnsw_ids   = {r.chunk_id for r in hnsw_results}
            overlap    = len(exact_ids & hnsw_ids)
            recall_at_k = overlap / max(len(exact_ids), 1)

            print(f"\n  Exact search  : {len(exact_results)} results  latency={exact_ms:.1f}ms")
            print(f"  HNSW search   : {len(hnsw_results)} results  latency={hnsw_ms:.1f}ms")
            print(f"  Overlap       : {overlap}/{args.top_k}  Recall@{args.top_k}={recall_at_k:.2f}")

            missing = exact_ids - hnsw_ids
            if missing:
                print(f"\n  Chunks in EXACT but NOT in HNSW (missed by ANN):")
                for r in exact_results:
                    if r.chunk_id in missing:
                        print(f"    exact_score={r.semantic_score:.4f}  {r.text[:60]}...")
            else:
                print(f"  Perfect recall — HNSW found all exact results.")

        # ── Main search ────────────────────────────────────────────────────────
        print(f"\n{'─' * 68}")
        print(f"  Results (mode={args.mode}):")
        print('─' * 68)

        t0 = time.perf_counter()
        results = retriever.search(
            args.query,
            top_k=args.top_k,
            filters=filters,
            mode=mode,
            ef_search=args.ef_search,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        print(f"  Embedding ms : {retriever.last_embedding_ms:.1f}ms")
        print(f"  Search ms    : {retriever.last_search_ms:.1f}ms")
        print(f"  Total ms     : {total_ms:.1f}ms")
        print(f"  Results      : {len(results)}")

        if not results:
            print("\n  (no results — check filters or index)")
        else:
            for i, r in enumerate(results, start=1):
                format_result(r, i)

    print(f"\n{bar}\n")


if __name__ == "__main__":
    main()
