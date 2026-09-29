"""
scripts/compare_phases.py  ─  Multi-phase retrieval benchmark comparison
══════════════════════════════════════════════════════════════════════════════
Runs all retrieval modes (Phase 1, 3, 4) on the benchmark question set and
produces a side-by-side comparison.

EXPERIMENTS RUN
────────────────
Experiment 1: FAISS vector retrieval (Phase 1/3) vs PostgreSQL exact vector
  Purpose: Verify moving the vector store doesn't change semantic behavior.
  Expected: Very high overlap (same embeddings → same ranking).

Experiment 2: PostgreSQL exact vs HNSW
  Metric: Recall@K, latency
  Expected: High recall (small corpus), HNSW faster.

Experiment 3: Unfiltered vs metadata-filtered retrieval
  Metric: Recall@K, latency, wrong-version retrieval rate
  Demonstrates: Filter improves precision for version-sensitive queries.

Experiment 4: Historical query vs current-version query
  Verifies: Relational version logic works correctly.
  "What was the requirement in 2024?" → v1
  "What is the current requirement?" → v2

Usage:
    python scripts/compare_phases.py
    python scripts/compare_phases.py --top-k 10
    python scripts/compare_phases.py --experiment 1
    python scripts/compare_phases.py --modes PHASE1 PHASE3 PG_HNSW
    python scripts/compare_phases.py --no-reranker   # skip cross-encoder download
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.retrieval.models import VECTOR_ONLY, HYBRID_RERANKED
from src.db.connection import get_db, ping_db
from src.db.repositories.retrieval import (
    PostgresRetriever,
    POSTGRES_EXACT,
    POSTGRES_HNSW,
)

QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
RESULTS_DIR    = PROJECT_ROOT / "data" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase 1/3/4 retrieval modes.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", nargs="+",
                        choices=["PHASE1", "PHASE3", "PG_EXACT", "PG_HNSW", "PG_FILTERED"],
                        default=["PHASE1", "PHASE3", "PG_HNSW", "PG_FILTERED"])
    parser.add_argument("--experiment", type=int, default=0,
                        help="Run specific experiment (1-4). 0=all.")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Skip cross-encoder reranker (faster)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to JSON")
    return parser.parse_args()


def load_questions(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [q for q in raw if "question" in q]


def compute_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & set(relevant)) / len(relevant)


def compute_mrr(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for i, rid in enumerate(retrieved, start=1):
        if rid in relevant_set:
            return 1.0 / i
    return 0.0


def run_faiss_mode(questions: List[Dict], top_k: int, mode_str: str, use_null_reranker: bool) -> Dict:
    """Run Phase 1 or Phase 3 FAISS-backed retrieval."""
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.models import VECTOR_ONLY, HYBRID_RERANKED

    faiss_mode = VECTOR_ONLY if mode_str == "PHASE1" else HYBRID_RERANKED
    retriever  = HybridRetriever(
        mode=faiss_mode,
        final_top_k=top_k,
        use_null_reranker=use_null_reranker,
    )

    results = []
    latencies = []
    for q in questions:
        t0 = time.perf_counter()
        sr = retriever.search(q["question"])
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        retrieved_ids = [r.chunk_id for r in sr.results]
        relevant_ids  = q.get("relevant_chunk_ids", [])
        results.append({
            "question_id": q["id"],
            "query": q["question"],
            "retrieved_ids": retrieved_ids,
            "relevant_ids":  relevant_ids,
            "recall@1":  compute_recall_at_k(retrieved_ids, relevant_ids, 1),
            "recall@3":  compute_recall_at_k(retrieved_ids, relevant_ids, 3),
            "recall@5":  compute_recall_at_k(retrieved_ids, relevant_ids, 5),
            "mrr":       compute_mrr(retrieved_ids, relevant_ids),
            "latency_ms": lat,
        })

    return _aggregate(results, latencies, mode_str)


def run_postgres_mode(
    questions: List[Dict],
    top_k: int,
    mode_str: str,
    pg_mode: str,
    filters: Optional[Dict] = None,
) -> Dict:
    """Run a Phase 4 PostgreSQL retrieval mode."""
    results = []
    latencies = []

    with get_db() as db:
        retriever = PostgresRetriever(db)
        for q in questions:
            t0 = time.perf_counter()
            res = retriever.search(q["question"], top_k=top_k, filters=filters, mode=pg_mode)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            retrieved_ids = [r.chunk_id for r in res]
            relevant_ids  = q.get("relevant_chunk_ids", [])

            results.append({
                "question_id": q["id"],
                "query": q["question"],
                "retrieved_ids": retrieved_ids,
                "relevant_ids":  relevant_ids,
                "recall@1":  compute_recall_at_k(retrieved_ids, relevant_ids, 1),
                "recall@3":  compute_recall_at_k(retrieved_ids, relevant_ids, 3),
                "recall@5":  compute_recall_at_k(retrieved_ids, relevant_ids, 5),
                "mrr":       compute_mrr(retrieved_ids, relevant_ids),
                "latency_ms": lat,
                # Phase 4 provenance
                "top_result_version": res[0].metadata.get("version_label") if res else None,
                "top_result_company": res[0].metadata.get("company") if res else None,
            })

    return _aggregate(results, latencies, mode_str)


def _aggregate(results: List[Dict], latencies: List[float], mode_name: str) -> Dict:
    n = len(results)
    if n == 0:
        return {"mode": mode_name, "n": 0}
    return {
        "mode":       mode_name,
        "n":          n,
        "recall@1":   sum(r["recall@1"] for r in results) / n,
        "recall@3":   sum(r["recall@3"] for r in results) / n,
        "recall@5":   sum(r["recall@5"] for r in results) / n,
        "mrr":        sum(r["mrr"]       for r in results) / n,
        "avg_latency_ms": sum(latencies) / n,
        "per_question":   results,
    }


def print_comparison_table(mode_results: List[Dict]) -> None:
    bar = "=" * 90
    print(f"\n{bar}")
    print("  RETRIEVAL MODE COMPARISON")
    print(bar)
    header = f"  {'Mode':<22} {'Recall@1':>8} {'Recall@3':>8} {'Recall@5':>8} {'MRR':>8} {'Latency':>10}"
    print(header)
    print("-" * 90)
    for r in mode_results:
        if r.get("n", 0) == 0:
            print(f"  {r['mode']:<22} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue
        print(
            f"  {r['mode']:<22} "
            f"{r['recall@1']:>8.3f} "
            f"{r['recall@3']:>8.3f} "
            f"{r['recall@5']:>8.3f} "
            f"{r['mrr']:>8.3f} "
            f"{r['avg_latency_ms']:>9.1f}ms"
        )
    print(bar)


def experiment_1_faiss_vs_pg_exact(questions, top_k, use_null_reranker):
    """Experiment 1: FAISS vector vs PostgreSQL exact vector."""
    print("\n" + "=" * 68)
    print("  Experiment 1: FAISS Vector vs PostgreSQL Exact")
    print("  Purpose: Verify semantic consistency across backends")
    print("=" * 68)

    try:
        faiss_r = run_faiss_mode(questions, top_k, "PHASE3_VECTOR", use_null_reranker)
    except Exception as e:
        print(f"  [SKIP] FAISS not available: {e}")
        faiss_r = None

    try:
        pg_r = run_postgres_mode(questions, top_k, "PG_EXACT", POSTGRES_EXACT)
    except Exception as e:
        print(f"  [SKIP] PostgreSQL not available: {e}")
        pg_r = None

    if faiss_r and pg_r:
        print_comparison_table([faiss_r, pg_r])

        # Compare per-question overlap
        faiss_q = {r["question_id"]: set(r["retrieved_ids"]) for r in faiss_r["per_question"]}
        pg_q    = {r["question_id"]: set(r["retrieved_ids"]) for r in pg_r["per_question"]}

        overlaps = []
        for qid in faiss_q:
            if qid in pg_q:
                overlap = len(faiss_q[qid] & pg_q[qid]) / max(len(faiss_q[qid]), 1)
                overlaps.append(overlap)
        avg_overlap = sum(overlaps) / max(len(overlaps), 1)
        print(f"\n  Average result overlap: {avg_overlap:.2%}")
        print(f"  Interpretation: {'High' if avg_overlap > 0.7 else 'Low'} — "
              f"{'backends agree' if avg_overlap > 0.7 else 'investigate differences'}")


def experiment_2_exact_vs_hnsw(questions, top_k):
    """Experiment 2: PostgreSQL exact vs HNSW — Recall@K and latency."""
    print("\n" + "=" * 68)
    print("  Experiment 2: PostgreSQL Exact vs HNSW")
    print("  Metric: Recall@K (HNSW results vs exact as ground truth)")
    print("=" * 68)

    try:
        exact_r = run_postgres_mode(questions, top_k, "PG_EXACT", POSTGRES_EXACT)
        hnsw_r  = run_postgres_mode(questions, top_k, "PG_HNSW",  POSTGRES_HNSW)
    except Exception as e:
        print(f"  [SKIP] PostgreSQL not available: {e}")
        return

    print_comparison_table([exact_r, hnsw_r])

    # Compute HNSW recall relative to exact search as ground truth
    exact_by_q = {r["question_id"]: set(r["retrieved_ids"]) for r in exact_r["per_question"]}
    hnsw_by_q  = {r["question_id"]: set(r["retrieved_ids"]) for r in hnsw_r["per_question"]}

    recalls = []
    for qid in exact_by_q:
        if qid in hnsw_by_q:
            exact_set = exact_by_q[qid]
            hnsw_set  = hnsw_by_q[qid]
            if exact_set:
                recalls.append(len(exact_set & hnsw_set) / len(exact_set))

    avg_recall = sum(recalls) / max(len(recalls), 1)
    latency_speedup = exact_r["avg_latency_ms"] / max(hnsw_r["avg_latency_ms"], 0.001)
    print(f"\n  HNSW Recall@{top_k} vs Exact: {avg_recall:.2%}")
    print(f"  Latency speedup (HNSW/Exact): {latency_speedup:.1f}x")
    print(f"  Note: With {sum(1 for _ in exact_by_q)} chunks, difference is expected to be minimal.")


def experiment_3_filtered_vs_unfiltered(questions, top_k):
    """Experiment 3: Unfiltered vs metadata-filtered retrieval."""
    print("\n" + "=" * 68)
    print("  Experiment 3: Unfiltered vs Metadata-Filtered Retrieval")
    print("  Demonstrates: Filter precision for version-sensitive queries")
    print("=" * 68)

    try:
        unfiltered = run_postgres_mode(questions, top_k, "PG_HNSW_Unfiltered",
                                        POSTGRES_HNSW, filters=None)
        filtered   = run_postgres_mode(questions, top_k, "PG_HNSW_CurrentOnly",
                                        POSTGRES_HNSW, filters={"is_current": True})
    except Exception as e:
        print(f"  [SKIP] PostgreSQL not available: {e}")
        return

    print_comparison_table([unfiltered, filtered])
    print("\n  Interpretation:")
    print("  If Recall@K drops with filter=is_current, the relevant chunks are in older versions.")
    print("  If precision improves, filter is correctly gating to the right document version.")


def experiment_4_version_queries(db_available: bool):
    """Experiment 4: Historical vs current version query."""
    print("\n" + "=" * 68)
    print("  Experiment 4: Historical vs Current Version Query")
    print("  Verifies relational version management logic")
    print("=" * 68)

    if not db_available:
        print("  [SKIP] Database not available.")
        return

    temporal_queries = [
        {
            "query": "What is the current data residency requirement for Germany?",
            "historical_filter": None,
            "current_filter": {"is_current": True},
            "note": "Should prefer v2 (current)"
        },
        {
            "query": "What data residency requirement applied in 2024?",
            "historical_filter": {"effective_date_lte": "2024-12-31"},
            "current_filter": {"is_current": True},
            "note": "Historical filter should retrieve v1"
        },
        {
            "query": "What risks were identified in the 2024 annual review?",
            "historical_filter": {"effective_date_lte": "2024-12-31"},
            "current_filter": {"is_current": True},
            "note": "Version-specific risk retrieval"
        },
    ]

    with get_db() as db:
        retriever = PostgresRetriever(db)
        for tq in temporal_queries:
            print(f"\n  Query: '{tq['query']}'")
            print(f"  Note : {tq['note']}")

            if tq["historical_filter"]:
                from datetime import datetime
                filters = {}
                if "effective_date_lte" in tq["historical_filter"]:
                    filters["effective_date_lte"] = datetime.strptime(
                        tq["historical_filter"]["effective_date_lte"], "%Y-%m-%d"
                    ).date()
                hist_results = retriever.search(tq["query"], top_k=3, filters=filters, mode=POSTGRES_HNSW)
                print(f"  Historical (filter by date):")
                for r in hist_results[:2]:
                    print(f"    v={r.metadata.get('version_label')}  "
                          f"eff={r.metadata.get('effective_date')}  "
                          f"sim={r.semantic_score:.3f}  "
                          f"{r.text[:60]}...")

            curr_results = retriever.search(tq["query"], top_k=3, filters=tq["current_filter"], mode=POSTGRES_HNSW)
            print(f"  Current (is_current=True):")
            for r in curr_results[:2]:
                print(f"    v={r.metadata.get('version_label')}  "
                      f"eff={r.metadata.get('effective_date')}  "
                      f"sim={r.semantic_score:.3f}  "
                      f"{r.text[:60]}...")


def main() -> None:
    args = parse_args()
    bar = "=" * 68

    print(f"\n{bar}")
    print("  DueDiligence-AI  |  Phase Comparison Benchmark")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(bar)

    questions = load_questions(QUESTIONS_PATH)
    q_with_gt = [q for q in questions if q.get("relevant_chunk_ids")]
    print(f"\n  Benchmark: {len(questions)} questions ({len(q_with_gt)} with ground truth)")

    db_available = ping_db()
    print(f"  Database: {'available' if db_available else 'unavailable'}")

    mode_results = []

    # ── Run configured modes ──────────────────────────────────────────────────
    for mode_str in args.modes:
        print(f"\n  Running mode: {mode_str}...")
        try:
            if mode_str == "PHASE1":
                r = run_faiss_mode(questions, args.top_k, "Phase 1 FAISS Vector",
                                    use_null_reranker=True)
                mode_results.append(r)
            elif mode_str == "PHASE3":
                r = run_faiss_mode(questions, args.top_k, "Phase 3 FAISS Hybrid+Rerank",
                                    use_null_reranker=args.no_reranker)
                mode_results.append(r)
            elif mode_str == "PG_EXACT" and db_available:
                r = run_postgres_mode(questions, args.top_k, "Phase 4 PG Exact",
                                       POSTGRES_EXACT)
                mode_results.append(r)
            elif mode_str == "PG_HNSW" and db_available:
                r = run_postgres_mode(questions, args.top_k, "Phase 4 PG HNSW",
                                       POSTGRES_HNSW)
                mode_results.append(r)
            elif mode_str == "PG_FILTERED" and db_available:
                r = run_postgres_mode(questions, args.top_k, "Phase 4 PG HNSW+Filter",
                                       POSTGRES_HNSW, filters={"is_current": True})
                mode_results.append(r)
            elif not db_available and mode_str.startswith("PG_"):
                print(f"  [SKIP] {mode_str}: database unavailable")
        except Exception as e:
            print(f"  [ERROR] {mode_str}: {e}")
            import traceback
            if args.no_reranker:
                traceback.print_exc()

    # ── Print comparison table ────────────────────────────────────────────────
    if mode_results:
        print_comparison_table(mode_results)

    # ── Run experiments ───────────────────────────────────────────────────────
    exp = args.experiment
    if exp == 0 or exp == 1:
        experiment_1_faiss_vs_pg_exact(questions, args.top_k, args.no_reranker)
    if exp == 0 or exp == 2:
        experiment_2_exact_vs_hnsw(questions, args.top_k)
    if exp == 0 or exp == 3:
        experiment_3_filtered_vs_unfiltered(questions, args.top_k)
    if exp == 0 or exp == 4:
        experiment_4_version_queries(db_available)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save and mode_results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"phase_comparison_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mode_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {out_path}")

    print(f"\n{bar}\n")


if __name__ == "__main__":
    main()
