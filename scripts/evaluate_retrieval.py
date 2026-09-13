"""
scripts/evaluate_retrieval.py  -  Full retrieval benchmark evaluation
======================================================================
Runs all 4 retrieval modes against the 30-question benchmark and reports:

  - Recall@1, @3, @5, @10
  - Precision@1, @5
  - MRR
  - Average query latency per mode
  - Per-query breakdown (which questions failed)
  - Ablation: Hybrid + metadata filter
  - Error analysis: where each mode fails

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --top-k 10
    python scripts/evaluate_retrieval.py --modes VECTOR_ONLY HYBRID
    python scripts/evaluate_retrieval.py --no-reranker    # skip 85MB model download

Output:
    Prints comparison table + saves results to data/evaluation/results_<timestamp>.json
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

# ── Windows encoding fix ──────────────────────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid   import HybridRetriever
from src.retrieval.metrics  import (
    evaluate_mode,
    format_metrics_table,
    per_query_breakdown,
)
from src.retrieval.models   import (
    VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED,
)

QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
RESULTS_DIR    = PROJECT_ROOT / "data" / "evaluation"

K_VALUES = [1, 3, 5, 10]


def load_questions(path: Path) -> List[Dict]:
    """Load and filter the benchmark questions — exclude comment objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [q for q in raw if "question" in q]   # skip _comment entries


def run_mode(
    retriever: HybridRetriever,
    questions: List[Dict],
    top_k: int,
    mode: str,
) -> Dict:
    """
    Run retriever in one mode over all questions.
    Returns:
      retrieved_list    : list of chunk_id lists (one per question)
      relevant_list     : list of ground-truth chunk_id lists
      latencies_ms      : per-question total latency
      question_results  : detailed per-question info
    """
    retriever.set_mode(mode)
    retrieved_list: List[List[str]] = []
    relevant_list:  List[List[str]] = []
    latencies_ms:   List[float]     = []
    question_results = []

    print(f"\n  Running {mode}... ", end="", flush=True)

    for q in questions:
        result = retriever.search(q["question"])
        retrieved_ids = [r.chunk_id for r in result.results[:top_k]]
        relevant_ids  = q.get("relevant_chunk_ids", [])

        retrieved_list.append(retrieved_ids)
        relevant_list.append(relevant_ids)
        latencies_ms.append(result.latency.total_ms)

        question_results.append({
            "id":            q.get("id", "?"),
            "question":      q["question"],
            "category":      q.get("category", "unknown"),
            "relevant_ids":  relevant_ids,
            "retrieved_ids": retrieved_ids,
            "latency_ms":    result.latency.total_ms,
            "latency":       result.latency.to_dict(),
        })

    print(f"done ({len(questions)} queries)")
    return {
        "retrieved_list":   retrieved_list,
        "relevant_list":    relevant_list,
        "latencies_ms":     latencies_ms,
        "question_results": question_results,
    }


def error_analysis(mode_results: Dict[str, Dict], questions: List[Dict]) -> str:
    """
    Identify cases where each mode succeeds/fails and where reranking helps.
    Returns a formatted report string.
    """
    lines = [
        "\n" + "=" * 70,
        "  RETRIEVAL ERROR ANALYSIS",
        "=" * 70,
    ]

    vector_retrieved = mode_results.get(VECTOR_ONLY, {}).get("retrieved_list", [])
    keyword_retrieved = mode_results.get(KEYWORD_ONLY, {}).get("retrieved_list", [])
    hybrid_retrieved  = mode_results.get(HYBRID, {}).get("retrieved_list", [])
    reranked_retrieved = mode_results.get(HYBRID_RERANKED, {}).get("retrieved_list", [])

    n = len(questions)

    sem_only = []   # semantic finds it, keyword doesn't
    kw_only  = []   # keyword finds it, semantic doesn't
    rerank_helps = []
    rerank_hurts = []

    for i, q in enumerate(questions):
        relev = set(q.get("relevant_chunk_ids", []))
        if not relev:
            continue  # skip no-answer questions for this analysis

        v_ids  = set(vector_retrieved[i])  if i < len(vector_retrieved)  else set()
        k_ids  = set(keyword_retrieved[i]) if i < len(keyword_retrieved) else set()
        h_ids  = set(hybrid_retrieved[i])  if i < len(hybrid_retrieved)  else set()
        r_ids  = set(reranked_retrieved[i]) if i < len(reranked_retrieved) else set()

        v_hit = bool(relev & v_ids)
        k_hit = bool(relev & k_ids)
        h_hit = bool(relev & h_ids)
        r_hit = bool(relev & r_ids)

        if v_hit and not k_hit:
            sem_only.append(q)
        if k_hit and not v_hit:
            kw_only.append(q)
        if not h_hit and r_hit:
            rerank_helps.append(q)
        if h_hit and not r_hit:
            rerank_hurts.append(q)

    lines.append(f"\n[A] Semantic finds it, Keyword misses it ({len(sem_only)} queries):")
    for q in sem_only:
        lines.append(f"     - [{q.get('category','?')}] {q['question'][:70]}")
    if not sem_only:
        lines.append("     (none in this benchmark)")

    lines.append(f"\n[B] Keyword finds it, Semantic misses it ({len(kw_only)} queries):")
    for q in kw_only:
        lines.append(f"     - [{q.get('category','?')}] {q['question'][:70]}")
    if not kw_only:
        lines.append("     (none in this benchmark)")

    # Only show reranking analysis if HYBRID_RERANKED was actually evaluated
    if HYBRID_RERANKED in mode_results and mode_results[HYBRID_RERANKED].get("retrieved_list"):
        lines.append(f"\n[C] Reranking IMPROVES result ({len(rerank_helps)} queries):")
        for q in rerank_helps:
            lines.append(f"     - {q['question'][:70]}")
        if not rerank_helps:
            lines.append("     (none in this benchmark)")

        lines.append(f"\n[D] Reranking HURTS result ({len(rerank_hurts)} queries):")
        for q in rerank_hurts:
            lines.append(f"     - {q['question'][:70]}")
        if not rerank_hurts:
            lines.append("     (none in this benchmark)")
    else:
        lines.append("\n[C/D] Reranking analysis skipped (HYBRID_RERANKED not evaluated).")
        lines.append("      Re-run without --no-reranker to see reranker movement analysis.")

    return "\n".join(lines)



def reranker_movement_analysis(
    retriever: HybridRetriever,
    questions: List[Dict],
    n_questions: int = 10,
    top_k: int = 5,
) -> str:
    """
    For the first n_questions, show how reranking changed chunk ordering.
    """
    lines = [
        "\n" + "=" * 70,
        "  RERANKER MOVEMENT ANALYSIS (HYBRID vs HYBRID_RERANKED)",
        "=" * 70,
    ]

    for i, q in enumerate(questions[:n_questions]):
        lines.append(f"\n  Q{i+1}: {q['question'][:70]}")
        lines.append(f"  Relevant: {q.get('relevant_chunk_ids', [])}")

        retriever.set_mode(HYBRID)
        hybrid_result = retriever.search(q["question"])
        hybrid_ids = [r.chunk_id for r in hybrid_result.results[:top_k]]

        retriever.set_mode(HYBRID_RERANKED)
        rerank_result = retriever.search(q["question"])
        rerank_ids = [r.chunk_id for r in rerank_result.results[:top_k]]

        lines.append(f"  Before reranking (HYBRID top-{top_k}):")
        for rank, r in enumerate(hybrid_result.results[:top_k], 1):
            marker = " <-- RELEVANT" if r.chunk_id in q.get("relevant_chunk_ids", []) else ""
            lines.append(f"    {rank}. {r.chunk_id}  rrf={r.fusion_score:.5f}  {r.section[:30]}{marker}")

        lines.append(f"  After reranking (HYBRID_RERANKED top-{top_k}):")
        for rank, r in enumerate(rerank_result.results[:top_k], 1):
            old_rank = hybrid_ids.index(r.chunk_id) + 1 if r.chunk_id in hybrid_ids else None
            change = ""
            if old_rank:
                d = old_rank - rank
                change = f"  [UP {d}]" if d > 0 else (f"  [DOWN {abs(d)}]" if d < 0 else "  [same]")
            else:
                change = "  [NEW]"
            marker = " <-- RELEVANT" if r.chunk_id in q.get("relevant_chunk_ids", []) else ""
            lines.append(f"    {rank}. {r.chunk_id}  rerank={r.rerank_score:.4f}{change}{marker}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all retrieval modes on the benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top-k", type=int, default=5,
                        help="Final results per query (default: 5)")
    parser.add_argument("--sem-k", type=int, default=9,
                        help="Semantic candidate pool size")
    parser.add_argument("--kw-k",  type=int, default=9,
                        help="Keyword candidate pool size")
    parser.add_argument("--modes", nargs="+",
                        default=[VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED],
                        choices=[VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED],
                        help="Retrieval modes to evaluate")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Skip HYBRID_RERANKED (avoids 85MB model download)")
    parser.add_argument("--no-movement", action="store_true",
                        help="Skip reranker movement analysis")
    parser.add_argument("--save", action="store_true",
                        help="Save full results JSON to data/evaluation/")
    args = parser.parse_args()

    if args.no_reranker and HYBRID_RERANKED in args.modes:
        args.modes.remove(HYBRID_RERANKED)

    # ── Load questions ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RETRIEVAL EVALUATION  --  DueDiligence-AI Phase 3")
    print("=" * 70)

    questions = load_questions(QUESTIONS_PATH)
    # Filter out no-answer / trick questions for metrics (they have no relevant chunks)
    eval_questions = [q for q in questions if q.get("relevant_chunk_ids")]
    trick_questions = [q for q in questions if not q.get("relevant_chunk_ids")]

    print(f"\n  Benchmark   : {QUESTIONS_PATH.name}")
    print(f"  Questions   : {len(eval_questions)} with ground truth + {len(trick_questions)} no-answer")
    print(f"  Modes       : {', '.join(args.modes)}")
    print(f"  Top-K       : {args.top_k}")
    print(f"  Sem-K / KW-K: {args.sem_k} / {args.kw_k}")

    # ── Build retriever (loads indexes once) ──────────────────────────────────
    print("\n  [Loading indexes and models...]")
    retriever = HybridRetriever(
        mode           = args.modes[0],
        semantic_top_k = args.sem_k,
        keyword_top_k  = args.kw_k,
        final_top_k    = args.top_k,
        index_dir      = str(PROJECT_ROOT / "data" / "index"),
    )

    # ── Run evaluation for each mode ─────────────────────────────────────────
    all_mode_results: Dict[str, Dict] = {}
    all_metrics: Dict[str, Dict] = {}

    for mode in args.modes:
        data = run_mode(retriever, eval_questions, args.top_k, mode)
        all_mode_results[mode] = data

        metrics = evaluate_mode(
            data["retrieved_list"],
            data["relevant_list"],
            k_values=K_VALUES,
        )
        avg_latency = sum(data["latencies_ms"]) / len(data["latencies_ms"])
        metrics["avg_latency_ms"] = round(avg_latency, 1)
        all_metrics[mode] = metrics

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    table = format_metrics_table(all_metrics, k_values=K_VALUES)
    print(table)

    # ── Latency breakdown ─────────────────────────────────────────────────────
    print("\n  Avg Query Latency (ms):")
    for mode, m in all_metrics.items():
        print(f"    {mode:<20}  {m.get('avg_latency_ms', 0):.1f} ms")

    # ── Per-category breakdown ────────────────────────────────────────────────
    categories = sorted(set(q.get("category", "unknown") for q in eval_questions))
    for mode in args.modes:
        data = all_mode_results[mode]
        print(f"\n  {mode} — Per-category Recall@{args.top_k}:")
        for cat in categories:
            cat_qs = [(q, r, v) for q, r, v in zip(
                eval_questions,
                data["retrieved_list"],
                data["relevant_list"],
            ) if q.get("category") == cat]
            if not cat_qs:
                continue
            cat_recall = sum(
                1 for _, retr, relev in cat_qs
                if any(rid in retr[:args.top_k] for rid in relev)
            ) / len(cat_qs)
            print(f"    {cat:<20}  Recall@{args.top_k}={cat_recall:.2f}  ({len(cat_qs)} queries)")

    # ── Per-query breakdown (best-recall mode) ────────────────────────────────
    if all_mode_results:
        # Show breakdown for the mode with highest Recall@top_k
        best_mode = max(
            all_metrics,
            key=lambda m: all_metrics[m].get(f"Recall@{args.top_k}", 0.0),
        )
        data = all_mode_results[best_mode]
        breakdown = per_query_breakdown(
            eval_questions, data["retrieved_list"], data["relevant_list"], k=args.top_k
        )
        print(f"\n  [Per-query breakdown for best mode: {best_mode}]")
        print(f"{breakdown}")


    # ── Error analysis ────────────────────────────────────────────────────────
    analysis = error_analysis(all_mode_results, eval_questions)
    print(analysis)

    # ── Reranker movement analysis ─────────────────────────────────────────────
    if not args.no_movement and HYBRID_RERANKED in args.modes and HYBRID in args.modes:
        print("\n  [Running reranker movement analysis on 10 questions...]")
        movement = reranker_movement_analysis(
            retriever, eval_questions[:10], n_questions=10, top_k=args.top_k
        )
        print(movement)

    # ── Ablation: HYBRID + metadata filter ───────────────────────────────────
    if HYBRID in args.modes:
        print("\n" + "=" * 70)
        print("  ABLATION: HYBRID + metadata filter (content_type=table)")
        print("=" * 70)
        retriever.set_mode(HYBRID)
        table_filter_results = []
        for q in eval_questions:
            r = retriever.search(q["question"], filters={"content_type": "table"})
            table_filter_results.append([res.chunk_id for res in r.results])

        ablation_metrics = evaluate_mode(
            table_filter_results,
            [q.get("relevant_chunk_ids", []) for q in eval_questions],
            k_values=K_VALUES,
        )
        print(f"\n  HYBRID + filter(table)  Recall@{args.top_k}={ablation_metrics.get(f'Recall@{args.top_k}', 0):.4f}  MRR={ablation_metrics.get('MRR', 0):.4f}")
        print("  (Expected: lower recall — table filter removes many relevant chunks)")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"results_{timestamp}.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            "timestamp": timestamp,
            "config": {
                "top_k": args.top_k,
                "sem_k": args.sem_k,
                "kw_k":  args.kw_k,
                "modes": args.modes,
            },
            "metrics": all_metrics,
            "mode_results": {
                mode: {
                    "question_results": data["question_results"],
                    "metrics": all_metrics.get(mode, {}),
                }
                for mode, data in all_mode_results.items()
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved -> {out_path}")

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
