"""
src/retrieval/metrics.py  -  Retrieval evaluation metrics
==========================================================
Phase 3: Recall@K, Precision@K, MRR — explained from first principles.

WHY THESE METRICS?
-------------------
We want to know: "Does the retriever find the right chunks?"

This requires GROUND TRUTH: for each query, a list of chunk_ids that
contain the answer. We compare the retrieved list against ground truth.

METRIC DEFINITIONS
-------------------

RECALL@K
---------
"Of all relevant chunks, how many did we retrieve in the top K?"

  Recall@K = (# relevant chunks in top K) / (# total relevant chunks)

  Range: [0, 1]
  1.0 = all relevant chunks were retrieved in top K
  0.0 = no relevant chunk was retrieved in top K

  Example:
    relevant = ["A", "B", "C"]   (3 total relevant chunks)
    retrieved@5 = ["A", "D", "B", "E", "F"]
    Recall@5 = 2/3 = 0.667  (found A and B, missed C)

  IMPORTANT: Recall@K can only increase as K increases.
  Recall@10 >= Recall@5 >= Recall@1 for the same retriever.

PRECISION@K
-----------
"Of the K chunks we retrieved, how many were actually relevant?"

  Precision@K = (# relevant chunks in top K) / K

  Range: [0, 1]
  1.0 = every retrieved chunk was relevant
  0.0 = no retrieved chunk was relevant

  Example (same as above):
    Precision@5 = 2/5 = 0.400  (retrieved 5, only 2 were relevant)

  IMPORTANT: Precision@K can decrease as K increases.
  Retrieving more chunks dilutes precision.
  This is the Precision-Recall tradeoff: higher K -> better recall but lower precision.

MRR (MEAN RECIPROCAL RANK)
---------------------------
"How high does the FIRST relevant chunk appear in the ranked list?"

  For a single query:
    RR = 1 / (rank of first relevant chunk)
    If no relevant chunk found: RR = 0

  MRR over N queries = mean of all RR values

  Range: [0, 1]
  MRR=1.0 = first relevant chunk is always rank 1
  MRR=0.5 = first relevant chunk is on average at rank 2
  MRR=0.0 = no relevant chunk ever found

  Example:
    retrieved = ["D", "A", "E", "B", "F"]   (A is relevant, at rank 2)
    RR = 1/2 = 0.5

  WHY MRR?
  MRR rewards having at least ONE relevant chunk near the top.
  It's useful when:
    - You only show the user the top result
    - The user stops reading after finding the first relevant answer

  MRR LIMITATION:
  MRR only considers the FIRST relevant chunk. If there are 3 relevant
  chunks and the second/third are ranked poorly, MRR doesn't capture this.
  That's why we combine MRR with Recall@K.

PRECISION/RECALL TRADEOFF
--------------------------
Increasing K improves Recall but hurts Precision.

  K=1:   High Precision (if rank-1 is relevant), but low Recall
  K=10:  Lower Precision (10 chunks, some may be irrelevant), but higher Recall

  The right K depends on the application:
    - QA with a single answer: optimize MRR and Recall@1
    - Research assistant (need all relevant context): optimize Recall@10
    - LLM context window is limited: balance Recall@5 and Precision@5
"""

from __future__ import annotations

from typing import Dict, List, Optional


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids:  List[str],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Receives : retrieved_ids — ordered list of chunk_ids from retriever
               relevant_ids  — ground truth list of relevant chunk_ids
               k             — how many retrieved results to consider
    Returns  : float in [0, 1]

    Edge cases:
      relevant_ids empty: returns 1.0 (vacuously, all 0 relevant found in top K of 0)
      retrieved_ids empty: returns 0.0 (unless relevant_ids also empty)
    """
    if not relevant_ids:
        return 1.0   # vacuously true: nothing to find

    top_k_ids = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    found = len(top_k_ids & relevant_set)
    return found / len(relevant_set)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids:  List[str],
    k: int,
) -> float:
    """
    Compute Precision@K.

    Receives : retrieved_ids — ordered list of chunk_ids from retriever
               relevant_ids  — ground truth list of relevant chunk_ids
               k             — how many retrieved results to consider
    Returns  : float in [0, 1]

    Edge cases:
      k=0: returns 0.0
      retrieved_ids empty: returns 0.0
    """
    if k == 0 or not retrieved_ids:
        return 0.0

    top_k_ids = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    found = sum(1 for rid in top_k_ids if rid in relevant_set)
    return found / k


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids:  List[str],
) -> float:
    """
    Compute Reciprocal Rank for a single query.

    Returns 1/rank of the first relevant chunk, or 0.0 if none found.

    Example:
        retrieved = ["A", "B", "C", "D"]
        relevant  = ["C", "E"]
        RR = 1/3 = 0.333  (C is at rank 3)
    """
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mrr(
    all_retrieved: List[List[str]],
    all_relevant:  List[List[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank over a set of queries.

    Receives : all_retrieved — list of retrieved_id lists, one per query
               all_relevant  — list of relevant_id lists, one per query
    Returns  : float in [0, 1]
    """
    if not all_retrieved:
        return 0.0

    rr_scores = [
        reciprocal_rank(retr, relev)
        for retr, relev in zip(all_retrieved, all_relevant)
    ]
    return sum(rr_scores) / len(rr_scores)


def evaluate_mode(
    retrieved_list: List[List[str]],   # one list of chunk_ids per query
    relevant_list:  List[List[str]],   # one list of ground truth ids per query
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute all metrics for one retrieval mode over the full benchmark.

    Receives : retrieved_list — list of ordered chunk_id lists (one per query)
               relevant_list  — list of ground truth chunk_id lists (one per query)
               k_values       — list of K values to compute Recall and Precision at
    Returns  : dict with metric names -> values

    Metrics computed:
        Recall@K     for each K in k_values
        Precision@K  for each K in k_values
        MRR
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    n = len(retrieved_list)
    if n == 0:
        return {}

    results: Dict[str, float] = {}

    # MRR
    results["MRR"] = round(mrr(retrieved_list, relevant_list), 4)

    # Recall@K and Precision@K
    for k in k_values:
        recall_scores = [
            recall_at_k(retr, relev, k)
            for retr, relev in zip(retrieved_list, relevant_list)
        ]
        precision_scores = [
            precision_at_k(retr, relev, k)
            for retr, relev in zip(retrieved_list, relevant_list)
        ]
        results[f"Recall@{k}"]    = round(sum(recall_scores)    / n, 4)
        results[f"Precision@{k}"] = round(sum(precision_scores) / n, 4)

    return results


def format_metrics_table(
    mode_metrics: Dict[str, Dict[str, float]],
    k_values: List[int] = None,
) -> str:
    """
    Format a comparison table of metrics across retrieval modes.

    Receives : mode_metrics — {mode_name -> {metric_name -> value}}
    Returns  : ASCII table string

    Example output:
        Mode             | Recall@1 | Recall@5 | Precision@5 | MRR
        VECTOR_ONLY      |   0.45   |   0.78   |    0.32     | 0.61
        KEYWORD_ONLY     |   0.40   |   0.72   |    0.29     | 0.55
        HYBRID           |   0.52   |   0.85   |    0.35     | 0.68
        HYBRID_RERANKED  |   0.60   |   0.85   |    0.35     | 0.74
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics_order = (
        [f"Recall@{k}" for k in k_values]
        + [f"Precision@{k}" for k in k_values[:2]]  # show P@1 and P@5
        + ["MRR"]
    )
    # Only include metrics that exist in the data
    sample = next(iter(mode_metrics.values())) if mode_metrics else {}
    metrics_order = [m for m in metrics_order if m in sample]

    col_w   = 10
    mode_w  = max(18, max(len(m) for m in mode_metrics)) + 2

    header = f"{'Mode':<{mode_w}}" + "".join(f"| {m:^{col_w}}" for m in metrics_order)
    sep    = "-" * len(header)

    lines = [sep, header, sep]
    for mode_name, metrics in mode_metrics.items():
        row = f"{mode_name:<{mode_w}}"
        for m in metrics_order:
            val = metrics.get(m, float("nan"))
            row += f"| {val:^{col_w}.4f}"
        lines.append(row)
    lines.append(sep)

    return "\n".join(lines)


def per_query_breakdown(
    questions:      List[Dict],
    retrieved_list: List[List[str]],
    relevant_list:  List[List[str]],
    k: int = 5,
) -> str:
    """
    Return a per-query table showing which questions were answered and which weren't.
    Helps identify failure cases.
    """
    lines = [
        f"Per-query breakdown @ Recall@{k}",
        "-" * 70,
    ]
    for i, (q, retr, relev) in enumerate(zip(questions, retrieved_list, relevant_list)):
        r = recall_at_k(retr, relev, k)
        rr = reciprocal_rank(retr, relev)
        status = "OK " if r > 0 else "FAIL"
        text = q.get("question", "?")[:60]
        lines.append(
            f"  [{status}] Q{i+1:02d} Recall@{k}={r:.2f} MRR={rr:.2f}  {text}"
        )
    return "\n".join(lines)
