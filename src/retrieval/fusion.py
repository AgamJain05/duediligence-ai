"""
src/retrieval/fusion.py  -  Reciprocal Rank Fusion (RRF)
=========================================================
Phase 3: Combine semantic and keyword result lists into a single ranking.

WHY FUSION?
------------
Semantic retrieval and keyword retrieval produce SEPARATE ranked lists.
Simply concatenating them would give duplicates and no principled ordering.
We need a merging algorithm that:

  1. Handles chunks appearing in BOTH lists (they get credit from both)
  2. Handles chunks appearing in ONLY ONE list
  3. Produces a single coherent ranking

RECIPROCAL RANK FUSION (RRF)
-----------------------------
RRF was introduced by Cormack, Clarke, Buettcher (SIGIR 2009) as a simple
but effective way to combine ranked lists without needing to know the
absolute score distributions of each list.

THE FORMULA
-----------
For a document d appearing at rank r in a ranked list L:

    RRF_score_from_L(d) = 1 / (k + r)

Where:
    r = rank of d in list L (1-indexed: 1 = top, 2 = second, ...)
    k = smoothing constant (default 60, recommended in the paper)

For a document appearing in MULTIPLE lists:

    RRF_total(d) = sum over all lists L of 1 / (k + rank_of_d_in_L)

If d does NOT appear in list L, it contributes 0 from that list.
Documents are then ranked by RRF_total descending.

WHY k=60?
----------
k prevents top-ranked documents from dominating too strongly.

Without k: rank-1 score = 1/1 = 1.0, rank-2 score = 1/2 = 0.5
           rank-1 is 2x better than rank-2 — too strong a preference

With k=60: rank-1 score = 1/61 ≈ 0.0164, rank-2 score = 1/62 ≈ 0.0161
           rank-1 is only slightly better — allows blending

k=60 was found empirically to work well. It's configurable here.

WORKED EXAMPLE
--------------
Semantic list:  [A, B, C, D]         (k=60)
Keyword list:   [C, A, E, F]

RRF scores:
  A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252  (appears in both)
  B: 1/(60+2) + 0         = 0.01613                       (semantic only)
  C: 1/(60+3) + 1/(60+1)  = 0.01587 + 0.01639 = 0.03226  (appears in both)
  D: 1/(60+4) + 0         = 0.01563                       (semantic only)
  E: 0         + 1/(60+3)  = 0.01587                       (keyword only)
  F: 0         + 1/(60+4)  = 0.01563                       (keyword only)

Final ranking: A (0.03252) > C (0.03226) > B (0.01613) > E (0.01587) > ...

Note: C was rank 1 in keyword but rank 3 in semantic — RRF pulls it up.
Note: E was NOT in the semantic list at all — keyword retrieval discovers it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.retrieval.models import RetrievalResult


class ReciprocalRankFusion:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Supports 2 input lists (semantic + keyword) but is generalized
    to handle any number of ranked lists.

    Usage:
        rrf = ReciprocalRankFusion(k=60)
        fused = rrf.fuse(semantic_results, keyword_results)

    Returns a new list of RetrievalResult with:
      - fusion_score set to the RRF total
      - semantic_rank and keyword_rank preserved from input lists
      - rerank_score still None (not set until reranker runs)
    """

    def __init__(self, k: int = 60) -> None:
        """
        k : RRF smoothing constant (default 60)
            Lower k = more emphasis on top ranks
            Higher k = more uniform blending
        """
        self.k = k
        self.last_fusion_ms: float = 0.0

    def fuse(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results:  List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Receives : semantic_results — from SemanticRetriever.search()
                   keyword_results  — from KeywordRetriever.search()
        Returns  : merged list sorted by fusion_score descending

        Duplicates are merged: a chunk appearing in both lists gets
        ONE entry in the output with contributions from both.

        The output preserves semantic_rank and keyword_rank from the
        input lists so debugging can show "where did each chunk come from?"
        """
        t0 = time.perf_counter()

        # Step 1: Collect all unique chunk_ids across both lists
        # Map chunk_id -> best RetrievalResult (for metadata access)
        all_results: Dict[str, RetrievalResult] = {}
        rrf_scores:  Dict[str, float] = defaultdict(float)

        # Process semantic list
        for result in semantic_results:
            cid = result.chunk_id
            all_results[cid] = result
            rrf_scores[cid] += 1.0 / (self.k + result.semantic_rank)  # semantic_rank is set

        # Process keyword list
        for result in keyword_results:
            cid = result.chunk_id
            rrf_score_from_kw = 1.0 / (self.k + result.keyword_rank)  # keyword_rank is set
            rrf_scores[cid]  += rrf_score_from_kw

            if cid not in all_results:
                # Chunk only in keyword list — add it
                all_results[cid] = result
            else:
                # Chunk in BOTH lists: merge keyword scores into the existing entry
                existing = all_results[cid]
                # Use object.__setattr__ since we might hit frozen dataclass issues
                # Instead, create a new merged object
                all_results[cid] = RetrievalResult(
                    chunk_id      = existing.chunk_id,
                    text          = existing.text,
                    source        = existing.source,
                    page_start    = existing.page_start,
                    page_end      = existing.page_end,
                    section       = existing.section,
                    section_path  = existing.section_path,
                    content_type  = existing.content_type,
                    metadata      = existing.metadata,
                    semantic_score = existing.semantic_score,
                    semantic_rank  = existing.semantic_rank,
                    keyword_score  = result.keyword_score,
                    keyword_rank   = result.keyword_rank,
                    fusion_score   = None,   # set below
                    rerank_score   = None,
                )

        # Step 2: Attach RRF scores and sort
        fused: List[RetrievalResult] = []
        for cid, rrf_score in rrf_scores.items():
            base = all_results[cid]
            fused.append(RetrievalResult(
                chunk_id      = base.chunk_id,
                text          = base.text,
                source        = base.source,
                page_start    = base.page_start,
                page_end      = base.page_end,
                section       = base.section,
                section_path  = base.section_path,
                content_type  = base.content_type,
                metadata      = base.metadata,
                semantic_score = base.semantic_score,
                semantic_rank  = base.semantic_rank,
                keyword_score  = base.keyword_score,
                keyword_rank   = base.keyword_rank,
                fusion_score   = round(rrf_score, 8),
                rerank_score   = None,
            ))

        fused.sort(key=lambda r: r.fusion_score, reverse=True)

        self.last_fusion_ms = (time.perf_counter() - t0) * 1000
        return fused

    def explain(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results:  List[RetrievalResult],
        chunk_id: str,
    ) -> str:
        """
        Show how a specific chunk's RRF score was computed.
        Useful for understanding why a chunk ranked where it did.
        """
        sem_rank = next(
            (r.semantic_rank for r in semantic_results if r.chunk_id == chunk_id), None
        )
        kw_rank = next(
            (r.keyword_rank for r in keyword_results if r.chunk_id == chunk_id), None
        )

        sem_contrib = 1.0 / (self.k + sem_rank) if sem_rank else 0.0
        kw_contrib  = 1.0 / (self.k + kw_rank)  if kw_rank  else 0.0
        total       = sem_contrib + kw_contrib

        lines = [
            f"RRF explanation for chunk {chunk_id} (k={self.k}):",
        ]
        if sem_rank:
            lines.append(f"  Semantic rank {sem_rank:2d}: 1 / ({self.k} + {sem_rank}) = {sem_contrib:.6f}")
        else:
            lines.append("  Semantic rank: NOT IN LIST -> 0.000000")
        if kw_rank:
            lines.append(f"  Keyword rank  {kw_rank:2d}: 1 / ({self.k} + {kw_rank}) = {kw_contrib:.6f}")
        else:
            lines.append("  Keyword rank:  NOT IN LIST -> 0.000000")
        lines.append(f"  RRF total score            = {total:.6f}")

        return "\n".join(lines)
