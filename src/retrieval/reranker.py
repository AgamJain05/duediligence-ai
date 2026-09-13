"""
src/retrieval/reranker.py  -  Cross-encoder reranking
======================================================
Phase 3: Score (query, document) pairs with a cross-encoder model.

BI-ENCODER vs CROSS-ENCODER
-----------------------------
Phase 3 uses TWO types of models:

  BI-ENCODER (used in semantic.py):
    Encodes query and document SEPARATELY into vectors.
    Comparison is done via dot product at query time.
    Fast: query embedding is computed once, then compared to all stored vectors.
    Limitation: the model never "sees" the (query, document) pair together.

  CROSS-ENCODER (used here):
    Takes (query, document) as a PAIR and passes them through the model together.
    Produces a single relevance score for that pair.
    The model can directly compare query and document wording, not just proximity.
    Slower: must run one forward pass PER (query, document) pair.

WHY THIS ORDER?
----------------
We FIRST retrieve a large candidate pool (bi-encoder is fast for this),
THEN rerank a smaller set with the cross-encoder (slow but accurate).

  retrieve 20 candidates (fast)
  rerank top 20 with cross-encoder (slower, but only 20 pairs)
  return top 5 (high quality)

NOT:
  rerank all 10000 chunks (would be impossibly slow)

This is the standard "retrieve then rerank" pipeline in production RAG systems.

WHY RERANKING IMPROVES RESULTS
--------------------------------
The bi-encoder embeds query and document in the same space, but it encodes
them independently. A question and its answer can have different surface forms:

  Query: "What percentage of FY2025 revenue came from the US?"
  Answer chunk: "United States 34%"

The bi-encoder sees similar vectors because "FY2025 revenue" and "34%" co-occur
in financial contexts. But it doesn't see that "United States 34%" is the literal
answer to "what percentage from the US".

The cross-encoder DOES see both together and gives a higher score to the
chunk that directly answers the question, vs. a chunk that's about US revenue
in a different context.

WHY RERANKING CAN ALSO HURT
-----------------------------
Cross-encoders are trained on specific data distributions (MS MARCO QA pairs).
For domain-specific questions (due-diligence finance), the model may prefer
generic-sounding passages over precise domain ones.

Example: The cross-encoder may rank a passage saying "revenue grew strongly"
higher than a table showing "Revenue | 121 | 154 | 186" because the first
pattern more closely matches training data. This is a known failure mode.

We expose this honestly in the evaluation.

MODEL USED
----------
cross-encoder/ms-marco-MiniLM-L-6-v2
  - 85 MB, runs on CPU
  - Trained on MS MARCO (passage retrieval)
  - Industry standard for learning cross-encoder reranking
  - Well-understood failure modes

SCORING
--------
CrossEncoder.predict() returns raw logits (not probabilities).
Higher is more relevant. Typical range: -10 to +10.
The scores are NOT comparable across different models.
"""

from __future__ import annotations

import time
from typing import List, Optional

from src.retrieval.models import RetrievalResult

# ── Default reranker model ────────────────────────────────────────────────────
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level model cache (same pattern as embeddings.py)
_reranker_model = None
_loaded_model_name: Optional[str] = None


def _get_reranker(model_name: str):
    """Load and cache the cross-encoder model. Thread-safe for single-process use."""
    global _reranker_model, _loaded_model_name

    if _reranker_model is None or _loaded_model_name != model_name:
        from sentence_transformers import CrossEncoder  # lazy import
        print(f"[Reranker] Loading cross-encoder model '{model_name}'...")
        _reranker_model   = CrossEncoder(model_name)
        _loaded_model_name = model_name
        print("[Reranker] Model ready.")

    return _reranker_model


class CrossEncoderReranker:
    """
    Cross-encoder reranker for Phase 3.

    Takes a query and a list of candidates, scores each (query, chunk) pair,
    and returns the candidates re-ordered by reranker score.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, candidates, top_k=5)

    Each result has rerank_score set (raw logit, higher = more relevant).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.model_name = model_name
        self.last_rerank_ms: float = 0.0
        # Lazy load: model is loaded on first rerank() call, not at construction
        # This allows creating the reranker before the model is needed
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = _get_reranker(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Receives : query       — the user query string
                   candidates  — list[RetrievalResult] (from fusion or retrieval)
                   top_k       — how many to return (None = return all, re-ordered)
        Returns  : list[RetrievalResult] sorted by rerank_score descending
                   with rerank_score set on each result

        Each candidate pair (query, chunk.text) is scored in a single batch call.
        The original semantic_rank, keyword_rank, fusion_score are preserved.
        Only rerank_score and the list ordering change.

        WHY BATCH?
        ----------
        CrossEncoder.predict() accepts a list of (query, text) pairs and
        processes them in one forward pass (batched). This is faster than
        calling predict() once per pair.
        """
        if not candidates:
            return []

        self._ensure_loaded()

        t0 = time.perf_counter()

        # Build (query, text) pairs — one per candidate
        pairs = [(query, c.text) for c in candidates]

        # Batch score all pairs
        # scores: numpy array of shape (len(candidates),)
        scores = self._model.predict(pairs)

        self.last_rerank_ms = (time.perf_counter() - t0) * 1000

        # Attach rerank scores and create new results
        scored_results = []
        for result, score in zip(candidates, scores):
            scored_results.append(RetrievalResult(
                chunk_id       = result.chunk_id,
                text           = result.text,
                source         = result.source,
                page_start     = result.page_start,
                page_end       = result.page_end,
                section        = result.section,
                section_path   = result.section_path,
                content_type   = result.content_type,
                metadata       = result.metadata,
                semantic_score = result.semantic_score,
                semantic_rank  = result.semantic_rank,
                keyword_score  = result.keyword_score,
                keyword_rank   = result.keyword_rank,
                fusion_score   = result.fusion_score,
                rerank_score   = float(score),
            ))

        # Sort by rerank_score descending
        scored_results.sort(key=lambda r: r.rerank_score, reverse=True)

        if top_k is not None:
            scored_results = scored_results[:top_k]

        return scored_results

    def explain_movement(
        self,
        before: List[RetrievalResult],
        after:  List[RetrievalResult],
    ) -> str:
        """
        Print a before/after table showing how reranking changed the order.
        Shows which chunks moved up, down, or stayed.
        """
        before_ids = [r.chunk_id for r in before]
        after_ids  = [r.chunk_id for r in after]

        lines = ["Reranker movement (before -> after):"]
        for new_rank, result in enumerate(after, start=1):
            cid = result.chunk_id
            old_rank = before_ids.index(cid) + 1 if cid in before_ids else None
            movement = ""
            if old_rank is not None:
                delta = old_rank - new_rank
                if delta > 0:
                    movement = f"  [UP {delta}]"
                elif delta < 0:
                    movement = f"  [DOWN {abs(delta)}]"
                else:
                    movement = "  [same]"
            else:
                movement = "  [NEW - not in before list]"

            lines.append(
                f"  Rank {new_rank:2d}  chunk_id={cid}  "
                f"rerank={result.rerank_score:.4f}  "
                f"prev_rank={old_rank}{movement}"
            )

        return "\n".join(lines)


class NullReranker:
    """
    Fallback reranker that returns candidates unchanged.

    Used when:
      - The reranker model cannot be loaded
      - Fast inference is needed for testing
      - RETRIEVAL_MODE=HYBRID (no reranking)

    This is NOT a fake reranker — it explicitly sets rerank_score=None
    so downstream code knows no reranking happened.

    The class matches the CrossEncoderReranker interface so it can be
    swapped in without changing the pipeline.
    """

    def __init__(self) -> None:
        self.last_rerank_ms: float = 0.0
        print("[NullReranker] WARNING: using NullReranker — no reranking applied.")

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Return candidates as-is, preserving their fusion/semantic order."""
        self.last_rerank_ms = 0.0
        result = candidates[:top_k] if top_k is not None else candidates
        return result

    def explain_movement(
        self,
        before: List[RetrievalResult],
        after:  List[RetrievalResult],
    ) -> str:
        return "NullReranker: no reranking applied, order unchanged."
