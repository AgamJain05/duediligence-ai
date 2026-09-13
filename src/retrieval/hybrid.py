"""
src/retrieval/hybrid.py  -  HybridRetriever orchestrator
=========================================================
Phase 3: Coordinates all retrieval components into a single pipeline.

This is the main entry point for Phase 3 retrieval. It supports all four modes:

  VECTOR_ONLY:     semantic retrieval only, no fusion, no reranking
  KEYWORD_ONLY:    BM25 keyword retrieval only
  HYBRID:          RRF fusion of semantic + keyword, no reranking
  HYBRID_RERANKED: RRF fusion + cross-encoder reranking

WHY MODES?
-----------
Each mode is an ablation: we systematically remove components to measure
their individual contribution.

  Comparing VECTOR_ONLY vs KEYWORD_ONLY shows which type of retrieval
  works better for different query types.

  Comparing HYBRID vs HYBRID_RERANKED isolates the reranker's contribution.

  Comparing VECTOR_ONLY vs HYBRID shows if fusion is helping.

Without ablation, you can't claim that any component is actually useful.

PIPELINE FLOW
--------------
VECTOR_ONLY:
  embed_query() -> FAISS.search() -> top_k results

KEYWORD_ONLY:
  tokenize(query) -> BM25.get_scores() -> top_k results

HYBRID:
  [semantic top_k] + [keyword top_k] -> RRF fusion -> [metadata filter] -> top_k fused

HYBRID_RERANKED:
  HYBRID -> cross-encoder.rerank(all fused candidates) -> top_k reranked

METADATA FILTER POSITION
--------------------------
The filter is applied AFTER fusion (as decided in the design review).
This preserves full recall from both retrieval systems before enforcing
structural constraints.

LATENCY BREAKDOWN
------------------
Each call records timing for every component separately.
This is important for understanding the latency cost of each addition:

  VECTOR_ONLY:     embedding_ms + semantic_ms
  KEYWORD_ONLY:    keyword_ms
  HYBRID:          embedding_ms + semantic_ms + keyword_ms + fusion_ms + filter_ms
  HYBRID_RERANKED: everything above + rerank_ms
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.retrieval.filters  import MetadataFilter
from src.retrieval.fusion   import ReciprocalRankFusion
from src.retrieval.models   import (
    HybridSearchResult,
    LatencyRecord,
    RetrievalMode,
    RetrievalResult,
    VECTOR_ONLY,
    KEYWORD_ONLY,
    HYBRID,
    HYBRID_RERANKED,
    VALID_MODES,
)
from src.retrieval.reranker  import CrossEncoderReranker, NullReranker
from src.retrieval.semantic  import SemanticRetriever
from src.retrieval.keyword   import KeywordRetriever


class HybridRetriever:
    """
    Orchestrates the full Phase 3 retrieval pipeline.

    Accepts a mode string at construction time, which controls which
    components are activated for each search() call.

    Usage:
        retriever = HybridRetriever(mode=HYBRID_RERANKED)
        result = retriever.search("What are OrionVault's key risks?")

        # Access final results
        for r in result.results:
            print(r.chunk_id, r.rerank_score, r.text[:80])

        # Access debug trace
        print(result.debug["semantic_results"])
        print(result.debug["fused_results"])
        print(result.latency.to_dict())
    """

    def __init__(
        self,
        mode:             str  = HYBRID_RERANKED,
        semantic_top_k:   int  = 20,
        keyword_top_k:    int  = 20,
        final_top_k:      int  = 5,
        rrf_k:            int  = 60,
        filters:          Optional[Dict] = None,
        index_dir:        str  = "data/index",
        reranker_model:   str  = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_null_reranker: bool = False,
    ) -> None:
        """
        mode:              VECTOR_ONLY | KEYWORD_ONLY | HYBRID | HYBRID_RERANKED
        semantic_top_k:    candidates from semantic retrieval (before reranking)
        keyword_top_k:     candidates from keyword retrieval (before reranking)
        final_top_k:       returned results after all stages
        rrf_k:             RRF smoothing constant (default 60)
        filters:           optional metadata filter dict applied after fusion
        index_dir:         path to data/index directory
        reranker_model:    HuggingFace model id for cross-encoder
        use_null_reranker: if True, skip cross-encoder (useful for fast testing)
        """
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {VALID_MODES}")

        self.mode           = mode
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k  = keyword_top_k
        self.final_top_k    = final_top_k
        self.filters        = filters or {}

        # Build semantic retriever (always needed, even KEYWORD_ONLY loads chunks)
        self._semantic = SemanticRetriever(index_dir=index_dir)

        # Build keyword retriever using the same chunks
        self._keyword = KeywordRetriever(chunks=self._semantic.all_chunks())

        # Build fusion component
        self._fusion = ReciprocalRankFusion(k=rrf_k)

        # Build metadata filter
        self._filter = MetadataFilter()

        self._reranker_model = reranker_model
        if use_null_reranker:
            self._reranker = NullReranker()
        elif mode == HYBRID_RERANKED:
            self._reranker = CrossEncoderReranker(model_name=reranker_model)
        else:
            self._reranker = None

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
    ) -> HybridSearchResult:
        """
        Run the configured retrieval pipeline and return a HybridSearchResult.

        filters: per-query override of instance-level filters (merged/overrides)

        The returned HybridSearchResult contains:
          .results   — final top_k chunks to send to the LLM
          .debug     — intermediate lists for inspection
          .latency   — per-component timing
        """
        if not query.strip():
            return HybridSearchResult(
                query=query, mode=self.mode, results=[], debug={}, latency=LatencyRecord()
            )

        # Merge instance filters with per-query filters
        active_filters = {**self.filters, **(filters or {})}

        latency = LatencyRecord()
        debug: Dict[str, List[RetrievalResult]] = {
            "semantic_results": [],
            "keyword_results":  [],
            "fused_results":    [],
            "filtered_results": [],
            "reranked_results": [],
        }

        t_total = time.perf_counter()

        # ── VECTOR_ONLY ────────────────────────────────────────────────────────
        if self.mode == VECTOR_ONLY:
            sem_results = self._semantic.search(query, top_k=self.semantic_top_k)
            latency.embedding_ms = self._semantic.last_embedding_ms
            latency.semantic_ms  = self._semantic.last_search_ms

            debug["semantic_results"] = sem_results

            # Apply metadata filter after retrieval
            t_f = time.perf_counter()
            filtered = self._filter.apply(sem_results, active_filters)
            latency.filter_ms = (time.perf_counter() - t_f) * 1000

            debug["filtered_results"] = filtered
            final = filtered[:self.final_top_k]

        # ── KEYWORD_ONLY ───────────────────────────────────────────────────────
        elif self.mode == KEYWORD_ONLY:
            t_kw = time.perf_counter()
            kw_results = self._keyword.search(query, top_k=self.keyword_top_k)
            latency.keyword_ms = self._keyword.last_search_ms

            debug["keyword_results"] = kw_results

            # Apply metadata filter after retrieval
            t_f = time.perf_counter()
            filtered = self._filter.apply(kw_results, active_filters)
            latency.filter_ms = (time.perf_counter() - t_f) * 1000

            debug["filtered_results"] = filtered
            final = filtered[:self.final_top_k]

        # ── HYBRID or HYBRID_RERANKED ──────────────────────────────────────────
        elif self.mode in {HYBRID, HYBRID_RERANKED}:
            # Step 1: Semantic retrieval
            sem_results = self._semantic.search(query, top_k=self.semantic_top_k)
            latency.embedding_ms = self._semantic.last_embedding_ms
            latency.semantic_ms  = self._semantic.last_search_ms
            debug["semantic_results"] = sem_results

            # Step 2: Keyword retrieval
            kw_results = self._keyword.search(query, top_k=self.keyword_top_k)
            latency.keyword_ms = self._keyword.last_search_ms
            debug["keyword_results"] = kw_results

            # Step 3: RRF fusion
            t_f = time.perf_counter()
            fused = self._fusion.fuse(sem_results, kw_results)
            latency.fusion_ms = (time.perf_counter() - t_f) * 1000
            debug["fused_results"] = fused

            # Step 4: Metadata filter (after fusion — preserves recall)
            t_mf = time.perf_counter()
            filtered = self._filter.apply(fused, active_filters)
            latency.filter_ms = (time.perf_counter() - t_mf) * 1000
            debug["filtered_results"] = filtered

            if self.mode == HYBRID:
                final = filtered[:self.final_top_k]

            else:  # HYBRID_RERANKED
                # Step 5: Cross-encoder reranking on ALL filtered candidates
                # (not just top_k — we let the reranker see the full pool)
                t_rr = time.perf_counter()
                reranked = self._reranker.rerank(
                    query, filtered, top_k=self.final_top_k
                )
                latency.rerank_ms = (time.perf_counter() - t_rr) * 1000
                debug["reranked_results"] = reranked
                final = reranked

        else:
            raise RuntimeError(f"Unhandled mode: {self.mode}")

        latency.total_ms = (time.perf_counter() - t_total) * 1000

        return HybridSearchResult(
            query   = query,
            mode    = self.mode,
            results = final,
            debug   = debug,
            latency = latency,
        )

    def set_mode(self, mode: str) -> None:
        """Change retrieval mode without re-loading indexes."""
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {VALID_MODES}")
        old = self.mode
        self.mode = mode

        # Initialize reranker if switching to HYBRID_RERANKED and we don't have one
        if mode == HYBRID_RERANKED and not isinstance(self._reranker, CrossEncoderReranker):
            model_name = getattr(self, "_reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._reranker = CrossEncoderReranker(model_name=model_name)

        print(f"[HybridRetriever] Mode changed: {old} -> {mode}")

    @property
    def chunk_count(self) -> int:
        return self._semantic.chunk_count
