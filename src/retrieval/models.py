"""
src/retrieval/models.py  -  Normalized retrieval result model
=============================================================
Phase 3: Common data structures shared across all retrieval components.

WHY A UNIFIED RESULT SCHEMA?
------------------------------
SemanticRetriever, KeywordRetriever, and HybridRetriever all return
information about the same underlying chunks, but each attaches different
scores (similarity, BM25, RRF, rerank). A single schema means:

  1. All retrievers are interchangeable in the pipeline
  2. Downstream code (metrics, inspect scripts) reads one field set
  3. Debugging is easier: one object shows ALL scores in one place

FIELD POPULATION BY MODE
--------------------------
Not every field is populated by every mode:

  VECTOR_ONLY:      semantic_score + semantic_rank filled; rest None
  KEYWORD_ONLY:     keyword_score + keyword_rank filled; rest None
  HYBRID:           semantic_* + keyword_* + fusion_score filled; rerank_score None
  HYBRID_RERANKED:  all fields filled

None means "this score was not computed", NOT "this chunk scored 0".
This is important for honest reporting.

RETRIEVAL MODES
---------------
The four modes expose different pipeline subsets:

  VECTOR_ONLY      - FAISS cosine similarity only
  KEYWORD_ONLY     - BM25 lexical matching only
  HYBRID           - RRF fusion of semantic + keyword
  HYBRID_RERANKED  - RRF fusion + cross-encoder reranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Retrieval mode constants ──────────────────────────────────────────────────
VECTOR_ONLY      = "VECTOR_ONLY"
KEYWORD_ONLY     = "KEYWORD_ONLY"
HYBRID           = "HYBRID"
HYBRID_RERANKED  = "HYBRID_RERANKED"

VALID_MODES = {VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED}


class RetrievalMode:
    """Namespace for retrieval mode string constants."""
    VECTOR_ONLY     = VECTOR_ONLY
    KEYWORD_ONLY    = KEYWORD_ONLY
    HYBRID          = HYBRID
    HYBRID_RERANKED = HYBRID_RERANKED


@dataclass
class RetrievalResult:
    """
    A single chunk with all retrieval scores attached.

    DESIGN: every field is Optional so the same class works for
    all four retrieval modes without forcing callers to populate
    fields they don't have.

    Sections:
      Identity   - what chunk this is
      Location   - where it came from in the document
      Content    - the actual text and type
      Scores     - retrieval scores (only the relevant ones are set)
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    chunk_id:     str
    text:         str

    # ── Source location ───────────────────────────────────────────────────────
    source:       str           # filename
    page_start:   int
    page_end:     int
    section:      str           # immediate section heading
    section_path: List[str]     # full hierarchy path

    # ── Content classification ────────────────────────────────────────────────
    content_type: str           # "text", "table", "list", "mixed"

    # ── Full metadata (original chunk dict) ───────────────────────────────────
    # Kept so downstream code can access any field without re-loading chunks
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Semantic retrieval scores ──────────────────────────────────────────────
    semantic_score: Optional[float] = None  # cosine similarity [-1, 1]
    semantic_rank:  Optional[int]   = None  # 1 = best, ascending

    # ── Keyword retrieval scores ───────────────────────────────────────────────
    keyword_score: Optional[float] = None   # BM25 score (unnormalized, > 0)
    keyword_rank:  Optional[int]   = None   # 1 = best, ascending

    # ── Fusion score ─────────────────────────────────────────────────────────
    # Sum of 1/(k + rank) for each list the chunk appears in
    fusion_score: Optional[float] = None

    # ── Reranking score ────────────────────────────────────────────────────────
    # Raw cross-encoder logit (higher = more relevant). Not normalized.
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict for JSON output."""
        return {
            "chunk_id":      self.chunk_id,
            "text":          self.text,
            "source":        self.source,
            "page_start":    self.page_start,
            "page_end":      self.page_end,
            "section":       self.section,
            "section_path":  self.section_path,
            "content_type":  self.content_type,
            "semantic_score": self.semantic_score,
            "semantic_rank":  self.semantic_rank,
            "keyword_score":  self.keyword_score,
            "keyword_rank":   self.keyword_rank,
            "fusion_score":   self.fusion_score,
            "rerank_score":   self.rerank_score,
        }

    def best_score(self) -> float:
        """
        Return the most informative score for sorting/display.
        Priority: rerank > fusion > semantic > keyword > 0.0
        """
        if self.rerank_score is not None:
            return self.rerank_score
        if self.fusion_score is not None:
            return self.fusion_score
        if self.semantic_score is not None:
            return self.semantic_score
        if self.keyword_score is not None:
            return self.keyword_score
        return 0.0


@dataclass
class LatencyRecord:
    """Timing breakdown for a single retrieval call (milliseconds)."""
    embedding_ms:  float = 0.0   # query embedding
    semantic_ms:   float = 0.0   # FAISS search
    keyword_ms:    float = 0.0   # BM25 search
    fusion_ms:     float = 0.0   # RRF computation
    filter_ms:     float = 0.0   # metadata filtering
    rerank_ms:     float = 0.0   # cross-encoder scoring
    total_ms:      float = 0.0   # wall-clock from query to final result

    def to_dict(self) -> Dict[str, float]:
        return {
            "embedding_ms": round(self.embedding_ms, 2),
            "semantic_ms":  round(self.semantic_ms, 2),
            "keyword_ms":   round(self.keyword_ms, 2),
            "fusion_ms":    round(self.fusion_ms, 2),
            "filter_ms":    round(self.filter_ms, 2),
            "rerank_ms":    round(self.rerank_ms, 2),
            "total_ms":     round(self.total_ms, 2),
        }


@dataclass
class HybridSearchResult:
    """
    Complete output of one HybridRetriever.search() call.

    The `debug` dict exposes all intermediate ranked lists so you can
    understand why a chunk was or was not included in the final answer.

    debug keys:
      "semantic_results"  - list[RetrievalResult] from SemanticRetriever
      "keyword_results"   - list[RetrievalResult] from KeywordRetriever
      "fused_results"     - list[RetrievalResult] after RRF
      "filtered_results"  - list[RetrievalResult] after metadata filter
      "reranked_results"  - list[RetrievalResult] after cross-encoder (if used)
    """
    query:   str
    mode:    str
    results: List[RetrievalResult]      # final top-K to send to LLM

    # Intermediate lists — may be empty if the mode doesn't use them
    debug:   Dict[str, List[RetrievalResult]] = field(default_factory=dict)
    latency: LatencyRecord = field(default_factory=LatencyRecord)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query":   self.query,
            "mode":    self.mode,
            "results": [r.to_dict() for r in self.results],
            "latency": self.latency.to_dict(),
        }
