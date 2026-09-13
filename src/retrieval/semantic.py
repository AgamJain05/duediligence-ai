"""
src/retrieval/semantic.py  -  Dense vector retrieval
====================================================
Phase 3: Wraps the Phase 2 FAISS index as a clean SemanticRetriever.

WHY SEMANTIC RETRIEVAL?
------------------------
Dense embeddings map text into a high-dimensional vector space where
semantically similar texts are geometrically close (small cosine distance).

Example:
  "How dependent is the company on external cloud providers?"
  "Cloud infrastructure concentration risk"

These two phrases share NO keywords, but their embedding vectors are close
because the model learned that "dependent on external cloud" and "cloud
infrastructure concentration" describe the same concept.

WHY THIS CAN FAIL
------------------
For exact terms, semantic search often finds the right neighborhood but
may rank a semantically-related chunk ABOVE the one with the exact figure.

Example:
  Query: "OrionVault FY2024 revenue 154"
  
  The chunk with the exact number 154 may rank below a chunk about "revenue
  growth" because the embedding model weights the concept (revenue, growth)
  more than the exact digit (154).

This is why keyword retrieval complements semantic retrieval.

CANDIDATE POOL SIZE
--------------------
We retrieve top_k=20 candidates by default, not the final top_k=5.
This is intentional: semantic retrieval generates a candidate pool for
the reranker to refine. The reranker can only pick from what retrieval gives it.

  retrieve many (20) -> rerank -> keep small set (5)
  NOT: retrieve few (5) -> rerank

Pipeline:
  embed_query() -> FAISS search -> map to RetrievalResult
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from src.embeddings import embed_query
from src.retrieval.models import RetrievalResult

# ── Paths (must match what scripts/ingest.py writes) ─────────────────────────
DEFAULT_INDEX_DIR = Path("data/index")
_SA_INDEX_FILE    = "faiss_structure_aware.index"
_SA_CHUNKS_FILE   = "chunks_structure_aware.json"
_P1_INDEX_FILE    = "faiss.index"
_P1_CHUNKS_FILE   = "chunks.json"


def _chunk_to_result(chunk: Dict, rank: int, score: float) -> RetrievalResult:
    """Convert a raw chunk dict to a RetrievalResult with semantic scores."""
    return RetrievalResult(
        chunk_id      = chunk.get("chunk_id", ""),
        text          = chunk.get("text", ""),
        source        = chunk.get("source_filename") or chunk.get("source", ""),
        page_start    = chunk.get("page_start") or chunk.get("page_num", 0),
        page_end      = chunk.get("page_end", chunk.get("page_start") or chunk.get("page_num", 0)),
        section       = chunk.get("section", ""),
        section_path  = chunk.get("section_path") or [],
        content_type  = chunk.get("content_type", "text"),
        metadata      = chunk,
        semantic_score = score,
        semantic_rank  = rank,
    )


class SemanticRetriever:
    """
    Dense vector retriever backed by FAISS IndexFlatIP.

    Loads the Phase-2 structure-aware index by default.
    Falls back to Phase-1 index if Phase-2 is unavailable.

    Usage:
        retriever = SemanticRetriever()
        results = retriever.search("What are the key risks?", top_k=20)

    Each result has semantic_score (cosine similarity) and semantic_rank set.
    """

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        prefer_phase2: bool = True,
    ) -> None:
        """
        Load the FAISS index and chunk store from disk.

        prefer_phase2=True: use structure-aware chunks (Phase 2)
        prefer_phase2=False: use fixed chunks (Phase 1)

        WHY load at construction?
        Loading the FAISS index takes ~10 ms but must only happen ONCE per
        process.  Doing it at construction means the retriever is ready for
        many queries without re-loading.
        """
        index_dir = Path(index_dir)

        if prefer_phase2:
            index_path  = index_dir / _SA_INDEX_FILE
            chunks_path = index_dir / _SA_CHUNKS_FILE
            if not index_path.exists():
                # Graceful fallback to Phase 1
                print(
                    "[SemanticRetriever] Phase-2 index not found. "
                    "Falling back to Phase-1 index."
                )
                index_path  = index_dir / _P1_INDEX_FILE
                chunks_path = index_dir / _P1_CHUNKS_FILE
        else:
            index_path  = index_dir / _P1_INDEX_FILE
            chunks_path = index_dir / _P1_CHUNKS_FILE

        if not index_path.exists():
            raise FileNotFoundError(
                f"[SemanticRetriever] FAISS index not found: {index_path}\n"
                "  Run: python scripts/ingest.py --strategy structure_aware"
            )

        self._index  = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as fh:
            self._chunks = json.load(fh)

        print(
            f"[SemanticRetriever] Loaded {self._index.ntotal} vectors "
            f"({index_path.name}) + {len(self._chunks)} chunks"
        )

        # Timing (set after each search call, useful for benchmarking)
        self.last_embedding_ms: float = 0.0
        self.last_search_ms:    float = 0.0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[RetrievalResult]:
        """
        Receives : query string (raw user question)
                   top_k     — candidate pool size (should be LARGER than final K)
        Returns  : list[RetrievalResult] sorted by semantic_score descending

        Timing:
          self.last_embedding_ms  — time to embed the query
          self.last_search_ms     — time for FAISS search

        NOTE: top_k is the CANDIDATE POOL size, not the final result count.
        The caller (HybridRetriever) will further filter/rerank these.
        Retrieving only 5 before reranking would defeat the purpose of reranking.
        """
        if not query.strip():
            return []

        # Step 1: Embed the query
        t0 = time.perf_counter()
        query_vec = embed_query(query)   # shape (1, 768)
        self.last_embedding_ms = (time.perf_counter() - t0) * 1000

        # Step 2: FAISS search
        t1 = time.perf_counter()
        effective_k = min(top_k, self._index.ntotal)
        if effective_k == 0:
            return []

        scores, indices = self._index.search(query_vec, effective_k)
        self.last_search_ms = (time.perf_counter() - t1) * 1000

        # Step 3: Map FAISS indices to chunks
        results: List[RetrievalResult] = []
        rank = 1
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue   # FAISS padding
            if idx >= len(self._chunks):
                continue   # safety
            chunk = self._chunks[int(idx)]
            results.append(_chunk_to_result(chunk, rank, float(score)))
            rank += 1

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Return a raw chunk dict by chunk_id (linear scan, OK for small corpora)."""
        for chunk in self._chunks:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None

    def all_chunks(self) -> List[Dict]:
        """Return all chunk dicts — used by KeywordRetriever to build BM25 index."""
        return self._chunks
