"""
src/ui_backend.py  ─  Thin adapter layer between the Streamlit UI and existing RAG pipeline
============================================================================================

PURPOSE
-------
This module exists to:
  1. Keep ui.py free of pipeline logic
  2. Centralise retriever construction and caching so Streamlit re-runs
     don't reload 400 MB of model weights on every widget interaction
  3. Translate RAG pipeline outputs into a flat, UI-friendly dict without
     changing the underlying retrieval algorithms

WHAT THIS MODULE DOES NOT DO
------------------------------
- It does NOT reimplement retrieval, embeddings, or generation
- It does NOT change any algorithm
- It does NOT import anything that wasn't already used in app.py or scripts/

RETRIEVER MODES AVAILABLE
--------------------------
Phase 3 FAISS modes (always available if data/index/ exists):
  VECTOR_ONLY      — FAISS cosine similarity
  KEYWORD_ONLY     — BM25 lexical matching
  HYBRID           — RRF fusion of semantic + keyword
  HYBRID_RERANKED  — RRF fusion + cross-encoder reranking

Phase 4 PostgreSQL modes (available when DATABASE_URL is set and DB is reachable):
  POSTGRES_EXACT    — Full table scan vector search
  POSTGRES_HNSW     — Approximate HNSW vector search
  POSTGRES_FILTERED — HNSW + relational WHERE filters
  POSTGRES_FTS      — PostgreSQL full-text search
  POSTGRES_HYBRID   — Postgres vector + Phase 3 BM25 via RRF
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Phase 3 imports ───────────────────────────────────────────────────────────
from src.retrieval.models import (
    VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED,
    VALID_MODES, RetrievalResult, HybridSearchResult,
)
from src.retrieval.hybrid import HybridRetriever
from src.generator import answer_query

# ── Phase 4 imports (conditional — DB may not be available) ──────────────────
_PG_AVAILABLE = False
try:
    from src.db.repositories.retrieval import (
        PostgresRetriever, search_chunks,
        POSTGRES_EXACT, POSTGRES_HNSW, POSTGRES_FILTERED,
        POSTGRES_FTS, POSTGRES_HYBRID, POSTGRES_MODES,
    )
    _PG_AVAILABLE = True
except Exception:
    POSTGRES_EXACT = "POSTGRES_EXACT"
    POSTGRES_HNSW = "POSTGRES_HNSW"
    POSTGRES_FILTERED = "POSTGRES_FILTERED"
    POSTGRES_FTS = "POSTGRES_FTS"
    POSTGRES_HYBRID = "POSTGRES_HYBRID"
    POSTGRES_MODES = {POSTGRES_EXACT, POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_HYBRID}

# ── All mode labels for the UI ────────────────────────────────────────────────
FAISS_MODES = {VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED}

MODE_LABELS: Dict[str, str] = {
    VECTOR_ONLY:      "Vector (FAISS cosine)",
    KEYWORD_ONLY:     "Keyword (BM25)",
    HYBRID:           "Hybrid — RRF fusion",
    HYBRID_RERANKED:  "Hybrid + Reranking (cross-encoder)",
    POSTGRES_EXACT:   "PostgreSQL — Exact vector scan",
    POSTGRES_HNSW:    "PostgreSQL — HNSW approximate",
    POSTGRES_FILTERED: "PostgreSQL — HNSW + relational filters",
    POSTGRES_FTS:     "PostgreSQL — Full-text search",
    POSTGRES_HYBRID:  "PostgreSQL — Vector + BM25 (RRF)",
}

INDEX_DIR = str(PROJECT_ROOT / "data" / "index")


# ─────────────────────────────────────────────────────────────────────────────
# Backend status checks (called at startup / sidebar)
# ─────────────────────────────────────────────────────────────────────────────

def check_faiss_index() -> Tuple[bool, str]:
    """Return (available, message) for the Phase 3 FAISS index."""
    idx_path = Path(INDEX_DIR) / "faiss.index"
    chunks_path = Path(INDEX_DIR) / "chunks.json"
    if idx_path.exists() and chunks_path.exists():
        return True, f"FAISS index found ({idx_path.stat().st_size // 1024} KB)"
    missing = []
    if not idx_path.exists():
        missing.append("faiss.index")
    if not chunks_path.exists():
        missing.append("chunks.json")
    return False, f"Missing: {', '.join(missing)} in {INDEX_DIR}. Run: python scripts/ingest.py"


def check_database() -> Tuple[bool, str]:
    """Return (available, message) for PostgreSQL."""
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        return False, "DATABASE_URL not set in .env — Phase 4 modes unavailable"
    if not _PG_AVAILABLE:
        return False, "PostgreSQL imports failed (check psycopg2 / pgvector install)"
    try:
        from src.db.connection import ping_db
        if ping_db():
            return True, "PostgreSQL connected"
        return False, "DATABASE_URL set but connection failed"
    except Exception as exc:
        return False, f"PostgreSQL error: {exc}"


def check_llm() -> Tuple[bool, str]:
    """Return (available, message) for the LLM generation API."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")
    if key:
        return True, f"OpenRouter API key set — model: {model}"
    return False, "OPENROUTER_API_KEY not set in .env — generation will fail"


def get_available_modes(faiss_ok: bool, db_ok: bool) -> List[str]:
    """Return the list of retrieval mode keys that are actually available."""
    modes = []
    if faiss_ok:
        modes += [VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED]
    if db_ok:
        modes += [POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_EXACT]
    return modes


# ─────────────────────────────────────────────────────────────────────────────
# Retriever construction (cached by callers using @st.cache_resource)
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_retriever(mode: str = HYBRID_RERANKED) -> HybridRetriever:
    """
    Build and return a HybridRetriever.

    Heavy — loads FAISS index, BM25 corpus, and optionally the cross-encoder.
    This function is designed to be wrapped in @st.cache_resource so it runs
    only once per Streamlit session.
    """
    return HybridRetriever(
        mode=mode,
        index_dir=INDEX_DIR,
        final_top_k=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Query runner — called on every user question
# ─────────────────────────────────────────────────────────────────────────────

def run_query(
    query: str,
    mode: str,
    top_k: int,
    filters: Optional[Dict] = None,
    faiss_retriever: Optional[HybridRetriever] = None,
) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for one query and return a structured result dict.

    This wraps existing pipeline logic (src/rag_pipeline.py pattern) but adds:
      - Full HybridSearchResult debug info for diagnostics panel
      - Phase 4 PostgreSQL path
      - Timing breakdown
      - Error handling that returns useful messages rather than crashing

    Returns a dict with keys:
      query, mode, answer, model_used,
      retrieved_chunks   — list of chunk dicts (id, text, source, page, scores)
      hybrid_result      — HybridSearchResult or None (for Phase 4)
      latency            — dict with timing info
      error              — str or None
    """
    t_total = time.perf_counter()

    # Normalise filters
    filters = {k: v for k, v in (filters or {}).items() if v not in (None, "", "All")}

    # ── Phase 4 PostgreSQL path ───────────────────────────────────────────────
    if mode in POSTGRES_MODES:
        return _run_pg_query(query, mode, top_k, filters, t_total)

    # ── Phase 3 FAISS path ────────────────────────────────────────────────────
    return _run_faiss_query(query, mode, top_k, filters, faiss_retriever, t_total)


def _run_faiss_query(
    query: str,
    mode: str,
    top_k: int,
    filters: Optional[Dict],
    retriever: Optional[HybridRetriever],
    t_total: float,
) -> Dict[str, Any]:
    if retriever is None:
        return _error_result(query, mode, "FAISS retriever not initialized")

    try:
        # Switch mode without re-loading index
        if retriever.mode != mode:
            retriever.set_mode(mode)
        retriever.final_top_k = top_k

        hybrid_result: HybridSearchResult = retriever.search(query, filters=filters)
    except Exception as exc:
        return _error_result(query, mode, f"Retrieval error: {exc}")

    # Map RetrievalResult objects to generator dict format
    retrieved_chunks = _map_results_to_chunks(hybrid_result.results)

    if not retrieved_chunks:
        return {
            "query": query, "mode": mode,
            "answer": "No relevant context found in the document store.\n\n"
                      "Make sure you have run `python scripts/ingest.py` and "
                      "that your PDFs contain text about this topic.",
            "model_used": "N/A",
            "retrieved_chunks": [],
            "hybrid_result": hybrid_result,
            "latency": _latency_dict(hybrid_result, t_total),
            "error": None,
        }

    try:
        generation = answer_query(query, retrieved_chunks)
    except Exception as exc:
        return _error_result(query, mode, f"Generation error: {exc}")

    return {
        "query": query,
        "mode": mode,
        "answer": generation["answer"],
        "model_used": generation["model_used"],
        "retrieved_chunks": retrieved_chunks,
        "hybrid_result": hybrid_result,
        "latency": _latency_dict(hybrid_result, t_total),
        "error": None,
    }


def _run_pg_query(
    query: str,
    mode: str,
    top_k: int,
    filters: Optional[Dict],
    t_total: float,
) -> Dict[str, Any]:
    if not _PG_AVAILABLE:
        return _error_result(query, mode, "PostgreSQL client libraries not installed")

    try:
        from src.db.connection import get_db
        with get_db() as db:
            pg_retriever = PostgresRetriever(db)
            results: List[RetrievalResult] = pg_retriever.search(
                query, top_k=top_k, filters=filters, mode=mode
            )
            embedding_ms = pg_retriever.last_embedding_ms
            search_ms = pg_retriever.last_search_ms
    except Exception as exc:
        return _error_result(query, mode, f"PostgreSQL retrieval error: {exc}")

    retrieved_chunks = _map_results_to_chunks(results)

    if not retrieved_chunks:
        return {
            "query": query, "mode": mode,
            "answer": "No relevant context found in the PostgreSQL knowledge base.\n\n"
                      "Ensure documents have been ingested (python scripts/ingest_pg.py) "
                      "and embeddings are stored.",
            "model_used": "N/A",
            "retrieved_chunks": [],
            "hybrid_result": None,
            "pg_results": results,
            "latency": {"embedding_ms": embedding_ms, "search_ms": search_ms,
                        "total_ms": (time.perf_counter() - t_total) * 1000},
            "error": None,
        }

    try:
        generation = answer_query(query, retrieved_chunks)
    except Exception as exc:
        return _error_result(query, mode, f"Generation error: {exc}")

    return {
        "query": query,
        "mode": mode,
        "answer": generation["answer"],
        "model_used": generation["model_used"],
        "retrieved_chunks": retrieved_chunks,
        "hybrid_result": None,
        "pg_results": results,
        "latency": {"embedding_ms": embedding_ms, "search_ms": search_ms,
                    "total_ms": (time.perf_counter() - t_total) * 1000},
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _map_results_to_chunks(results: List[RetrievalResult]) -> List[Dict]:
    """
    Convert RetrievalResult objects to the generator's expected dict format.
    Also preserves all score fields and metadata for the diagnostics panel.
    """
    chunks = []
    for r in results:
        chunks.append({
            # Generator expects these exact keys
            "source": r.source,
            "page_num": r.page_start,
            "text": r.text,
            "score": r.best_score(),
            "chunk_id": r.chunk_id,
            # Extra fields for UI display
            "page_end": r.page_end,
            "section": r.section,
            "section_path": r.section_path,
            "content_type": r.content_type,
            "semantic_score": r.semantic_score,
            "semantic_rank": r.semantic_rank,
            "keyword_score": r.keyword_score,
            "keyword_rank": r.keyword_rank,
            "fusion_score": r.fusion_score,
            "rerank_score": r.rerank_score,
            # Phase 4 provenance (from metadata dict, None for FAISS results)
            "document": r.metadata.get("document"),
            "document_type": r.metadata.get("document_type"),
            "version_label": r.metadata.get("version_label"),
            "effective_date": r.metadata.get("effective_date"),
            "is_current": r.metadata.get("is_current"),
            "company": r.metadata.get("company"),
        })
    return chunks


def _latency_dict(hybrid_result: HybridSearchResult, t_total: float) -> Dict:
    """Build latency dict from HybridSearchResult.latency."""
    d = hybrid_result.latency.to_dict()
    d["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
    return d


def _error_result(query: str, mode: str, message: str) -> Dict[str, Any]:
    return {
        "query": query,
        "mode": mode,
        "answer": f"**Error:** {message}",
        "model_used": "N/A",
        "retrieved_chunks": [],
        "hybrid_result": None,
        "latency": {},
        "error": message,
    }
