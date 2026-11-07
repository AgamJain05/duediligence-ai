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

# ── Phase 5 imports (conditional — LangGraph may not be installed) ────────────
_GRAPH_AVAILABLE = False
try:
    from src.orchestration.graph import ResearchGraph
    from src.planning.models import PlanningConfig
    _GRAPH_AVAILABLE = True
except Exception:
    pass

RESEARCH_AGENT = "RESEARCH_AGENT"

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
    RESEARCH_AGENT:   "Research Agent (Phase 5 — LangGraph)",
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
    if _GRAPH_AVAILABLE and (faiss_ok or db_ok):
        modes.append(RESEARCH_AGENT)
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

    # ── Phase 5 Research Agent path ───────────────────────────────────────────
    if mode == RESEARCH_AGENT:
        return _run_graph_query(query, top_k, filters, t_total)

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
# Phase 5: Research Agent (LangGraph)
# ─────────────────────────────────────────────────────────────────────────────

def _run_graph_query(
    query: str,
    top_k: int,
    filters: Optional[Dict],
    t_total: float,
) -> Dict[str, Any]:
    """
    Run the Phase 5 LangGraph research graph for one query.

    Uses PostgreSQL as the retrieval backend if available, otherwise
    falls back to FAISS (db_session=None triggers FAISS inside the graph).
    """
    if not _GRAPH_AVAILABLE:
        return _error_result(
            query, RESEARCH_AGENT,
            "Phase 5 (LangGraph) dependencies are not installed. "
            "Run: pip install langgraph",
        )

    # Attempt to open a DB session for Phase 4-backed retrieval inside the graph.
    # Falls back to FAISS automatically when db_session=None.
    db_session = None
    if _PG_AVAILABLE:
        try:
            from src.db.connection import SessionLocal
            db_session = SessionLocal()
        except Exception:
            db_session = None

    try:
        config = PlanningConfig.default()
        # Respect the user's top_k slider for per-task retrieval depth
        if top_k > config.max_task_retrieval:
            config = config.model_copy(update={"max_task_retrieval": top_k})

        graph = ResearchGraph(config=config, db_session=db_session)
        state = graph.run(query)
    except Exception as exc:
        return _error_result(query, RESEARCH_AGENT, f"Research graph execution error: {exc}")
    finally:
        if db_session is not None:
            try:
                db_session.close()
            except Exception:
                pass

    return _state_to_result(query, state, t_total)


def _state_to_result(query: str, state: Dict[str, Any], t_total: float) -> Dict[str, Any]:
    """Map a completed ResearchState dict to the standard UI result dict."""
    if state.get("fatal_error"):
        return _error_result(query, RESEARCH_AGENT, state["fatal_error"])

    analysis = state.get("analysis") or {}

    # ── Flatten evidence list → retrieved_chunks (same format as FAISS/PG paths) ──
    retrieved_chunks: List[Dict] = []
    for ev in state.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        retrieved_chunks.append({
            # Fields the generator and source display expect
            "chunk_id":       ev.get("chunk_id", ""),
            "text":           ev.get("text", ""),
            "source":         ev.get("source", ""),
            "page_num":       ev.get("page_start", 0),
            "page_end":       ev.get("page_end", 0),
            "section":        ev.get("section", ""),
            "section_path":   ev.get("section_path", []),
            "content_type":   ev.get("content_type", "text"),
            "score":          ev.get("rerank_score") or ev.get("semantic_score") or 0.0,
            # Phase 4 provenance (populated when DB backend used)
            "document":       ev.get("document", ""),
            "document_type":  ev.get("document_type", ""),
            "version_label":  ev.get("version_label", ""),
            "effective_date": ev.get("effective_date"),
            "is_current":     ev.get("is_current", False),
            "company":        ev.get("company", ""),
            # Score breakdown (rerank + semantic only; no keyword/fusion in Phase 5)
            "semantic_score": ev.get("semantic_score"),
            "rerank_score":   ev.get("rerank_score"),
            "keyword_score":  None,
            "fusion_score":   None,
            # Phase 5 provenance
            "task_id":        ev.get("task_id", ""),
        })

    # ── Evidence grouped by task (for the task-panel display) ─────────────────
    evidence_by_task: Dict[str, List[Dict]] = {}
    for task_id, records in (state.get("evidence_by_task") or {}).items():
        evidence_by_task[task_id] = [
            {
                "chunk_id":      ev.get("chunk_id", ""),
                "text":          ev.get("text", ""),
                "source":        ev.get("source", ""),
                "page_num":      ev.get("page_start", 0),
                "document":      ev.get("document", ""),
                "section":       ev.get("section", ""),
                "content_type":  ev.get("content_type", "text"),
                "semantic_score": ev.get("semantic_score"),
                "rerank_score":   ev.get("rerank_score"),
                "score":          ev.get("rerank_score") or ev.get("semantic_score") or 0.0,
            }
            for ev in records
            if isinstance(ev, dict)
        ]

    latency = dict(state.get("latency") or {})
    latency["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)

    return {
        "query":                  query,
        "mode":                   RESEARCH_AGENT,
        "answer":                 state.get("answer") or "_No answer generated._",
        "model_used":             os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it"),
        "retrieved_chunks":       retrieved_chunks,
        "hybrid_result":          None,
        "latency":                latency,
        "error":                  state.get("fatal_error"),
        # ── Phase 5 extras ────────────────────────────────────────────────────
        "query_type":             analysis.get("query_type", "SIMPLE"),
        "route":                  state.get("route", "simple"),
        "requires_decomposition": analysis.get("requires_decomposition", False),
        "research_plan":          state.get("research_plan"),
        "evidence_by_task":       evidence_by_task,
        "plan_warnings":          state.get("errors") or [],
        "classifier_method":      analysis.get("classifier_method", "rules"),
        "classifier_confidence":  analysis.get("confidence", 1.0),
        "classifier_reasoning":   analysis.get("classifier_reasoning", ""),
        "entities":               analysis.get("entities") or [],
        "topics":                 analysis.get("topics") or [],
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
