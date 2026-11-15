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
import traceback
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

# ── Phase 6 imports (conditional — tools + financial layer) ──────────────────
_P6_AVAILABLE = False
try:
    from src.planning.analyzer_p6 import ToolAwareAnalyzer
    from src.planning.models_p6 import AnswerPath
    from src.tools.financial_data import FinancialDataTool
    from src.tools.calculator import CalculatorTool
    from src.tools.period_compare import PeriodCompareTool
    from src.tools.document_search import DocumentSearchTool
    from src.orchestration.context_builder_p6 import build_p6_context as _build_p6_ctx
    _P6_AVAILABLE = True
except Exception:
    pass

P6_METRICS = "P6_METRICS"   # Metrics DB lookup + optional calculator
P6_SMART   = "P6_SMART"     # Auto-routes: metrics / calc / docs / hybrid

P6_MODES = {P6_METRICS, P6_SMART}

# ── All mode labels for the UI ────────────────────────────────────────────────
FAISS_MODES = {VECTOR_ONLY, KEYWORD_ONLY, HYBRID, HYBRID_RERANKED}

MODE_LABELS: Dict[str, str] = {
    VECTOR_ONLY:       "Vector (FAISS cosine)",
    KEYWORD_ONLY:      "Keyword (BM25)",
    HYBRID:            "Hybrid — RRF fusion",
    HYBRID_RERANKED:   "Hybrid + Reranking (cross-encoder)",
    POSTGRES_EXACT:    "PostgreSQL — Exact vector scan",
    POSTGRES_HNSW:     "PostgreSQL — HNSW approximate",
    POSTGRES_FILTERED: "PostgreSQL — HNSW + relational filters",
    POSTGRES_FTS:      "PostgreSQL — Full-text search",
    POSTGRES_HYBRID:   "PostgreSQL — Vector + BM25 (RRF)",
    RESEARCH_AGENT:    "Research Agent (Phase 5 — LangGraph)",
    P6_METRICS:        "Phase 6 — Metrics Lookup",
    P6_SMART:          "Phase 6 — Smart (auto-route)",
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
        modes += [POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_EXACT, POSTGRES_HYBRID]
    if _GRAPH_AVAILABLE and (faiss_ok or db_ok):
        modes.append(RESEARCH_AGENT)
    if _P6_AVAILABLE and db_ok:
        modes.append(P6_METRICS)
    if _P6_AVAILABLE and (faiss_ok or db_ok):
        modes.append(P6_SMART)
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

    # ── Phase 6 path ─────────────────────────────────────────────────────────
    if mode in P6_MODES:
        return _run_p6_query(query, mode, top_k, filters, faiss_retriever, t_total)

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
        traceback.print_exc()
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

    t_gen = time.perf_counter()
    try:
        generation = answer_query(query, retrieved_chunks)
    except Exception as exc:
        traceback.print_exc()
        return _error_result(query, mode, f"Generation error: {exc}")
    generate_ms = round((time.perf_counter() - t_gen) * 1000, 2)

    latency = _latency_dict(hybrid_result, t_total)
    latency["generate_ms"] = generate_ms

    return {
        "query": query,
        "mode": mode,
        "answer": generation["answer"],
        "model_used": generation["model_used"],
        "retrieved_chunks": retrieved_chunks,
        "hybrid_result": hybrid_result,
        "latency": latency,
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
        traceback.print_exc()
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

    t_gen = time.perf_counter()
    try:
        generation = answer_query(query, retrieved_chunks)
    except Exception as exc:
        traceback.print_exc()
        return _error_result(query, mode, f"Generation error: {exc}")
    generate_ms = round((time.perf_counter() - t_gen) * 1000, 2)

    return {
        "query": query,
        "mode": mode,
        "answer": generation["answer"],
        "model_used": generation["model_used"],
        "retrieved_chunks": retrieved_chunks,
        "hybrid_result": None,
        "pg_results": results,
        "latency": {
            "embedding_ms": embedding_ms,
            "search_ms":    search_ms,
            "generate_ms":  generate_ms,
            "total_ms":     round((time.perf_counter() - t_total) * 1000, 2),
        },
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Tools + Structured Financial Data + Deterministic Calculator
# ─────────────────────────────────────────────────────────────────────────────

def _run_p6_query(
    query: str,
    mode: str,
    top_k: int,
    filters: Optional[Dict],
    faiss_retriever,
    t_total: float,
) -> Dict[str, Any]:
    """
    Dispatch a Phase 6 query.

    P6_METRICS:  financial metrics DB lookup + optional deterministic calculator.
                 Requires DATABASE_URL and financial_metrics_p6 seeded.
    P6_SMART:    ToolAwareAnalyzer classifies the query and auto-routes:
                   DOCS_ONLY        → falls through to _run_faiss_query
                   METRICS_ONLY     → metrics DB lookup
                   METRICS_PLUS_CALC→ metrics + calculator
                   HYBRID           → metrics + document search + calc
    """
    if not _P6_AVAILABLE:
        return _error_result(query, mode,
            "Phase 6 tools not available. Check src/tools/ and src/financial/ imports.")

    # ── Analyze query ─────────────────────────────────────────────────────────
    t_analyze = time.perf_counter()
    try:
        analyzer = ToolAwareAnalyzer()
        analysis = analyzer.analyze(query)
    except Exception as exc:
        return _error_result(query, mode, f"Phase 6 analyzer error: {exc}")
    analyze_ms = round((time.perf_counter() - t_analyze) * 1000, 2)

    # Determine effective answer path
    if mode == P6_METRICS:
        # Force metrics path regardless of what analyzer says
        effective_path = AnswerPath.METRICS_ONLY
        if analysis.needs_calculation or len(analysis.fiscal_periods) >= 2:
            effective_path = AnswerPath.METRICS_PLUS_CALC
    else:  # P6_SMART — trust analyzer
        effective_path = analysis.answer_path

    # DOCS_ONLY → fall through to existing FAISS retrieval (Phase 3 quality)
    if effective_path == AnswerPath.DOCS_ONLY:
        result = _run_faiss_query(query, HYBRID_RERANKED, top_k, filters,
                                  faiss_retriever, t_total)
        result["mode"] = mode
        result["answer_path"] = "DOCS_ONLY"
        return result

    # ── Open DB session for tool calls ────────────────────────────────────────
    db_session = None
    if _PG_AVAILABLE:
        try:
            from src.db.connection import SessionLocal
            db_session = SessionLocal()
        except Exception:
            pass

    try:
        return _p6_execute(
            query, mode, effective_path, analysis,
            top_k, filters, faiss_retriever, db_session,
            analyze_ms, t_total,
        )
    finally:
        if db_session:
            try:
                db_session.close()
            except Exception:
                pass


def _p6_execute(
    query: str,
    mode: str,
    answer_path,
    analysis,
    top_k: int,
    filters: Optional[Dict],
    faiss_retriever,
    db_session,
    analyze_ms: float,
    t_total: float,
) -> Dict[str, Any]:
    """Run Phase 6 tools, build structured context, generate answer."""
    import re

    fd_tool      = FinancialDataTool(db_session=db_session)
    calc_tool    = CalculatorTool()
    compare_tool = PeriodCompareTool(db_session=db_session)
    doc_tool     = DocumentSearchTool(faiss_retriever=faiss_retriever,
                                      db_session=db_session)

    tool_results:   List[Dict] = []
    metric_rows:    List[Dict] = []
    conflict_detected = False
    conflict_details: List[Dict] = []
    formula:   Optional[str]  = None
    calc_result_summary: Optional[Dict] = None
    doc_chunks: List[Dict] = []

    # ── STEP 1: Metric lookups ────────────────────────────────────────────────
    t_tool = time.perf_counter()

    companies  = analysis.companies or []
    metrics    = analysis.metrics_mentioned or []
    periods    = analysis.fiscal_periods or []

    if answer_path in (AnswerPath.METRICS_ONLY, AnswerPath.METRICS_PLUS_CALC,
                       AnswerPath.HYBRID):
        # Parse period strings → (fiscal_year, fiscal_period)
        parsed_periods: List[tuple] = []
        for p in periods:
            m = re.match(r'(FY|Q[1-4]|CY)(\d{4})', p, re.IGNORECASE)
            if m:
                parsed_periods.append((m.group(1).upper(), int(m.group(2))))
            else:
                try:
                    yr = int(p)
                    parsed_periods.append(("FY", yr))
                except ValueError:
                    pass

        for company in (companies or ["NexaCore Technologies"]):
            for metric in metrics:
                if parsed_periods:
                    for fp, yr in parsed_periods:
                        r = fd_tool.safe_run(
                            company=company, metric=metric,
                            fiscal_year=yr,
                            fiscal_period=fp if fp != "FY" else None,
                        )
                        tool_results.append(r.model_dump_json_safe())
                        if r.is_ok():
                            metric_rows.extend(r.data.get("rows", []))
                            if r.conflict_detected:
                                conflict_detected = True
                                conflict_details.append({
                                    "metric":  metric,
                                    "company": company,
                                    "period":  f"{fp}{yr}",
                                    "values": [
                                        {"value": row["value"],
                                         "unit":  row["unit"],
                                         "source": row.get("source_text", "")[:100]}
                                        for row in r.data.get("rows", [])
                                    ],
                                })
                else:
                    # No period specified — get all years
                    r = fd_tool.safe_run(company=company, metric=metric)
                    tool_results.append(r.model_dump_json_safe())
                    if r.is_ok():
                        metric_rows.extend(r.data.get("rows", []))
                        if r.conflict_detected:
                            conflict_detected = True
                            conflict_details.append({
                                "metric":  metric,
                                "company": company,
                                "period":  "all",
                                "values": [
                                    {"value": row["value"],
                                     "unit":  row["unit"],
                                     "source": row.get("source_text", "")[:100]}
                                    for row in r.data.get("rows", [])
                                ],
                            })

    # ── STEP 2: Calculation (if needed + we have ≥2 metric rows) ─────────────
    if answer_path == AnswerPath.METRICS_PLUS_CALC and len(metric_rows) >= 2:
        # Sort by fiscal_year and try percent_change between oldest and newest
        sorted_rows = sorted(metric_rows, key=lambda r: (r.get("fiscal_year", 0),
                                                          r.get("fiscal_period", "")))
        old_row = sorted_rows[0]
        new_row = sorted_rows[-1]
        same_metric = old_row.get("metric_name") == new_row.get("metric_name")
        same_unit   = old_row.get("unit") == new_row.get("unit")

        if same_metric and same_unit and old_row != new_row:
            calc_r = calc_tool.safe_run(
                operation="percent_change",
                inputs=[
                    {"label": f"FY{old_row.get('fiscal_year')} {old_row.get('metric_name')}",
                     "value": old_row["value"], "unit": old_row["unit"],
                     "provenance": {"source_type": "metric_row",
                                    "source_text": old_row.get("source_text", "")}},
                    {"label": f"FY{new_row.get('fiscal_year')} {new_row.get('metric_name')}",
                     "value": new_row["value"], "unit": new_row["unit"],
                     "provenance": {"source_type": "metric_row",
                                    "source_text": new_row.get("source_text", "")}},
                ],
            )
            tool_results.append(calc_r.model_dump_json_safe())
            if calc_r.is_ok():
                formula = calc_r.data.get("formula")
                calc_result_summary = {
                    "operation": calc_r.data.get("operation"),
                    "formula":   formula,
                    "result":    calc_r.data.get("result"),
                    "unit":      calc_r.data.get("unit"),
                }

    # ── STEP 3: Document search (HYBRID path) ─────────────────────────────────
    if answer_path == AnswerPath.HYBRID:
        doc_r = doc_tool.safe_run(query=query, top_k=top_k, filters=filters)
        tool_results.append(doc_r.model_dump_json_safe())
        if doc_r.is_ok():
            doc_chunks = doc_r.data.get("chunks", [])

    tool_ms = round((time.perf_counter() - t_tool) * 1000, 2)

    # ── STEP 4: Build LLM context ─────────────────────────────────────────────
    context = _build_p6_llm_context(
        query, metric_rows, conflict_details,
        calc_result_summary, doc_chunks,
    )

    # ── STEP 5: Generate answer ───────────────────────────────────────────────
    t_gen = time.perf_counter()
    try:
        generation = _call_llm_with_context(query, context)
    except Exception as exc:
        return _error_result(query, mode, f"Generation error: {exc}")
    generate_ms = round((time.perf_counter() - t_gen) * 1000, 2)

    # Map doc_chunks to the standard retrieved_chunks format for Sources panel
    retrieved_chunks = [
        {
            "chunk_id":      c.get("chunk_id", ""),
            "text":          c.get("text", ""),
            "source":        c.get("source", ""),
            "page_num":      c.get("page", 0),
            "section":       c.get("section", ""),
            "content_type":  c.get("content_type", "text"),
            "score":         c.get("score", 0.0),
            "document":      c.get("source", ""),
            "semantic_score": c.get("score"),
        }
        for c in doc_chunks
    ]

    return {
        "query":            query,
        "mode":             mode,
        "answer":           generation.get("answer", ""),
        "model_used":       generation.get("model_used", ""),
        "retrieved_chunks": retrieved_chunks,
        "hybrid_result":    None,
        "latency": {
            "analyze_ms":  analyze_ms,
            "tool_ms":     tool_ms,
            "generate_ms": generate_ms,
            "total_ms":    round((time.perf_counter() - t_total) * 1000, 2),
        },
        "error": None,
        # ── Phase 6 specific ──────────────────────────────────────────────────
        "answer_path":       answer_path.value if hasattr(answer_path, "value") else str(answer_path),
        "tool_results":      tool_results,
        "conflict_detected": conflict_detected,
        "conflict_details":  conflict_details,
        "metric_rows":       metric_rows,
        "formula":           formula,
        "calc_summary":      calc_result_summary,
        "analysis_p6":       analysis.model_dump_json_safe() if hasattr(analysis, "model_dump_json_safe") else {},
    }


def _build_p6_llm_context(
    query: str,
    metric_rows: List[Dict],
    conflict_details: List[Dict],
    calc_summary: Optional[Dict],
    doc_chunks: List[Dict],
) -> str:
    """Build structured LLM context with separated sections."""
    SEP  = "═" * 60
    DASH = "─" * 50
    parts = [SEP, "RESEARCH QUESTION", query, SEP, ""]

    # Structured data section
    if metric_rows:
        parts.append("STRUCTURED DATA  (from financial database — use these numbers, not your own)")
        parts.append(DASH)
        for r in metric_rows:
            dim = f" [{r.get('dimension_type')}={r.get('dimension_value')}]" \
                  if r.get("dimension_type") else ""
            parts.append(
                f"  {r.get('metric_name','?')}{dim} | "
                f"FY{r.get('fiscal_year','?')}/{r.get('fiscal_period','?')} | "
                f"{r.get('value','?')} {r.get('unit','?')} ({r.get('scale','?')}) | "
                f"Source: {str(r.get('source_text',''))[:80]}"
            )
        if conflict_details:
            parts.append("")
            parts.append("  ⚠  CONFLICTS DETECTED — multiple sources report different values:")
            for cd in conflict_details:
                parts.append(f"  {cd['metric']} {cd['period']}:")
                for v in cd["values"]:
                    parts.append(f"    • {v['value']} {v['unit']} — {v['source']}")
        parts.append("")

    # Computations section
    if calc_summary:
        parts.append("COMPUTATIONS  (deterministic calculator — these results are exact)")
        parts.append(DASH)
        parts.append(f"  Operation : {calc_summary.get('operation')}")
        parts.append(f"  Formula   : {calc_summary.get('formula')}")
        parts.append(f"  Result    : {calc_summary.get('result')} {calc_summary.get('unit','')}")
        parts.append("")

    # Document evidence section
    if doc_chunks:
        parts.append("DOCUMENT EVIDENCE")
        parts.append(DASH)
        for i, c in enumerate(doc_chunks, 1):
            source  = c.get("source", "?")
            page    = c.get("page", "?")
            section = c.get("section", "")
            parts.append(f"  [Source {i}] {source} | p.{page}" +
                         (f" | {section}" if section else ""))
            parts.append(f"  {c.get('text','')[:400]}")
            parts.append("")

    # Instructions
    parts += [
        SEP, "INSTRUCTIONS",
        "• Answer using ONLY the numbers from STRUCTURED DATA and COMPUTATIONS — never invent figures.",
        "• If sources conflict, report ALL values and name each source. Do NOT choose one.",
        "• Cite document sources as (Source N).",
        "• State fiscal period and unit for every number.",
        "• If data is missing, say so explicitly.",
        SEP,
    ]
    return "\n".join(parts)


def _call_llm_with_context(query: str, context: str) -> Dict[str, Any]:
    """Call OpenRouter with a pre-built Phase 6 structured context string."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model   = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")

    if not api_key:
        return {"answer": "[ERROR] OPENROUTER_API_KEY not set.", "model_used": "N/A"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                    "You are a meticulous financial analyst specializing in company due diligence. "
                    "Answer using ONLY the evidence in the context. "
                    "Cite every numeric claim with its source. "
                    "Do NOT perform arithmetic — use numbers from COMPUTATIONS only. "
                    "If sources conflict, report all values."},
                {"role": "user", "content": f"{context}\n\nQUESTION: {query}"},
            ],
            max_tokens=1200,
            temperature=0.1,
        )
        return {"answer": response.choices[0].message.content.strip(), "model_used": model}
    except Exception as exc:
        return {"answer": f"[ERROR] Generation failed: {exc}", "model_used": "N/A"}


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
        # Always use the user's top_k slider (minimum 5 to avoid under-retrieval).
        # Previously only overrode when top_k > 10, so top_k=5 was silently ignored
        # and every task retrieved 10 chunks regardless of the UI setting.
        task_retrieval = max(top_k, 5)
        config = config.model_copy(update={"max_task_retrieval": task_retrieval})

        graph = ResearchGraph(config=config, db_session=db_session)
        state = graph.run(query)
    except Exception as exc:
        traceback.print_exc()
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
