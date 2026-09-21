"""
src/db/repositories/retrieval.py  ─  PostgreSQL vector + relational retrieval
══════════════════════════════════════════════════════════════════════════════════
Phase 4: PostgresRetriever — the Phase 4 counterpart to FAISS SemanticRetriever.

RETRIEVAL MODES
───────────────
POSTGRES_EXACT:     Full table scan vector search (bypasses HNSW index)
POSTGRES_HNSW:      Approximate nearest-neighbor via HNSW index (default)
POSTGRES_FILTERED:  Relational WHERE filters + vector search
POSTGRES_FTS:       PostgreSQL full-text search (tsvector/tsquery)
POSTGRES_HYBRID:    PostgreSQL vector + Phase 3 BM25, fused via RRF

FILTER → VECTOR SEARCH (not VECTOR → FILTER)
──────────────────────────────────────────────
Phase 3 applied filters AFTER retrieval (post-fusion filter on a small result set).
Phase 4 applies filters AS WHERE CLAUSES in SQL, before the planner executes
the vector scan. This is the correct approach for metadata-gated retrieval:

  Correct  : WHERE company=X AND doc_type=Y  →  ORDER BY embedding <=> $q  LIMIT K
  Incorrect: ORDER BY embedding <=> $q LIMIT K  →  filter in Python for X,Y

The "incorrect" approach can miss relevant chunks because the pre-filter set
was too small. pgvector's planner handles the WHERE + ORDER BY together.

DISTANCE METRIC
────────────────
Operator: <=>  (cosine distance)
Rationale: BAAI/bge-base-en-v1.5 produces L2-normalized vectors.
           For normalized vectors: cosine_distance = 1 - inner_product.
           Similarity = 1 - distance. Results sorted ascending by distance.

The HNSW index was built with vector_cosine_ops, so <=> uses the index.
Using <#> (negative inner product) would NOT use this index.

EXACT VS APPROXIMATE SEARCH
─────────────────────────────
POSTGRES_EXACT: SET LOCAL enable_indexscan = off
  Forces PostgreSQL to ignore the HNSW index and perform a sequential scan.
  This is exact nearest-neighbor — every vector is compared.
  Used for: benchmarking recall of HNSW, verifying correctness.

POSTGRES_HNSW: Default — planner uses the HNSW index automatically.
  This is approximate nearest-neighbor — a graph traversal that finds
  good-but-not-guaranteed nearest neighbors faster than exact search.
  Quality controlled by hnsw.ef_search (set in connection.py).

PROVENANCE
───────────
Every result carries the full provenance chain:
  chunk_id → document_version → document → company
  Plus: effective_date, published_date, section, page
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.models import Chunk, Company, Document, DocumentVersion, Section
from src.embeddings import embed_query, get_embedding_dim
from src.retrieval.models import (
    HybridSearchResult,
    LatencyRecord,
    RetrievalResult,
)

# ── Phase 4 retrieval mode constants ─────────────────────────────────────────
POSTGRES_EXACT    = "POSTGRES_EXACT"
POSTGRES_HNSW     = "POSTGRES_HNSW"
POSTGRES_FILTERED = "POSTGRES_FILTERED"
POSTGRES_FTS      = "POSTGRES_FTS"
POSTGRES_HYBRID   = "POSTGRES_HYBRID"

POSTGRES_MODES = {POSTGRES_EXACT, POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_HYBRID}


def _build_provenance_result(
    row: Any,
    rank: int,
    distance: float,
    mode: str,
) -> RetrievalResult:
    """
    Convert a SQL result row into a RetrievalResult with full provenance.

    Every Phase 4 result carries:
      - company name
      - document title, type
      - document version label, effective_date, published_date
      - section title, section_path
      - page range
      - source filename
      - chunk_id (hash)
      - similarity score (1 - cosine_distance)
    """
    similarity = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0

    metadata = {
        # Provenance
        "company":              row.company_name,
        "company_id":           row.company_id,
        "document":             row.document_title,
        "document_type":        row.document_type,
        "document_id":          row.document_id,
        "document_version_id":  row.document_version_id,
        "version_label":        row.version_label,
        "effective_date":       str(row.effective_date) if row.effective_date else None,
        "published_date":       str(row.published_date) if row.published_date else None,
        "is_current":           row.is_current,
        "source_filename":      row.source_filename,
        # Location
        "page_start":           row.page_start,
        "page_end":             row.page_end,
        "section":              row.section_title or "",
        "section_path":         row.section_path or [],
        # Content
        "content_type":         row.content_type,
        "token_count":          row.token_count,
        "chunk_hash":           row.chunk_hash,
        # Scores
        "cosine_distance":      distance,
        "retrieval_mode":       mode,
    }

    return RetrievalResult(
        chunk_id      = row.chunk_hash,
        text          = row.chunk_text,
        source        = row.source_filename or "",
        page_start    = row.page_start,
        page_end      = row.page_end,
        section       = row.section_title or "",
        section_path  = row.section_path or [],
        content_type  = row.content_type,
        metadata      = metadata,
        semantic_score = similarity,
        semantic_rank  = rank,
    )


def _build_filters_sql(filters: Optional[Dict]) -> tuple[str, Dict]:
    """
    Convert a filter dict into a SQL WHERE clause fragment and params dict.

    Supported filters:
      company         - company name exact match (case-insensitive)
      company_id      - company primary key
      document_type   - document type exact match
      document_id     - document primary key
      is_current      - boolean (True = only current versions)
      effective_date_lte - datetime.date: versions with effective_date <= this
      effective_date_gte - datetime.date: versions with effective_date >= this
      version_label   - exact version label match
      section         - substring match on section title
      content_type    - exact content_type match
      page_start_gte  - chunk.page_start >= value
      page_end_lte    - chunk.page_end <= value

    Returns (where_clause, params) where where_clause starts with "AND " if non-empty.

    WHY PARAMETERIZED?
    Using SQLAlchemy :param syntax prevents SQL injection. The query is
    parameterized even for string values.
    """
    if not filters:
        return "", {}

    clauses = []
    params = {}

    if "company" in filters:
        clauses.append("LOWER(co.name) = LOWER(:filter_company)")
        params["filter_company"] = filters["company"]

    if "company_id" in filters:
        clauses.append("co.id = :filter_company_id")
        params["filter_company_id"] = int(filters["company_id"])

    if "document_type" in filters:
        clauses.append("LOWER(d.document_type) = LOWER(:filter_doc_type)")
        params["filter_doc_type"] = filters["document_type"]

    if "document_id" in filters:
        clauses.append("d.id = :filter_document_id")
        params["filter_document_id"] = int(filters["document_id"])

    if "is_current" in filters:
        val = "TRUE" if filters["is_current"] else "FALSE"
        clauses.append(f"dv.is_current = {val}")

    if "effective_date_lte" in filters:
        clauses.append("dv.effective_date <= :filter_eff_lte")
        params["filter_eff_lte"] = filters["effective_date_lte"]

    if "effective_date_gte" in filters:
        clauses.append("dv.effective_date >= :filter_eff_gte")
        params["filter_eff_gte"] = filters["effective_date_gte"]

    if "version_label" in filters:
        clauses.append("LOWER(dv.version_label) = LOWER(:filter_version)")
        params["filter_version"] = filters["version_label"]

    if "section" in filters:
        clauses.append("LOWER(s.title) LIKE LOWER(:filter_section)")
        params["filter_section"] = f"%{filters['section']}%"

    if "content_type" in filters:
        clauses.append("LOWER(c.content_type) = LOWER(:filter_content_type)")
        params["filter_content_type"] = filters["content_type"]

    if "page_start_gte" in filters:
        clauses.append("c.page_start >= :filter_page_start")
        params["filter_page_start"] = int(filters["page_start_gte"])

    if "page_end_lte" in filters:
        clauses.append("c.page_end <= :filter_page_end")
        params["filter_page_end"] = int(filters["page_end_lte"])

    if not clauses:
        return "", {}

    return "AND " + " AND ".join(clauses), params


# ── Base SQL query template ───────────────────────────────────────────────────
# This is the core vector + relational retrieval query.
# Filters are injected as WHERE clauses before the ORDER BY.
# The distance operator <=> computes cosine distance (1 - cosine_similarity).
_BASE_VECTOR_SQL = """
    SELECT
        c.id                                AS chunk_db_id,
        c.chunk_hash                        AS chunk_hash,
        c.text                              AS chunk_text,
        c.content_type,
        c.page_start,
        c.page_end,
        c.token_count,
        c.embedding <=> CAST(:query_vec AS vector)  AS distance,
        dv.id                               AS document_version_id,
        dv.version_label,
        dv.effective_date,
        dv.published_date,
        dv.is_current,
        dv.source_filename,
        d.id                                AS document_id,
        d.title                             AS document_title,
        d.document_type,
        co.id                               AS company_id,
        co.name                             AS company_name,
        s.title                             AS section_title,
        s.section_path
    FROM chunks c
    JOIN document_versions dv ON c.document_version_id = dv.id
    JOIN documents         d  ON dv.document_id = d.id
    JOIN companies         co ON d.company_id = co.id
    LEFT JOIN sections     s  ON c.section_id = s.id
    WHERE c.embedding IS NOT NULL
    {extra_filters}
    ORDER BY c.embedding <=> CAST(:query_vec AS vector)
    LIMIT :top_k
"""

_FTS_SQL = """
    SELECT
        c.id                                AS chunk_db_id,
        c.chunk_hash                        AS chunk_hash,
        c.text                              AS chunk_text,
        c.content_type,
        c.page_start,
        c.page_end,
        c.token_count,
        ts_rank_cd(to_tsvector('english', c.text), plainto_tsquery('english', :query_text)) AS ts_rank,
        dv.id                               AS document_version_id,
        dv.version_label,
        dv.effective_date,
        dv.published_date,
        dv.is_current,
        dv.source_filename,
        d.id                                AS document_id,
        d.title                             AS document_title,
        d.document_type,
        co.id                               AS company_id,
        co.name                             AS company_name,
        s.title                             AS section_title,
        s.section_path
    FROM chunks c
    JOIN document_versions dv ON c.document_version_id = dv.id
    JOIN documents         d  ON dv.document_id = d.id
    JOIN companies         co ON d.company_id = co.id
    LEFT JOIN sections     s  ON c.section_id = s.id
    WHERE to_tsvector('english', c.text) @@ plainto_tsquery('english', :query_text)
    {extra_filters}
    ORDER BY ts_rank DESC
    LIMIT :top_k
"""


class PostgresRetriever:
    """
    Vector + relational retriever backed by PostgreSQL + pgvector.

    Replaces FAISS SemanticRetriever for Phase 4 retrieval modes.
    Phase 1–3 FAISS retrievers remain available; this is additive.

    Usage:
        retriever = PostgresRetriever(db_session)
        results = retriever.search(
            "What are the key risks?",
            top_k=5,
            filters={"is_current": True, "document_type": "annual_review"},
            mode=POSTGRES_HNSW,
        )

    All results include full provenance (company, document, version, section, page).
    """

    def __init__(self, db: Session) -> None:
        """
        Args:
            db: Active SQLAlchemy session (from get_db() context manager)
        """
        self._db = db
        self.last_embedding_ms: float = 0.0
        self.last_search_ms:    float = 0.0

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        mode: str = POSTGRES_HNSW,
        ef_search: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Run PostgreSQL vector or full-text search.

        Args:
            query:     User query string
            top_k:     Number of results to return
            filters:   Optional relational filters (see _build_filters_sql)
            mode:      Retrieval mode (POSTGRES_EXACT, POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS)
            ef_search: Override hnsw.ef_search for this query (None = use connection default)

        Returns:
            List of RetrievalResult with full provenance metadata
        """
        if not query.strip():
            return []

        if mode not in POSTGRES_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Valid: {POSTGRES_MODES}")

        if mode == POSTGRES_FTS:
            return self._search_fts(query, top_k, filters)

        # ── Embed query ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        query_vec = embed_query(query)  # shape (1, 768)
        self.last_embedding_ms = (time.perf_counter() - t0) * 1000

        # ── Validate dimension ────────────────────────────────────────────────
        actual_dim = query_vec.shape[-1]
        from src.db.models import EMBEDDING_DIM
        if actual_dim != EMBEDDING_DIM:
            raise ValueError(
                f"[PostgresRetriever] Query embedding dim={actual_dim} "
                f"!= configured EMBEDDING_DIM={EMBEDDING_DIM}"
            )

        # Format as pgvector string: "[0.1, 0.2, ...]"
        vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vec[0]) + "]"

        # ── Build SQL ─────────────────────────────────────────────────────────
        filter_clause, filter_params = _build_filters_sql(filters)
        sql_str = _BASE_VECTOR_SQL.format(extra_filters=filter_clause)

        params = {"query_vec": vec_str, "top_k": top_k, **filter_params}

        # ── Execute ───────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        try:
            if mode == POSTGRES_EXACT:
                # Disable index scan to force exact brute-force search
                self._db.execute(text("SET LOCAL enable_indexscan = off"))

            if ef_search is not None:
                self._db.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

            rows = self._db.execute(text(sql_str), params).fetchall()
        finally:
            if mode == POSTGRES_EXACT:
                self._db.execute(text("SET LOCAL enable_indexscan = on"))

        self.last_search_ms = (time.perf_counter() - t1) * 1000

        # ── Convert to RetrievalResult ────────────────────────────────────────
        results = []
        for rank, row in enumerate(rows, start=1):
            results.append(_build_provenance_result(row, rank, row.distance, mode))

        return results

    def _search_fts(
        self, query: str, top_k: int, filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """
        Full-text search using PostgreSQL tsvector/tsquery.

        WHY FTS?
        Provides a database-native keyword search to compare against:
          - Phase 3 BM25 (in-memory, Python)
          - Phase 4 vector search

        Uses plainto_tsquery which parses the query as plain text
        (no need for the user to use tsquery syntax).

        Ranking: ts_rank_cd (cover density) rather than ts_rank, because
        it accounts for chunk size — otherwise short chunks with one hit
        rank the same as long chunks with many hits.
        """
        filter_clause, filter_params = _build_filters_sql(filters)
        sql_str = _FTS_SQL.format(extra_filters=filter_clause)
        params = {"query_text": query, "top_k": top_k, **filter_params}

        t1 = time.perf_counter()
        rows = self._db.execute(text(sql_str), params).fetchall()
        self.last_search_ms = (time.perf_counter() - t1) * 1000

        results = []
        for rank, row in enumerate(rows, start=1):
            # FTS score is ts_rank (higher=better), not distance
            ts_score = float(row.ts_rank) if row.ts_rank is not None else 0.0
            result = _build_provenance_result(row, rank, 1.0 - ts_score, POSTGRES_FTS)
            result.keyword_score = ts_score
            result.keyword_rank  = rank
            results.append(result)

        return results

    def count_candidate_chunks(self, filters: Optional[Dict] = None) -> int:
        """
        Return the number of embedded chunks matching the given filters.

        Useful for understanding how many candidates a vector search has
        to work with before running the actual query.
        """
        filter_clause, filter_params = _build_filters_sql(filters)
        sql = f"""
            SELECT COUNT(*) FROM chunks c
            JOIN document_versions dv ON c.document_version_id = dv.id
            JOIN documents         d  ON dv.document_id = d.id
            JOIN companies         co ON d.company_id = co.id
            LEFT JOIN sections     s  ON c.section_id = s.id
            WHERE c.embedding IS NOT NULL
            {filter_clause}
        """
        result = self._db.execute(text(sql), filter_params).scalar()
        return int(result or 0)

    def explain_query(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        mode: str = POSTGRES_HNSW,
    ) -> str:
        """
        Return the EXPLAIN ANALYZE output for a vector query.

        WHY: EXPLAIN ANALYZE shows:
          - Whether the HNSW index is used (look for "Index Scan using chunks_embedding_hnsw_idx")
          - Estimated vs actual row counts (planner accuracy)
          - Execution time breakdown by node
          - Buffer I/O statistics

        This is the authoritative way to verify index usage — don't assume
        the index is used just because it exists.

        Usage:
            plan = retriever.explain_query("key risks", top_k=5)
            print(plan)
        """
        if not query.strip():
            return "(empty query)"

        query_vec = embed_query(query)
        vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vec[0]) + "]"

        filter_clause, filter_params = _build_filters_sql(filters)
        sql_str = _BASE_VECTOR_SQL.format(extra_filters=filter_clause)
        explain_sql = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + sql_str

        params = {"query_vec": vec_str, "top_k": top_k, **filter_params}

        if mode == POSTGRES_EXACT:
            self._db.execute(text("SET LOCAL enable_indexscan = off"))

        rows = self._db.execute(text(explain_sql), params).fetchall()

        if mode == POSTGRES_EXACT:
            self._db.execute(text("SET LOCAL enable_indexscan = on"))

        return "\n".join(str(r[0]) for r in rows)


def search_chunks(
    db: Session,
    query: str,
    top_k: int = 10,
    filters: Optional[Dict] = None,
    mode: str = POSTGRES_HNSW,
    ef_search: Optional[int] = None,
) -> List[RetrievalResult]:
    """
    Top-level search function — the Phase 4 query API.

    This is the primary entry point for Phase 4 retrieval.
    Creates a PostgresRetriever and runs one search call.

    Args:
        db:        Active SQLAlchemy session
        query:     User query string
        top_k:     Number of results
        filters:   Optional relational filters
        mode:      POSTGRES_EXACT | POSTGRES_HNSW | POSTGRES_FILTERED | POSTGRES_FTS
        ef_search: Override hnsw.ef_search for this query

    Returns:
        List[RetrievalResult] with full provenance
    """
    retriever = PostgresRetriever(db)
    return retriever.search(query, top_k=top_k, filters=filters, mode=mode, ef_search=ef_search)
