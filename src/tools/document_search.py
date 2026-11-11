"""
src/tools/document_search.py — Wrapper over Phase 4/3 retrieval for the tool framework.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool, ToolError, ToolResult, Provenance


class DocumentSearchTool(BaseTool):
    name = "search_documents"
    description = (
        "Search the document corpus using semantic vector search + BM25 hybrid retrieval. "
        "Returns relevant text passages with source attribution. "
        "Use for qualitative information: strategy, risk factors, management commentary, "
        "narrative context. Use search_financial_data for precise numeric lookups."
    )

    def __init__(self, faiss_retriever=None, db_session=None):
        self._retriever = faiss_retriever
        self._session = db_session

    def run(
        self,
        query: str,
        top_k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
    ) -> ToolResult:
        if self._retriever is None and self._session is None:
            return ToolResult(
                tool=self.name, status="ERROR",
                error=ToolError(code="NO_RETRIEVER", message="No retrieval backend available."),
            )

        filters = filters or {}

        # Try FAISS retriever first
        if self._retriever is not None:
            try:
                from src.retrieval.models import HYBRID_RERANKED
                use_mode = mode or HYBRID_RERANKED
                if self._retriever.mode != use_mode:
                    try:
                        self._retriever.set_mode(use_mode)
                    except Exception:
                        pass
                self._retriever.final_top_k = top_k
                hybrid_result = self._retriever.search(query, filters=filters)
                chunks = hybrid_result.results
            except Exception as exc:
                return ToolResult(
                    tool=self.name, status="ERROR",
                    error=ToolError(code="RETRIEVAL_ERROR", message=str(exc)),
                )
        else:
            # PostgreSQL path
            try:
                from src.db.repositories.retrieval import PostgresRetriever, POSTGRES_HYBRID
                pg = PostgresRetriever(self._session)
                chunks = pg.search(query, top_k=top_k, filters=filters, mode=POSTGRES_HYBRID)
            except Exception as exc:
                return ToolResult(
                    tool=self.name, status="ERROR",
                    error=ToolError(code="PG_RETRIEVAL_ERROR", message=str(exc)),
                )

        if not chunks:
            return ToolResult(
                tool=self.name, status="EMPTY",
                data={"query": query, "chunks": []},
            )

        chunk_dicts = []
        provenance = []
        for c in chunks:
            # Handle both RetrievalResult objects and dicts
            if hasattr(c, "text"):
                text = c.text
                source = getattr(c, "source", "")
                page = getattr(c, "page_start", 0)
                section = getattr(c, "section", "")
                chunk_id = getattr(c, "chunk_id", "")
                score = c.best_score() if hasattr(c, "best_score") else 0.0
            else:
                text = c.get("text", "")
                source = c.get("source", "")
                page = c.get("page_num", 0)
                section = c.get("section", "")
                chunk_id = c.get("chunk_id", "")
                score = c.get("score", 0.0)

            chunk_dicts.append({
                "chunk_id": chunk_id,
                "text": text,
                "source": source,
                "page": page,
                "section": section,
                "score": float(score),
            })
            provenance.append(Provenance(
                source_type="document_chunk",
                chunk_id=chunk_id,
                source_text=text[:200],
                section=section,
                page=page,
            ))

        return ToolResult(
            tool=self.name,
            status="SUCCESS",
            data={"query": query, "chunks": chunk_dicts, "count": len(chunk_dicts)},
            provenance=provenance,
        )
