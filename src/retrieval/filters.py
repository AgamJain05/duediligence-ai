"""
src/retrieval/filters.py  -  Metadata-based chunk filtering
=============================================================
Phase 3: Optional post-fusion filter applied to the candidate pool.

FILTERING vs RETRIEVAL
-----------------------
Metadata filtering and semantic retrieval solve DIFFERENT problems:

  Retrieval:  "Given a query, which chunks are RELEVANT?"
              Uses embedding similarity or BM25 term overlap.
              Relevance is a measure of content.

  Filtering:  "Given a candidate pool, which chunks QUALIFY?"
              Uses structured metadata fields.
              Qualification is a hard constraint.

EXAMPLE:
  Query: "What are the financial results?"
  Filter: content_type="table"

  Without filter: retrieves all relevant chunks, text + tables mixed
  With filter:    only table chunks pass through
                  → guarantees the LLM sees structured financial data

WHY FILTER AFTER FUSION?
--------------------------
Filtering BEFORE retrieval would reduce the search space — fewer chunks
for semantic search and BM25 to consider. This can hurt recall:

  Problem: the best chunk for a query might be filtered out before
           retrieval even sees it.

Filtering AFTER fusion preserves recall from both retrieval systems and
then enforces constraints. This is the safer design for Phase 3.

The trade-off: filtering after fusion doesn't speed up retrieval.
If speed is critical, pre-filtering can be added later as an optimization.

AVAILABLE FILTERS
------------------
  section         - exact or substring match on section heading
  content_type    - exact match: "text", "table", "list", "mixed"
  page_start      - minimum page number (inclusive)
  page_end        - maximum page number (inclusive)
  source_filename - exact filename match

All filters are optional. Passing filters={} or filters=None returns
all candidates unchanged.

Case sensitivity: string comparisons are case-insensitive.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.retrieval.models import RetrievalResult


class MetadataFilter:
    """
    Apply structured metadata constraints to a list of RetrievalResults.

    This is intentionally simple and explicit — no query parsing,
    no LLM inference, no automatic filter detection. The caller specifies
    exactly what filters to apply.

    Usage:
        f = MetadataFilter()
        filtered = f.apply(
            candidates,
            filters={"content_type": "table", "section": "financial"}
        )
    """

    SUPPORTED_FILTERS = {
        "section",
        "content_type",
        "page_start",
        "page_end",
        "source_filename",
    }

    def apply(
        self,
        candidates: List[RetrievalResult],
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Receives : list of RetrievalResult (from fusion or retrieval)
                   filters dict — field -> value constraints
        Returns  : subset of candidates passing ALL filter conditions

        If filters is None or {}, returns candidates unchanged.

        FILTER SEMANTICS:
          - String fields: case-insensitive substring match
            e.g., section="risk" matches "Risk Factors", "Key Risks"
          - Numeric fields (page_start, page_end): range constraint
            e.g., page_start=2 means chunk.page_start >= 2
               page_end=4 means chunk.page_end <= 4
          - content_type: exact match (case-insensitive)

        WHY SUBSTRING for section?
        Section names vary ("Risk Factors", "Key Risks", "Risks").
        Substring matching lets you use "risk" to match all of them.
        Exact match would require knowing the exact section name.
        """
        if not filters:
            return candidates

        # Warn about unrecognized filter keys
        unknown = set(filters.keys()) - self.SUPPORTED_FILTERS
        if unknown:
            print(
                f"[MetadataFilter] WARNING: unknown filter keys: {unknown}\n"
                f"  Supported: {self.SUPPORTED_FILTERS}"
            )

        t0 = time.perf_counter()
        result = [c for c in candidates if self._passes(c, filters)]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        n_removed = len(candidates) - len(result)
        if n_removed > 0:
            print(
                f"[MetadataFilter] {n_removed}/{len(candidates)} chunks removed "
                f"by filters {filters} in {elapsed_ms:.1f}ms"
            )

        return result

    def _passes(self, result: RetrievalResult, filters: Dict[str, Any]) -> bool:
        """Return True if the result satisfies ALL filter conditions."""
        for key, value in filters.items():

            if key == "section":
                # Case-insensitive substring match on section
                if value.lower() not in result.section.lower():
                    # Also check full section_path
                    path_str = " ".join(result.section_path).lower()
                    if value.lower() not in path_str:
                        return False

            elif key == "content_type":
                if result.content_type.lower() != str(value).lower():
                    return False

            elif key == "page_start":
                # Filter: only chunks starting at or after this page
                if result.page_start < int(value):
                    return False

            elif key == "page_end":
                # Filter: only chunks ending at or before this page
                if result.page_end > int(value):
                    return False

            elif key == "source_filename":
                if str(value).lower() not in result.source.lower():
                    return False

        return True

    def explain(
        self,
        candidates: List[RetrievalResult],
        filters: Dict[str, Any],
    ) -> str:
        """
        Return a human-readable explanation of what the filter removed.
        Useful for debugging why a useful chunk didn't make it through.
        """
        if not filters:
            return "No filters applied — all candidates pass."

        lines = [f"Filter applied: {filters}", ""]
        for result in candidates:
            passes = self._passes(result, filters)
            status = "PASS" if passes else "FAIL"
            lines.append(
                f"  [{status}] chunk_id={result.chunk_id}  "
                f"section='{result.section}'  "
                f"content_type={result.content_type}  "
                f"pages={result.page_start}-{result.page_end}"
            )

        passed  = sum(1 for r in candidates if self._passes(r, filters))
        lines.append(f"\n{passed}/{len(candidates)} candidates pass the filter.")
        return "\n".join(lines)
