"""
src/orchestration/executor.py  ─  Async Parallel Task Executor
════════════════════════════════════════════════════════════════
Phase 5: Executes research tasks, respecting dependencies,
running independent tasks in parallel with asyncio.

PARALLEL EXECUTION STRATEGY
────────────────────────────
A ResearchPlan has a dependency DAG.

Independent tasks (depends_on=[]) → run concurrently with asyncio.gather()
Dependent tasks (depends_on=[...]) → wait for prerequisites, then run

Algorithm:
  1. Identify all tasks with no unmet dependencies → "ready batch"
  2. Run ready batch concurrently
  3. After each batch completes, update the "completed" set
  4. Repeat until no more tasks remain

This is equivalent to topological-sort-based level execution.

Example:
  task_1 (independent) ──┐
  task_2 (independent) ──┼──→ [run in parallel] → task_3 (depends on 1, 2)
  Sequence: [batch_1: task_1, task_2] → [batch_2: task_3]

WHY ASYNCIO?
  retrieval calls are I/O-bound (database/FAISS).
  asyncio.gather() overlaps waiting, reducing total latency.
  No threads, no distributed workers — just Python async/await.

TIMEOUT:
  Each task has config.task_timeout_sec.
  On timeout → task marked FAILED, execution continues with other tasks.
  A failed prerequisite causes its dependents to be SKIPPED.

RETRIEVAL INTEGRATION:
  Tasks call search_chunks() which wraps the Phase 4 PostgresRetriever.
  The executor is retrieval-agnostic: it calls a callable passed in at
  construction time. This keeps the executor independently testable.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from src.planning.models import (
    EvidenceRecord,
    PlanningConfig,
    ResearchPlan,
    ResearchTask,
    TaskStatus,
)


# ── Retrieval adapter ─────────────────────────────────────────────────────────

def make_pg_search_fn(db_session, retrieval_mode: str = "POSTGRES_HNSW"):
    """
    Create a search function backed by Phase 4 PostgresRetriever.

    Returns an async callable:
        search_fn(query, filters, top_k) → List[RetrievalResult]
    """
    from src.db.repositories.retrieval import PostgresRetriever

    retriever = PostgresRetriever(db_session)

    async def _search(query: str, filters: Dict[str, Any], top_k: int):
        loop = asyncio.get_event_loop()
        # Run the synchronous DB call in a thread pool to avoid blocking
        result = await loop.run_in_executor(
            None,
            lambda: retriever.search(query, top_k=top_k, filters=filters, mode=retrieval_mode)
        )
        return result

    return _search


def make_faiss_search_fn(top_k: int = 8):
    """
    Fallback search function backed by Phase 3 FAISS hybrid retriever.
    Used when PostgreSQL is not available.
    """
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.models import HYBRID_RERANKED

    retriever = HybridRetriever(mode=HYBRID_RERANKED, final_top_k=top_k)

    async def _search(query: str, filters: Dict[str, Any], top_k_: int):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: retriever.search(query).results
        )
        return result[:top_k_]

    return _search


# ── Evidence helpers ──────────────────────────────────────────────────────────

def _result_to_evidence(r, task_id: str) -> EvidenceRecord:
    """
    Convert a RetrievalResult (Phase 3 or 4) into an EvidenceRecord.

    Handles both Phase 3 (FAISS RetrievalResult) and Phase 4 (PG RetrievalResult)
    by reading fields defensively with .get() on metadata.
    """
    meta = getattr(r, "metadata", {}) or {}
    return EvidenceRecord(
        chunk_id       = str(r.chunk_id),
        task_id        = task_id,
        text           = r.text,
        source         = getattr(r, "source", meta.get("source_filename", "")),
        document       = meta.get("document", meta.get("source_filename", "")),
        document_type  = meta.get("document_type", ""),
        version_label  = meta.get("version_label", ""),
        effective_date = str(meta.get("effective_date", "") or ""),
        is_current     = bool(meta.get("is_current", False)),
        section        = getattr(r, "section", meta.get("section", "")),
        section_path   = getattr(r, "section_path", []),
        page_start     = getattr(r, "page_start", meta.get("page_start", 0)) or 0,
        page_end       = getattr(r, "page_end", meta.get("page_end", 0)) or 0,
        content_type   = getattr(r, "content_type", meta.get("content_type", "text")),
        semantic_score = getattr(r, "semantic_score", None),
        rerank_score   = getattr(r, "rerank_score", None),
    )


# ── Task Executor ─────────────────────────────────────────────────────────────

class TaskExecutor:
    """
    Executes a ResearchPlan's tasks with asyncio parallelism and dependency tracking.

    Usage:
        executor = TaskExecutor(config, search_fn)
        results = await executor.execute(plan)

    Returns:
        Dict[task_id, List[EvidenceRecord]]
    """

    def __init__(
        self,
        config: Optional[PlanningConfig] = None,
        search_fn: Optional[Callable] = None,
    ) -> None:
        self.config    = config or PlanningConfig.default()
        self.search_fn = search_fn  # async callable(query, filters, top_k)
        self._timings: Dict[str, float] = {}

    async def execute(self, plan: ResearchPlan) -> Dict[str, List[EvidenceRecord]]:
        """
        Execute all tasks in the plan, respecting dependencies.

        Returns a mapping of task_id → List[EvidenceRecord].
        Failed tasks return empty lists; their dependents are skipped.
        """
        results: Dict[str, List[EvidenceRecord]] = {}
        completed_ids: Set[str]                  = set()
        failed_ids:    Set[str]                  = set()
        remaining      = list(plan.tasks)

        while remaining:
            # Find tasks ready to execute this round
            ready = [
                t for t in remaining
                if t.is_ready(completed_ids) and
                not any(dep in failed_ids for dep in t.depends_on)
            ]
            # Find tasks that should be skipped (dependency failed)
            skippable = [
                t for t in remaining
                if any(dep in failed_ids for dep in t.depends_on)
            ]

            for t in skippable:
                t.status = TaskStatus.SKIPPED
                results[t.task_id] = []
                failed_ids.add(t.task_id)
                remaining.remove(t)

            if not ready:
                # Guard: if remaining tasks exist but nothing is ready → dependency stall
                for t in remaining:
                    t.status = TaskStatus.FAILED
                    t.error  = "Dependency stall: could not determine execution order"
                    results[t.task_id] = []
                break

            # Run all ready tasks concurrently
            batch_results = await asyncio.gather(
                *[self._execute_task(t) for t in ready],
                return_exceptions=True,
            )

            for task, result in zip(ready, batch_results):
                remaining.remove(task)
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.error  = str(result)
                    results[task.task_id] = []
                    failed_ids.add(task.task_id)
                else:
                    task.status       = TaskStatus.DONE
                    task.result_count = len(result)
                    results[task.task_id] = result
                    completed_ids.add(task.task_id)

        return results

    async def _execute_task(self, task: ResearchTask) -> List[EvidenceRecord]:
        """Run a single task with timeout."""
        task.status = TaskStatus.RUNNING
        t0 = time.perf_counter()

        try:
            raw_results = await asyncio.wait_for(
                self._call_search(task),
                timeout=self.config.task_timeout_sec,
            )
        except asyncio.TimeoutError:
            task.latency_ms = (time.perf_counter() - t0) * 1000
            raise TimeoutError(
                f"Task '{task.task_id}' timed out after {self.config.task_timeout_sec}s"
            )

        task.latency_ms = (time.perf_counter() - t0) * 1000

        # Convert to EvidenceRecord
        evidence = [_result_to_evidence(r, task.task_id) for r in raw_results]
        return evidence[:self.config.max_task_retrieval]

    async def _call_search(self, task: ResearchTask):
        """Dispatch to search_fn or a no-op stub if None."""
        if self.search_fn is None:
            return []
        return await self.search_fn(task.query, task.filters, task.top_k)

    @property
    def timings(self) -> Dict[str, float]:
        return self._timings


def run_executor(
    plan: ResearchPlan,
    config: Optional[PlanningConfig] = None,
    search_fn: Optional[Callable] = None,
) -> Dict[str, List[EvidenceRecord]]:
    """
    Synchronous wrapper for TaskExecutor.execute().
    Used by LangGraph nodes (which are sync functions).
    """
    executor = TaskExecutor(config=config, search_fn=search_fn)
    return asyncio.run(executor.execute(plan))
