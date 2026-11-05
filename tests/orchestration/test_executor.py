"""
tests/orchestration/test_executor.py  ─  TaskExecutor tests
═════════════════════════════════════════════════════════════
Tests:
  - Independent tasks run (mock search_fn called once per task)
  - Dependent tasks run in order (task_3 after task_1 + task_2)
  - Task timeout → status=FAILED
  - Dependent task skipped when prerequisite fails
  - Empty search results → empty evidence list
  - Evidence deduplication (same chunk_id from 2 tasks → deduplicated)
  - run_executor synchronous wrapper
  - max_task_retrieval limit applied
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.planning.models import (
    EvidenceRecord,
    PlanningConfig,
    ResearchPlan,
    ResearchTask,
    TaskStatus,
)
from src.orchestration.executor import TaskExecutor, run_executor
from src.orchestration.context_builder import aggregate_evidence


CONFIG = PlanningConfig(
    max_tasks=6,
    max_task_retrieval=5,
    task_timeout_sec=2.0,
    max_total_evidence=20,
    allow_llm_classifier=False,
    allow_llm_planner=False,
)


def make_task(task_id, depends_on=None, top_k=5) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        description=f"Task {task_id}",
        query=f"test query for {task_id}",
        depends_on=depends_on or [],
        top_k=top_k,
    )


def make_evidence(chunk_id: str, task_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        chunk_id=chunk_id,
        task_id=task_id,
        text=f"Evidence text {chunk_id}",
        semantic_score=0.8,
    )


def make_mock_result(chunk_id: str):
    """Create a mock RetrievalResult-like object."""
    r = MagicMock()
    r.chunk_id     = chunk_id
    r.text         = f"chunk text {chunk_id}"
    r.source       = "test.pdf"
    r.metadata     = {}
    r.section      = ""
    r.section_path = []
    r.page_start   = 1
    r.page_end     = 1
    r.content_type = "text"
    r.semantic_score = 0.8
    r.rerank_score   = None
    return r


class TestIndependentTasks:
    def test_all_tasks_run(self):
        call_log = []

        async def search_fn(query, filters, top_k):
            call_log.append(query)
            return [make_mock_result(f"chunk_{query[-1]}")]

        plan = ResearchPlan(
            objective="test",
            tasks=[make_task("task_1"), make_task("task_2"), make_task("task_3")],
        )
        executor = TaskExecutor(config=CONFIG, search_fn=search_fn)
        results  = asyncio.run(executor.execute(plan))

        assert len(results) == 3
        assert len(call_log) == 3
        for task_id in ["task_1", "task_2", "task_3"]:
            assert task_id in results

    def test_all_tasks_done_status(self):
        async def search_fn(query, filters, top_k):
            return [make_mock_result("c1")]

        plan = ResearchPlan(
            objective="test",
            tasks=[make_task("task_1"), make_task("task_2")],
        )
        executor = TaskExecutor(config=CONFIG, search_fn=search_fn)
        asyncio.run(executor.execute(plan))

        for t in plan.tasks:
            assert t.status == TaskStatus.DONE


class TestDependentTasks:
    def test_dependent_task_runs_after_prerequisites(self):
        execution_order = []

        async def search_fn(query, filters, top_k):
            execution_order.append(query)
            return []

        plan = ResearchPlan(
            objective="test",
            tasks=[
                make_task("task_1"),
                make_task("task_2"),
                make_task("task_3", depends_on=["task_1", "task_2"]),
            ],
        )
        executor = TaskExecutor(config=CONFIG, search_fn=search_fn)
        asyncio.run(executor.execute(plan))

        # task_3 must be called after task_1 and task_2
        idx_1 = execution_order.index("test query for task_1")
        idx_2 = execution_order.index("test query for task_2")
        idx_3 = execution_order.index("test query for task_3")
        assert idx_3 > idx_1 and idx_3 > idx_2

    def test_dependent_task_skipped_when_dep_fails(self):
        async def search_fn(query, filters, top_k):
            if "task_1" in query:
                raise RuntimeError("Simulated retrieval failure")
            return [make_mock_result("c1")]

        plan = ResearchPlan(
            objective="test",
            tasks=[
                make_task("task_1"),
                make_task("task_2", depends_on=["task_1"]),
            ],
        )
        executor = TaskExecutor(config=CONFIG, search_fn=search_fn)
        results  = asyncio.run(executor.execute(plan))

        assert results["task_1"] == []  # failed
        assert results["task_2"] == []  # skipped

        task_statuses = {t.task_id: t.status for t in plan.tasks}
        assert task_statuses["task_1"] == TaskStatus.FAILED
        assert task_statuses["task_2"] == TaskStatus.SKIPPED


class TestTimeout:
    def test_timeout_marks_task_failed(self):
        import asyncio as asyncio_

        async def slow_search(query, filters, top_k):
            await asyncio_.sleep(10.0)  # longer than timeout
            return []

        config = PlanningConfig(task_timeout_sec=0.1, allow_llm_classifier=False)
        plan   = ResearchPlan(
            objective="test",
            tasks=[make_task("task_1")],
        )
        executor = TaskExecutor(config=config, search_fn=slow_search)
        results  = asyncio.run(executor.execute(plan))

        assert results["task_1"] == []
        assert plan.tasks[0].status == TaskStatus.FAILED


class TestEmptyResults:
    def test_empty_search_returns_empty_evidence(self):
        async def search_fn(query, filters, top_k):
            return []

        plan = ResearchPlan(
            objective="test",
            tasks=[make_task("task_1")],
        )
        executor = TaskExecutor(config=CONFIG, search_fn=search_fn)
        results  = asyncio.run(executor.execute(plan))
        assert results["task_1"] == []


class TestRunExecutorSync:
    def test_sync_wrapper(self):
        async def search_fn(query, filters, top_k):
            return [make_mock_result("chunk_001")]

        plan = ResearchPlan(
            objective="test",
            tasks=[make_task("task_1")],
        )
        results = run_executor(plan, config=CONFIG, search_fn=search_fn)
        assert "task_1" in results
        assert len(results["task_1"]) == 1


class TestEvidenceAggregation:
    def test_deduplication(self):
        """Same chunk_id from two tasks → only one record in flat list."""
        task_results = {
            "task_1": [make_evidence("chunk_A", "task_1"), make_evidence("chunk_B", "task_1")],
            "task_2": [make_evidence("chunk_A", "task_2"), make_evidence("chunk_C", "task_2")],
        }
        plan = ResearchPlan(
            objective="test",
            tasks=[make_task("task_1"), make_task("task_2")],
        )
        flat, by_task = aggregate_evidence(task_results, plan, CONFIG)
        ids = [r.chunk_id for r in flat]
        assert ids.count("chunk_A") == 1, "chunk_A should appear only once (deduplicated)"
        assert len(flat) == 3  # A, B, C

    def test_evidence_grouped_by_task(self):
        task_results = {
            "task_1": [make_evidence("c1", "task_1"), make_evidence("c2", "task_1")],
            "task_2": [make_evidence("c3", "task_2")],
        }
        plan = ResearchPlan(
            objective="test",
            tasks=[make_task("task_1"), make_task("task_2")],
        )
        _, by_task = aggregate_evidence(task_results, plan, CONFIG)
        assert len(by_task.get("task_1", [])) == 2
        assert len(by_task.get("task_2", [])) == 1

    def test_total_evidence_capped(self):
        config = PlanningConfig(max_total_evidence=2, allow_llm_classifier=False)
        task_results = {
            "task_1": [make_evidence(f"c{i}", "task_1") for i in range(5)],
        }
        plan = ResearchPlan(
            objective="test",
            tasks=[make_task("task_1")],
        )
        flat, _ = aggregate_evidence(task_results, plan, config)
        assert len(flat) <= 2

    def test_sorted_by_score_descending(self):
        e1 = make_evidence("c1", "task_1"); e1.semantic_score = 0.9
        e2 = make_evidence("c2", "task_1"); e2.semantic_score = 0.5
        e3 = make_evidence("c3", "task_1"); e3.semantic_score = 0.7

        task_results = {"task_1": [e1, e2, e3]}
        plan = ResearchPlan(objective="test", tasks=[make_task("task_1")])
        flat, _ = aggregate_evidence(task_results, plan, CONFIG)
        scores = [r.best_score() for r in flat]
        assert scores == sorted(scores, reverse=True)
