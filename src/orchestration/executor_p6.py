"""
src/orchestration/executor_p6.py — Phase 6 tool-aware task executor.

Extends the Phase 5 executor. Dispatches ToolTask objects to the tool registry.
Independent tasks run in parallel (asyncio). Dependency order is respected.
Skips tasks when upstream dependencies fail.

Phase 5 executor is NOT modified.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from src.planning.models_p6 import ToolTask, ToolAwareResearchPlan, ToolTaskType
from src.planning.models import TaskStatus
from src.tools.base import ToolResult
from src.tools.registry import get_tool

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_QUERY = 10
MAX_TASKS = 6


class ToolTaskExecutor:
    """
    Executes a ToolAwareResearchPlan respecting task dependencies.

    Parallel execution: tasks with no pending dependencies run concurrently.
    Sequential: tasks with dependencies wait for upstream completion.
    Partial success: a failed task does not stop the plan; downstream tasks
    that depend on it are marked SKIPPED_DEPENDENCY_FAILED.
    """

    def __init__(self, timeout_per_task: float = 30.0):
        self.timeout = timeout_per_task
        self._total_calls = 0

    async def execute(self, plan: ToolAwareResearchPlan) -> Dict[str, ToolResult]:
        """
        Execute all tasks and return {task_id: ToolResult}.
        """
        results: Dict[str, ToolResult] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()

        tasks_by_id = {t.task_id: t for t in plan.tasks}
        remaining = list(plan.tasks)

        while remaining:
            # Find tasks that are ready
            ready = [
                t for t in remaining
                if t.is_ready(completed)
                and not any(dep in failed for dep in t.depends_on)
            ]

            # Tasks whose dependency failed
            skippable = [
                t for t in remaining
                if any(dep in failed for dep in t.depends_on)
            ]
            for t in skippable:
                remaining.remove(t)
                t.status = TaskStatus.FAILED
                t.error = f"SKIPPED_DEPENDENCY_FAILED: {[d for d in t.depends_on if d in failed]}"
                results[t.task_id] = ToolResult(
                    tool=t.tool,
                    status="ERROR",
                    error={"code": "SKIPPED_DEPENDENCY_FAILED", "message": t.error},
                )
                failed.add(t.task_id)
                logger.warning(f"Task {t.task_id} skipped — upstream failure")

            if not ready:
                if remaining:
                    logger.error(f"Deadlock detected in plan. Remaining: {[t.task_id for t in remaining]}")
                break

            # Run ready tasks in parallel
            async_tasks = [
                self._run_one(t, results)
                for t in ready
                if self._total_calls < MAX_TOOL_CALLS_PER_QUERY
            ]

            if async_tasks:
                done = await asyncio.gather(*async_tasks, return_exceptions=True)
                for t, outcome in zip(ready, done):
                    remaining.remove(t)
                    if isinstance(outcome, Exception):
                        t.status = TaskStatus.FAILED
                        t.error = str(outcome)
                        failed.add(t.task_id)
                        results[t.task_id] = ToolResult(
                            tool=t.tool, status="ERROR",
                            error={"code": "EXECUTOR_ERROR", "message": str(outcome)},
                        )
                    else:
                        result: ToolResult = outcome
                        results[t.task_id] = result
                        t.tool_result = result.model_dump_json_safe()
                        t.result_count = len(result.data.get("rows", result.data.get("chunks", [])))
                        if result.status in ("SUCCESS", "PARTIAL", "CONFLICT", "EMPTY"):
                            t.status = TaskStatus.DONE
                            completed.add(t.task_id)
                        else:
                            t.status = TaskStatus.FAILED
                            t.error = result.error.message if result.error else "Unknown error"
                            failed.add(t.task_id)
            else:
                # Rate limit hit
                for t in ready:
                    remaining.remove(t)
                    results[t.task_id] = ToolResult(
                        tool=t.tool, status="ERROR",
                        error={"code": "RATE_LIMIT", "message": f"Max tool calls ({MAX_TOOL_CALLS_PER_QUERY}) reached"},
                    )
                    failed.add(t.task_id)

        return results

    async def _run_one(self, task: ToolTask, existing_results: Dict[str, ToolResult]) -> ToolResult:
        """Execute a single task asynchronously with timeout."""
        self._total_calls += 1
        task.status = TaskStatus.RUNNING
        t0 = time.perf_counter()

        tool = get_tool(task.tool)
        if tool is None:
            task.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return ToolResult(
                tool=task.tool, status="ERROR",
                error={"code": "UNKNOWN_TOOL", "message": f"Tool '{task.tool}' not in registry."},
            )

        # Resolve input bindings from upstream results
        resolved_args = dict(task.tool_args)
        for arg_name, binding in task.input_bindings.items():
            try:
                resolved_args[arg_name] = _resolve_binding(binding, existing_results)
            except KeyError as e:
                return ToolResult(
                    tool=task.tool, status="ERROR",
                    error={"code": "BINDING_ERROR", "message": f"Cannot resolve binding '{binding}': {e}"},
                )

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, lambda: tool.safe_run(**resolved_args)),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                tool=task.tool, status="ERROR",
                error={"code": "TIMEOUT", "message": f"Task exceeded {self.timeout}s timeout."},
            )

        task.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result


def _resolve_binding(binding_path: str, results: Dict[str, ToolResult]) -> Any:
    """
    Resolve a binding like "task_1.data.rows.0.value" from existing ToolResults.
    Supports dot-notation path traversal.
    """
    parts = binding_path.split(".")
    task_id = parts[0]
    if task_id not in results:
        raise KeyError(f"Task '{task_id}' not yet completed")
    obj: Any = results[task_id].model_dump()
    for part in parts[1:]:
        if isinstance(obj, dict):
            obj = obj[part]
        elif isinstance(obj, list):
            obj = obj[int(part)]
        else:
            raise KeyError(f"Cannot traverse '{part}' on {type(obj)}")
    return obj
