"""
tests/planning/test_validator.py  ─  PlanValidator tests
══════════════════════════════════════════════════════════
Tests:
  - Valid plan passes
  - Empty plan rejected
  - Exceeds max_tasks rejected
  - Duplicate task_ids rejected
  - Empty query rejected
  - Missing dependency reference rejected
  - Circular dependency detected
  - Auto-correction removes invalid depends_on
  - Auto-correction deduplicates task_ids
  - Auto-correction truncates excess tasks
  - Two invalid plans → raises after correction
"""

from __future__ import annotations

import pytest
from src.planning.models import PlanningConfig, ResearchPlan, ResearchTask
from src.planning.validator import PlanValidator, PlanValidationError


CONFIG = PlanningConfig(max_tasks=4)


def make_task(
    task_id: str,
    query: str = "OrionVault risk factors",
    description: str = "Retrieve risk information",
    depends_on: list = None,
) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        description=description,
        query=query,
        depends_on=depends_on or [],
    )


def make_plan(*tasks: ResearchTask, objective: str = "Test plan") -> ResearchPlan:
    return ResearchPlan(objective=objective, tasks=list(tasks))


@pytest.fixture
def validator():
    return PlanValidator(CONFIG)


class TestValidPlan:
    def test_single_task_passes(self, validator):
        plan = make_plan(make_task("task_1"))
        validator.validate(plan)  # should not raise

    def test_multiple_independent_tasks_pass(self, validator):
        plan = make_plan(
            make_task("task_1"),
            make_task("task_2"),
            make_task("task_3"),
        )
        validator.validate(plan)

    def test_valid_dependency_passes(self, validator):
        plan = make_plan(
            make_task("task_1"),
            make_task("task_2"),
            make_task("task_3", depends_on=["task_1", "task_2"]),
        )
        validator.validate(plan)


class TestEmptyPlan:
    def test_empty_tasks_rejected(self, validator):
        plan = make_plan()  # no tasks
        with pytest.raises(PlanValidationError) as exc_info:
            validator.validate(plan)
        assert "zero tasks" in str(exc_info.value).lower()


class TestTaskCountLimit:
    def test_exceeds_max_tasks_rejected(self, validator):
        tasks = [make_task(f"task_{i}") for i in range(1, 7)]  # 6 tasks, max=4
        plan  = make_plan(*tasks)
        with pytest.raises(PlanValidationError) as exc_info:
            validator.validate(plan)
        assert "max_tasks" in str(exc_info.value)


class TestDuplicateTaskIds:
    def test_duplicate_ids_rejected(self, validator):
        plan = make_plan(
            make_task("task_1"),
            make_task("task_1"),  # duplicate
        )
        with pytest.raises(PlanValidationError) as exc_info:
            validator.validate(plan)
        assert "duplicate" in str(exc_info.value).lower()


class TestEmptyQuery:
    def test_empty_query_rejected(self, validator):
        plan = make_plan(
            make_task("task_1", query=""),
        )
        with pytest.raises(PlanValidationError) as exc_info:
            validator.validate(plan)
        assert "empty query" in str(exc_info.value).lower()

    def test_whitespace_query_rejected(self, validator):
        plan = make_plan(
            make_task("task_1", query="   "),
        )
        with pytest.raises(PlanValidationError):
            validator.validate(plan)


class TestMissingDependency:
    def test_references_nonexistent_task(self, validator):
        plan = make_plan(
            make_task("task_1"),
            make_task("task_2", depends_on=["task_99"]),  # task_99 doesn't exist
        )
        with pytest.raises(PlanValidationError) as exc_info:
            validator.validate(plan)
        assert "task_99" in str(exc_info.value)


class TestCircularDependency:
    def test_simple_cycle_detected(self, validator):
        # task_1 → task_2 → task_1 (cycle)
        plan = make_plan(
            make_task("task_1", depends_on=["task_2"]),
            make_task("task_2", depends_on=["task_1"]),
        )
        with pytest.raises(PlanValidationError) as exc_info:
            validator.validate(plan)
        assert "circular" in str(exc_info.value).lower()

    def test_three_node_cycle_detected(self, validator):
        # task_1 → task_2 → task_3 → task_1
        plan = make_plan(
            make_task("task_1", depends_on=["task_3"]),
            make_task("task_2", depends_on=["task_1"]),
            make_task("task_3", depends_on=["task_2"]),
        )
        with pytest.raises(PlanValidationError) as exc_info:
            validator.validate(plan)
        assert "circular" in str(exc_info.value).lower()

    def test_self_referential_cycle(self, validator):
        plan = make_plan(
            make_task("task_1", depends_on=["task_1"]),  # self-loop
        )
        # After correction, self-loop should be removed
        corrected = validator.validate_and_correct(plan)
        assert corrected.tasks[0].depends_on == []


class TestAutoCorrectionDepends:
    def test_removes_invalid_dependency(self, validator):
        plan = make_plan(
            make_task("task_1"),
            make_task("task_2", depends_on=["task_nonexistent"]),
        )
        corrected = validator.validate_and_correct(plan)
        # Invalid dep should be removed
        t2 = corrected.get_task("task_2")
        assert t2 is not None
        assert "task_nonexistent" not in t2.depends_on

    def test_removes_duplicate_task_ids(self, validator):
        plan = make_plan(
            make_task("task_1"),
            make_task("task_1"),  # duplicate
            make_task("task_2"),
        )
        corrected = validator.validate_and_correct(plan)
        ids = [t.task_id for t in corrected.tasks]
        assert ids.count("task_1") == 1

    def test_truncates_excess_tasks(self, validator):
        tasks = [make_task(f"task_{i}") for i in range(1, 8)]  # 7 tasks, max=4
        plan  = make_plan(*tasks)
        corrected = validator.validate_and_correct(plan)
        assert len(corrected.tasks) <= CONFIG.max_tasks


class TestValidateAndCorrect:
    def test_still_invalid_after_correction_raises(self, validator):
        """A plan that can't be fixed must raise even after correction attempt."""
        # Empty plan cannot be corrected (no tasks to work with)
        plan = make_plan()
        with pytest.raises(PlanValidationError):
            validator.validate_and_correct(plan)

    def test_version_incremented_on_correction(self, validator):
        plan = make_plan(
            make_task("task_1"),
            make_task("task_2", depends_on=["task_nonexistent"]),
        )
        corrected = validator.validate_and_correct(plan)
        assert corrected.version > plan.version
