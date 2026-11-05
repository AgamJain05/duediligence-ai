"""
src/planning/validator.py  ─  Plan Validation
═══════════════════════════════════════════════
Phase 5: Validates a ResearchPlan before execution.

WHY VALIDATE?
──────────────
The LLM planner can produce malformed plans:
  - task_ids with typos → dependency reference won't resolve
  - self-referential depends_on → circular loop
  - duplicate task_ids → ambiguous execution
  - empty queries → retrieval will fail silently
  - too many tasks → cost/latency explosion

Catching these before execution gives a clear error message
rather than a cryptic runtime failure.

VALIDATION STEPS (in order):
  1. Non-empty task list
  2. Task count within limit
  3. task_id uniqueness
  4. Every task has a non-empty query
  5. Every task has a non-empty description
  6. All depends_on references exist in the plan
  7. No circular dependencies (DFS cycle detection)

CORRECTION PASS:
  If validation fails, PlanValidator.validate_and_correct() tries one
  automatic correction (removing invalid depends_on references, dedup IDs).
  If the corrected plan still fails, it raises PlanValidationError.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set


class PlanValidationError(Exception):
    """Raised when a plan is structurally invalid and cannot be corrected."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(f"Plan validation failed:\n" + "\n".join(f"  • {e}" for e in errors))


class PlanValidator:
    """
    Validates and optionally corrects a ResearchPlan.

    Usage:
        validator = PlanValidator(config)
        validator.validate(plan)       # raises PlanValidationError if invalid
        clean = validator.validate_and_correct(plan)  # attempts auto-fix
    """

    def __init__(self, config=None) -> None:
        from src.planning.models import PlanningConfig
        self.config = config or PlanningConfig.default()

    def validate(self, plan) -> None:
        """
        Validate the plan. Raises PlanValidationError listing all problems.
        Does NOT modify the plan.
        """
        errors = self._collect_errors(plan)
        if errors:
            raise PlanValidationError(errors)

    def validate_and_correct(self, plan):
        """
        Try to validate. On failure, attempt one correction pass.
        If still invalid, raise PlanValidationError.

        Returns the (possibly corrected) plan.
        """
        errors = self._collect_errors(plan)
        if not errors:
            return plan

        corrected = self._correct(plan)
        errors_after = self._collect_errors(corrected)
        if errors_after:
            raise PlanValidationError(
                [f"[original] {e}" for e in errors] +
                [f"[after correction] {e}" for e in errors_after]
            )
        corrected.notes += " [auto-corrected]"
        return corrected

    # ── Private methods ────────────────────────────────────────────────────────

    def _collect_errors(self, plan) -> List[str]:
        errors: List[str] = []
        tasks = plan.tasks

        # 1. Non-empty
        if not tasks:
            errors.append("Plan has zero tasks. A plan must have at least one task.")
            return errors  # can't run further checks

        # 2. Task count
        if len(tasks) > self.config.max_tasks:
            errors.append(
                f"Plan has {len(tasks)} tasks, exceeding max_tasks={self.config.max_tasks}."
            )

        # 3. task_id uniqueness
        seen_ids: Set[str] = set()
        duplicate_ids: List[str] = []
        for t in tasks:
            if t.task_id in seen_ids:
                duplicate_ids.append(t.task_id)
            seen_ids.add(t.task_id)
        if duplicate_ids:
            errors.append(f"Duplicate task_ids: {duplicate_ids}")

        # 4. Non-empty query
        for t in tasks:
            if not t.query or not t.query.strip():
                errors.append(f"Task '{t.task_id}' has an empty query string.")

        # 5. Non-empty description
        for t in tasks:
            if not t.description or not t.description.strip():
                errors.append(f"Task '{t.task_id}' has an empty description.")

        # 6. Dependency references
        for t in tasks:
            for dep in t.depends_on:
                if dep not in seen_ids:
                    errors.append(
                        f"Task '{t.task_id}' depends on '{dep}', which does not exist in the plan."
                    )

        # 7. Circular dependencies
        cycle = self._detect_cycle(tasks)
        if cycle:
            errors.append(f"Circular dependency detected: {' → '.join(cycle)}")

        return errors

    def _detect_cycle(self, tasks) -> Optional[List[str]]:
        """
        DFS-based cycle detection.
        Returns the cycle path if found, None if no cycle.
        """
        graph: Dict[str, List[str]] = {t.task_id: t.depends_on for t in tasks}
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    result = dfs(neighbor, path + [neighbor])
                    if result:
                        return result
                elif neighbor in rec_stack:
                    return path + [neighbor]
            rec_stack.discard(node)
            return None

        for task_id in graph:
            if task_id not in visited:
                result = dfs(task_id, [task_id])
                if result:
                    return result
        return None

    def _correct(self, plan):
        """
        Attempt automatic correction of common issues:
          - Remove duplicate task_ids (keep first occurrence)
          - Remove invalid depends_on references
          - Truncate tasks exceeding max_tasks
          - Fill empty descriptions
        """
        import copy
        from src.planning.models import ResearchPlan

        tasks = list(plan.tasks)

        # Remove duplicates (keep first)
        seen: Set[str] = set()
        unique_tasks = []
        for t in tasks:
            if t.task_id not in seen:
                seen.add(t.task_id)
                unique_tasks.append(t)
        tasks = unique_tasks

        # Truncate to max_tasks
        tasks = tasks[:self.config.max_tasks]
        valid_ids = {t.task_id for t in tasks}

        # Remove invalid depends_on references
        corrected_tasks = []
        for t in tasks:
            t_copy = t.model_copy()
            t_copy.depends_on = [d for d in t.depends_on if d in valid_ids and d != t.task_id]
            if not t_copy.description:
                t_copy.description = f"Retrieve information for {t_copy.task_id}"
            corrected_tasks.append(t_copy)

        return ResearchPlan(
            plan_id   = plan.plan_id,
            query_id  = plan.query_id,
            objective = plan.objective or "Research plan",
            tasks     = corrected_tasks,
            version   = plan.version + 1,
            notes     = plan.notes,
        )
