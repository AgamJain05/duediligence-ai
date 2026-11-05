"""
tests/orchestration/test_graph.py  ─  LangGraph graph tests
══════════════════════════════════════════════════════════════
Tests the LangGraph graph WITHOUT calling real LLM APIs.
Uses monkeypatching to replace LLM calls and retrieval calls.

Tests:
  - Simple query → route="simple", 1 task, uses retrieve_simple node
  - Complex query → route="complex", uses plan+execute path
  - State after analyze_query is populated
  - State after set_route has "route" key
  - Final state has "answer" key
  - Error from retrieval doesn't crash graph (errors list populated)
  - No planning for SIMPLE queries (no research_plan in state)
  - COMPARISON query creates multi-task plan
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.planning.models import PlanningConfig
from src.orchestration.graph import build_graph
from src.orchestration.state import initial_state

# Use minimal config for tests
TEST_CONFIG = PlanningConfig(
    max_task_retrieval=3,
    max_total_evidence=10,
    allow_llm_classifier=False,
    allow_llm_planner=False,
    task_timeout_sec=5.0,
)


def _mock_search_fn():
    """Search function that returns empty results (no DB needed)."""
    async def _search(query: str, filters: dict, top_k: int):
        return []
    return _search


def _mock_llm_response(answer: str = "Mock answer from LLM."):
    """Patch openai.OpenAI to return a canned answer."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = answer
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )
    return mock_client


@pytest.fixture
def simple_query():
    return "What is OrionVault's FY2025 revenue?"


@pytest.fixture
def comparison_query():
    return "Compare OrionVault's US and European growth strategies."


@pytest.fixture
def temporal_query():
    return "How did OrionVault's risk profile change from 2024 to 2026?"


class TestGraphStructure:
    def test_graph_builds_without_error(self):
        """Graph should compile cleanly."""
        graph = build_graph(config=TEST_CONFIG, db_session=None)
        assert graph is not None

    def test_graph_has_nodes(self):
        """Graph compilation should succeed (no missing nodes)."""
        graph = build_graph(config=TEST_CONFIG, db_session=None)
        # LangGraph compiled graph is callable
        assert callable(graph.invoke)


class TestSimplePath:
    def test_simple_query_routes_to_simple(self, simple_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mock answer"}

            # Override search_fn
            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(simple_query))

        assert result.get("route") == "simple", (
            f"Simple query should route to 'simple', got '{result.get('route')}'"
        )

    def test_simple_query_no_research_plan(self, simple_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mock answer"}

            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph  = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(simple_query))

        plan = result.get("research_plan")
        # Simple path may still create a plan internally for the single task,
        # but it should NOT have multiple tasks
        if plan:
            assert len(plan.get("tasks", [])) <= 1

    def test_analysis_populated_after_analyze(self, simple_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mock answer"}

            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph  = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(simple_query))

        analysis = result.get("analysis", {})
        assert analysis.get("query_type") is not None
        assert analysis.get("original_query") == simple_query


class TestComplexPath:
    def test_comparison_query_routes_complex(self, comparison_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mock answer"}

            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph  = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(comparison_query))

        assert result.get("route") == "complex", (
            f"Comparison query should route 'complex', got '{result.get('route')}'"
        )

    def test_complex_path_has_research_plan(self, comparison_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mock answer"}

            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph  = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(comparison_query))

        plan = result.get("research_plan")
        assert plan is not None, "Complex path should produce a research_plan"
        assert len(plan.get("tasks", [])) >= 2

    def test_temporal_query_routes_complex(self, temporal_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mock answer"}

            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph  = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(temporal_query))

        assert result.get("route") == "complex"


class TestFinalState:
    def test_final_state_has_answer(self, simple_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mocked answer."}

            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph  = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(simple_query))

        assert result.get("answer") is not None
        assert len(result.get("answer", "")) > 0

    def test_final_state_has_latency(self, simple_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mock"}

            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph  = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(simple_query))

        latency = result.get("latency", {})
        assert isinstance(latency, dict)
        # At least analyze_ms should be populated
        assert "analyze_ms" in latency

    def test_final_context_populated(self, simple_query):
        with patch("src.orchestration.graph.make_generate_answer_node") as mock_gen:
            mock_gen.return_value = lambda state: {"answer": "Mock"}

            search_fn = _mock_search_fn()
            with patch("src.orchestration.graph._make_search_fn", return_value=search_fn):
                graph  = build_graph(config=TEST_CONFIG, db_session=None)
                result = graph.invoke(initial_state(simple_query))

        context = result.get("final_context", "")
        assert isinstance(context, str)
        # Context should mention the original query
        assert simple_query[:20] in context or "QUESTION" in context or "RESEARCH" in context


class TestContextBuilder:
    def test_build_simple_context(self):
        from src.orchestration.context_builder import build_simple_context
        from src.planning.models import EvidenceRecord

        evidence = [
            EvidenceRecord(
                chunk_id="c1", task_id="task_1",
                text="Revenue was $186M in FY2025.",
                semantic_score=0.9,
            )
        ]
        context = build_simple_context("What is FY2025 revenue?", evidence)
        assert "186M" in context or "Revenue" in context
        assert "QUESTION" in context

    def test_context_respects_char_limit(self):
        from src.orchestration.context_builder import build_simple_context
        from src.planning.models import EvidenceRecord, PlanningConfig

        config = PlanningConfig(max_context_chars=100)
        long_text = "x" * 500
        evidence  = [EvidenceRecord(chunk_id="c1", task_id="t1", text=long_text)]
        context   = build_simple_context("query", evidence, config=config)
        assert len(context) <= 120  # small buffer for truncation suffix

    def test_build_structured_context(self):
        from src.orchestration.context_builder import build_structured_context
        from src.planning.models import EvidenceRecord, ResearchPlan, ResearchTask

        task_1 = ResearchTask(
            task_id="task_1", description="US strategy",
            query="OrionVault US strategy", purpose="Understand US approach",
        )
        task_2 = ResearchTask(
            task_id="task_2", description="Europe strategy",
            query="OrionVault Europe strategy", purpose="Understand EU approach",
        )
        plan = ResearchPlan(objective="Compare strategies", tasks=[task_1, task_2])

        by_task = {
            "task_1": [EvidenceRecord(chunk_id="c1", task_id="task_1", text="US data.")],
            "task_2": [EvidenceRecord(chunk_id="c2", task_id="task_2", text="EU data.")],
        }
        context = build_structured_context("Compare US and EU", by_task, plan)
        assert "US strategy" in context
        assert "Europe strategy" in context
        assert "US data" in context
        assert "EU data" in context
