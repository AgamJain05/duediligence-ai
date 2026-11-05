"""
src/planning/planner.py  ─  Research Planner
══════════════════════════════════════════════
Phase 5: Transforms a QueryAnalysis into a structured ResearchPlan.

PLANNER RESPONSIBILITY
───────────────────────
The planner decides WHAT information is needed and WHY.
The retrieval layer decides HOW to retrieve it.

This separation is fundamental:
  - The planner thinks about information needs
  - The retriever thinks about vector spaces and SQL filters
  - Neither crosses the other's boundary

IMPLEMENTATION STRATEGY
────────────────────────
Two modes:

1. RULE-BASED (fast, deterministic)
   Templates for each query type produce plans without LLM calls.
   Used for clear-cut COMPARISON, TEMPORAL, MULTI_TOPIC queries.

2. LLM-BASED (flexible, handles novel phrasings)
   Sends the QueryAnalysis to the LLM and asks for a structured plan.
   The LLM generates task descriptions, sub-queries, and rationale.
   Output is parsed into ResearchPlan via Pydantic.

RULE VS LLM CHOICE:
   Rule-based covers the well-defined templates.
   LLM-based handles MULTI_HOP and CROSS_DOCUMENT where the dependency
   structure is hard to pre-define deterministically.

SUB-QUERY REWRITING
────────────────────
Sub-queries are written for retrieval engines, NOT for users.
They are dense keyword str ings that maximize retrieval recall:

  User: "Compare US and European strategies."
  Sub-queries:
    → "OrionVault United States enterprise sales expansion strategy FY2026"
    → "OrionVault Europe channel partnership growth strategy FY2026"

The planner handles this internally. The decomposer module provides
additional rewriting utilities.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from src.planning.models import (
    PlanningConfig,
    QueryAnalysis,
    QueryType,
    ResearchPlan,
    ResearchTask,
)


# ── Rule-based plan templates ─────────────────────────────────────────────────

def _plan_simple(analysis: QueryAnalysis, config: PlanningConfig) -> ResearchPlan:
    """Single retrieval task — the SIMPLE path."""
    return ResearchPlan(
        query_id=id(analysis.original_query).__str__(),
        objective=f"Answer: {analysis.original_query}",
        tasks=[
            ResearchTask(
                task_id="task_1",
                description="Retrieve relevant information to answer the query.",
                query=analysis.original_query,
                purpose="Direct answer retrieval for simple factual query.",
                depends_on=[],
                top_k=config.max_task_retrieval,
            )
        ],
    )


def _plan_comparison(analysis: QueryAnalysis, config: PlanningConfig) -> ResearchPlan:
    """
    COMPARISON plan template.

    Structure:
      task_1: Retrieve side A
      task_2: Retrieve side B
      task_3: Retrieve risks / challenges for A
      task_4: Retrieve risks / challenges for B

    Tasks 1 and 2 are independent (parallel).
    Tasks 3 and 4 are independent (parallel).
    Tasks 3, 4 depend on no specific tasks — they are parallel with 1, 2.
    """
    q = analysis.original_query
    topics = analysis.topics or ["strategy"]

    # Normalize entities: strip possessives, prefer geographic terms
    geo_terms  = ["US", "United States", "North America", "Europe", "EMEA", "Germany",
                  "UK", "Asia", "Pacific", "Global"]
    prod_terms = ["Sentinel", "VaultGuard", "VaultSync", "Copilot"]

    # Try to find geographic or product entities as comparison targets first
    def _clean(e: str) -> str:
        return e.rstrip("'s").rstrip("'").strip()

    cleaned_entities = [_clean(e) for e in analysis.entities]
    geo_found  = [e for e in cleaned_entities if any(g.lower() in e.lower() for g in geo_terms)]
    prod_found = [e for e in cleaned_entities if any(p.lower() in e.lower() for p in prod_terms)]
    other      = [e for e in cleaned_entities if e not in geo_found and e not in prod_found
                  and e.lower() != "orionvault"]

    # Build side_a, side_b prioritising geographic then product then others
    candidates = (geo_found + prod_found + other + cleaned_entities)
    # deduplicate preserving order
    seen_c: set = set()
    unique_candidates = []
    for c in candidates:
        if c.lower() not in seen_c:
            seen_c.add(c.lower())
            unique_candidates.append(c)

    side_a = unique_candidates[0] if len(unique_candidates) >= 1 else "first subject"
    side_b = unique_candidates[1] if len(unique_candidates) >= 2 else "second subject"
    topic  = topics[0] if topics else "approach"

    tasks = [
        ResearchTask(
            task_id="task_1",
            description=f"Retrieve {side_a} {topic}",
            query=f"OrionVault {side_a} {topic} strategy expansion",
            purpose=f"Understand {side_a} side of the comparison",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_2",
            description=f"Retrieve {side_b} {topic}",
            query=f"OrionVault {side_b} {topic} strategy expansion",
            purpose=f"Understand {side_b} side of the comparison",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_3",
            description=f"Retrieve risks and challenges for {side_a}",
            query=f"OrionVault {side_a} risks challenges barriers execution",
            purpose=f"Identify risk factors affecting the {side_a} approach",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_4",
            description=f"Retrieve risks and challenges for {side_b}",
            query=f"OrionVault {side_b} risks challenges barriers execution",
            purpose=f"Identify risk factors affecting the {side_b} approach",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
    ]

    return ResearchPlan(
        objective=f"Compare {side_a} and {side_b} {topic} with associated risks",
        tasks=tasks[:config.max_tasks],
    )


def _plan_temporal(analysis: QueryAnalysis, config: PlanningConfig) -> ResearchPlan:
    """
    TEMPORAL plan template.

    Structure:
      task_1: Retrieve evidence from earlier period
      task_2: Retrieve evidence from later period
      task_3: Retrieve what changed / drivers of change
    """
    periods = analysis.temporal_constraints
    topics  = analysis.topics or ["risk profile", "strategy"]
    topic   = topics[0]

    period_a = periods[0] if len(periods) >= 1 else "2024"
    period_b = periods[1] if len(periods) >= 2 else "2026"

    # Map period to version filter hint
    def _period_filter(period: str) -> dict:
        year = re.search(r"20\d\d", period)
        if year:
            y = int(year.group())
            return {"effective_date_lte": f"{y}-12-31"}
        return {}

    tasks = [
        ResearchTask(
            task_id="task_1",
            description=f"Retrieve {topic} information for {period_a}",
            query=f"OrionVault {topic} {period_a}",
            purpose=f"Establish baseline: {topic} in {period_a}",
            depends_on=[],
            filters=_period_filter(period_a),
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_2",
            description=f"Retrieve {topic} information for {period_b}",
            query=f"OrionVault {topic} {period_b}",
            purpose=f"Establish current state: {topic} in {period_b}",
            depends_on=[],
            filters={"is_current": True} if period_b >= "2026" else _period_filter(period_b),
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_3",
            description=f"Retrieve drivers of change in {topic} between {period_a} and {period_b}",
            query=f"OrionVault {topic} change improvement drivers factors {period_a} {period_b}",
            purpose="Explain what changed and why",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
    ]

    return ResearchPlan(
        objective=f"Analyze how {topic} changed from {period_a} to {period_b}",
        tasks=tasks[:config.max_tasks],
    )


def _plan_cross_document(analysis: QueryAnalysis, config: PlanningConfig) -> ResearchPlan:
    """
    CROSS_DOCUMENT plan template.

    Structure:
      task_1: Retrieve from management commentary / annual review
      task_2: Retrieve from formal risk disclosures
      task_3: Retrieve any supporting context
    """
    topics = analysis.topics or ["Sentinel Copilot"]
    topic  = topics[0]

    tasks = [
        ResearchTask(
            task_id="task_1",
            description=f"Retrieve management commentary about {topic}",
            query=f"OrionVault {topic} management commentary strategy highlights",
            purpose="Capture optimistic management narrative",
            depends_on=[],
            filters={"document_type": "annual_review", "is_current": True},
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_2",
            description=f"Retrieve formal risk disclosures about {topic}",
            query=f"OrionVault {topic} risk factors challenges concerns disclosures",
            purpose="Capture formal risk disclosures and cautions",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_3",
            description=f"Retrieve supporting technical details about {topic}",
            query=f"OrionVault {topic} product features capabilities technical",
            purpose="Provide factual context for the comparison",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
    ]

    return ResearchPlan(
        objective=f"Compare management commentary vs formal risk disclosures for {topic}",
        tasks=tasks[:config.max_tasks],
    )


def _plan_multi_hop(analysis: QueryAnalysis, config: PlanningConfig) -> ResearchPlan:
    """
    MULTI_HOP plan template.

    Multi-hop is hard to template — defer to LLM if available.
    This is the minimal fallback if LLM is unavailable.

    Structure:
      task_1: Retrieve the primary fact/event
      task_2: Retrieve causal factors / background
      task_3: Retrieve consequences / risks (depends on task_1 + task_2)
    """
    q      = analysis.original_query
    topics = analysis.topics or ["financial performance", "risk factors"]

    tasks = [
        ResearchTask(
            task_id="task_1",
            description="Retrieve primary facts related to the query",
            query=f"OrionVault {' '.join(topics[:2])} key findings",
            purpose="Establish the primary facts that the question hinges on",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_2",
            description="Retrieve causal factors and drivers",
            query=f"OrionVault drivers factors causes {' '.join(topics[:1])}",
            purpose="Understand root causes underlying the primary facts",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ),
        ResearchTask(
            task_id="task_3",
            description="Retrieve risks that could affect the outcome",
            query=f"OrionVault risks challenges {' '.join(topics)} outlook",
            purpose="Understand what could reverse or amplify the trend",
            depends_on=["task_1", "task_2"],
            top_k=config.max_task_retrieval,
        ),
    ]

    return ResearchPlan(
        objective=f"Multi-hop research: {q}",
        tasks=tasks[:config.max_tasks],
    )


def _plan_multi_topic(analysis: QueryAnalysis, config: PlanningConfig) -> ResearchPlan:
    """
    MULTI_TOPIC plan template.

    Creates one task per identified topic.
    Caps at config.max_tasks.
    """
    topics = analysis.topics or ["growth strategy", "risk factors", "financial performance"]
    tasks  = []

    for i, topic in enumerate(topics[:config.max_tasks], start=1):
        tasks.append(ResearchTask(
            task_id=f"task_{i}",
            description=f"Retrieve information about {topic}",
            query=f"OrionVault {topic}",
            purpose=f"Cover the '{topic}' topic area in the response",
            depends_on=[],
            top_k=config.max_task_retrieval,
        ))

    return ResearchPlan(
        objective=f"Multi-topic research covering: {', '.join(topics[:config.max_tasks])}",
        tasks=tasks,
    )


_TEMPLATE_MAP = {
    QueryType.SIMPLE:         _plan_simple,
    QueryType.COMPARISON:     _plan_comparison,
    QueryType.TEMPORAL:       _plan_temporal,
    QueryType.CROSS_DOCUMENT: _plan_cross_document,
    QueryType.MULTI_HOP:      _plan_multi_hop,
    QueryType.MULTI_TOPIC:    _plan_multi_topic,
}


# ── LLM-based planner ─────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a research planner for a RAG system over company due-diligence documents.

Given a QueryAnalysis JSON, output a research plan as valid JSON with this schema:
{
  "objective": "One sentence describing the overall research goal",
  "tasks": [
    {
      "task_id": "task_1",
      "description": "What this task retrieves",
      "query": "Retrieval-optimized sub-query (keyword-dense, NOT the original question)",
      "purpose": "Why this information is needed",
      "depends_on": [],
      "filters": {},
      "top_k": 8
    }
  ]
}

RULES:
1. Maximum {max_tasks} tasks. Fewer is better.
2. Sub-queries must be dense keyword strings for vector search (not questions).
   BAD:  "What are the risks in the European market?"
   GOOD: "OrionVault Europe GDPR regulatory risk market entry challenges"
3. Each task must have a DISTINCT purpose. No duplicate tasks.
4. Parallel tasks: set depends_on=[] for tasks that can run simultaneously.
5. Dependent tasks: list prerequisite task_ids in depends_on.
6. SIMPLE queries: produce exactly 1 task with depends_on=[].
7. Allowed filter keys: is_current (bool), document_type (str), effective_date_lte (date str).
8. task_id must match pattern: task_N where N is sequential integer.
9. Output ONLY the JSON object. No markdown, no explanation outside JSON.
"""


def _llm_plan(analysis: QueryAnalysis, config: PlanningConfig) -> Optional[ResearchPlan]:
    """
    Call the LLM to generate a research plan from the query analysis.
    Returns None on failure (caller uses rule-based fallback).
    """
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        model   = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")
        if not api_key:
            return None

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        system = _PLANNER_SYSTEM.replace("{max_tasks}", str(config.max_tasks))

        user_content = (
            f"QueryAnalysis:\n{json.dumps(analysis.model_dump_json_safe(), indent=2)}\n\n"
            f"Original query: {analysis.original_query}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=800,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data  = json.loads(raw)
        tasks = []
        for td in data.get("tasks", []):
            tasks.append(ResearchTask(
                task_id     = td["task_id"],
                description = td.get("description", ""),
                query       = td.get("query", analysis.original_query),
                purpose     = td.get("purpose", ""),
                depends_on  = td.get("depends_on", []),
                filters     = td.get("filters", {}),
                top_k       = min(td.get("top_k", 8), config.max_task_retrieval),
            ))

        return ResearchPlan(
            objective=data.get("objective", f"Research: {analysis.original_query}"),
            tasks=tasks[:config.max_tasks],
        )

    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"[planner] LLM plan failed: {exc}")
        return None


# ── Main Planner class ────────────────────────────────────────────────────────

class ResearchPlanner:
    """
    Transforms a QueryAnalysis into a ResearchPlan.

    Strategy:
      - SIMPLE queries → always use rule template (1 task)
      - MULTI_HOP, CROSS_DOCUMENT → prefer LLM (dependency structure is complex)
      - COMPARISON, TEMPORAL, MULTI_TOPIC → prefer rule template (well-structured)
      - LLM disabled or fails → always fall back to rule template

    Usage:
        planner = ResearchPlanner(config)
        plan = planner.plan(analysis)
    """

    # Query types where LLM planner produces much better plans than templates
    _LLM_PREFERRED = {QueryType.MULTI_HOP, QueryType.CROSS_DOCUMENT}

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        self.config = config or PlanningConfig.default()

    def plan(self, analysis: QueryAnalysis) -> ResearchPlan:
        """
        Generate a research plan for the given query analysis.

        Returns a ResearchPlan ready for validation and execution.
        """
        qt = analysis.query_type

        # Always use rule template for SIMPLE — no LLM overhead
        if qt == QueryType.SIMPLE:
            plan = _plan_simple(analysis, self.config)
            plan.notes = "Simple path — single retrieval task, no decomposition."
            return plan

        # For LLM-preferred types, try LLM first
        if qt in self._LLM_PREFERRED and self.config.allow_llm_planner:
            llm_plan = _llm_plan(analysis, self.config)
            if llm_plan and llm_plan.tasks:
                llm_plan.notes = "LLM-generated plan"
                return llm_plan
            # Fall through to template

        # Template-based planning
        template_fn = _TEMPLATE_MAP.get(qt, _plan_simple)
        plan = template_fn(analysis, self.config)

        # For non-simple types, optionally improve with LLM if template succeeded
        # but we have remaining LLM budget. We skip this to keep latency low.

        plan.notes = f"Template-based plan for {qt.value}"
        return plan
