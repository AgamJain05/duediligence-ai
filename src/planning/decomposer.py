"""
src/planning/decomposer.py  ─  Sub-Query Rewriting
════════════════════════════════════════════════════
Phase 5: Rewrites user questions into retrieval-optimized sub-queries.

WHY REWRITE?
─────────────
User questions are conversational:
  "What were the main factors that led to OrionVault's margin improvement?"

Vector search works best on dense keyword strings that match chunk vocabulary:
  "OrionVault operating margin improvement drivers gross margin FY2025"

DIFFERENCE FROM QUERY DECOMPOSITION:
  Decomposition = splitting one question into multiple sub-questions
  Rewriting     = rephrasing ONE question as a better retrieval query

This module handles rewriting. Decomposition (creating multiple tasks) is the
responsibility of the Planner.

APPROACH: Rule-based expansion
  1. Entity injection — add company name if missing
  2. Temporal expansion — add fiscal year / period if mentioned
  3. Keyword extraction — remove stop words and question words
  4. Domain term injection — add related domain vocabulary

OPTIONAL: LLM-based rewriting
  For MULTI_HOP and CROSS_DOCUMENT, the sub-query benefit from LLM rewriting.
  The LLM is called per task only if allow_llm_planner is enabled.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

# Domain vocabulary expansions — adds related terms to improve recall
_DOMAIN_EXPANSIONS = {
    "revenue":         "revenue ARR subscription total revenue",
    "margin":          "gross margin operating margin EBITDA profitability",
    "risk":            "risk factors challenges threats concerns",
    "strategy":        "strategy growth expansion plan roadmap",
    "customer":        "customer retention churn NPS enterprise",
    "competition":     "competitive landscape competitors market share",
    "regulation":      "regulatory compliance GDPR SOC2 ISO data residency",
    "product":         "product roadmap features Sentinel Copilot VaultGuard",
    "cash":            "free cash flow FCF liquidity capital",
    "employee":        "headcount talent hiring retention workforce",
    "europe":          "Europe EMEA Germany France UK GDPR market",
    "us":              "United States US North America enterprise sales",
    "growth":          "growth rate CAGR expansion market penetration",
}

_STOP_WORDS = {
    "what", "how", "why", "when", "where", "who", "which", "is", "are",
    "was", "were", "the", "a", "an", "in", "of", "to", "for", "and",
    "or", "but", "on", "at", "by", "from", "with", "that", "this",
    "it", "its", "their", "they", "have", "has", "had", "did", "do",
    "does", "can", "could", "would", "should", "may", "might", "will",
    "about", "between", "compare", "explain", "describe", "tell", "give",
    "discuss", "analysis", "analyze", "review", "looking", "looking",
}


def _inject_company(query: str, company: str = "OrionVault") -> str:
    """Ensure the company name is present in the sub-query."""
    if company.lower() not in query.lower():
        return f"{company} {query}"
    return query


def _inject_temporal(query: str, periods: List[str]) -> str:
    """Add temporal markers to the query if not already present."""
    for p in periods:
        if p.lower() not in query.lower():
            query = f"{query} {p}"
    return query


def _extract_keywords(query: str) -> List[str]:
    """Extract non-stop-word tokens from the query."""
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", query)
    return [t for t in tokens if t.lower() not in _STOP_WORDS and len(t) > 2]


def _expand_domain_terms(keywords: List[str]) -> List[str]:
    """Add domain-specific related terms for known keywords."""
    expansions = list(keywords)
    for kw in keywords:
        kw_lower = kw.lower()
        for domain_kw, expansion in _DOMAIN_EXPANSIONS.items():
            if domain_kw in kw_lower:
                new_terms = expansion.split()
                for t in new_terms:
                    if t not in expansions:
                        expansions.append(t)
                break
    return expansions


def rewrite_for_retrieval(
    query: str,
    company: str = "OrionVault",
    periods: Optional[List[str]] = None,
    expand_domain: bool = True,
) -> str:
    """
    Rewrite a user question as a retrieval-optimized sub-query.

    Steps:
      1. Extract keywords (remove stop words, question words)
      2. Inject company name
      3. Inject temporal markers
      4. Expand with domain vocabulary (optional)

    Example:
      Input:  "What factors caused the operating margin improvement?"
      Output: "OrionVault operating margin improvement factors drivers profitability"
    """
    periods = periods or []

    # 1. Extract keywords
    keywords = _extract_keywords(query)

    # 2. Inject company
    if company.lower() not in [k.lower() for k in keywords]:
        keywords.insert(0, company)

    # 3. Inject temporal markers
    for p in periods:
        if p not in keywords:
            keywords.append(p)

    # 4. Expand with domain terms
    if expand_domain:
        keywords = _expand_domain_terms(keywords)

    # Deduplicate preserving order, cap at 20 tokens
    seen = set()
    result = []
    for kw in keywords:
        kl = kw.lower()
        if kl not in seen:
            seen.add(kl)
            result.append(kw)

    return " ".join(result[:20])


def _llm_rewrite(
    task_description: str,
    original_query: str,
    context: str = "",
) -> Optional[str]:
    """
    LLM-based sub-query rewriting for complex tasks.

    Returns None on failure — caller uses rule-based rewrite.
    """
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        model   = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")
        if not api_key:
            return None

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        prompt = f"""\
Original query: {original_query}
Task to execute: {task_description}
Context (query type, entities): {context}

Write ONE retrieval sub-query for this task.
Requirements:
- Dense keyword string (NOT a question)
- 8-15 words
- Include company name "OrionVault"
- Include topic-specific vocabulary
- Do NOT start with "What", "How", "Why"

Output: the sub-query string only. No quotes, no explanation."""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().strip('"\'')
        if result and len(result.split()) >= 4:
            return result
        return None

    except Exception:
        return None


class SubQueryDecomposer:
    """
    Rewrites task descriptions into retrieval-optimized sub-queries.

    The planner creates task descriptions in natural language.
    The decomposer rewrites those descriptions into dense keyword strings
    suitable for vector search.

    Usage:
        decomposer = SubQueryDecomposer(config, analysis)
        task.query = decomposer.rewrite(task)
    """

    def __init__(self, config=None, analysis=None) -> None:
        from src.planning.models import PlanningConfig
        self.config   = config or PlanningConfig.default()
        self.analysis = analysis  # QueryAnalysis for context

    def rewrite(self, task) -> str:
        """
        Rewrite a task's query for retrieval.

        If the task already has a good query (set by LLM planner),
        just return it with minor cleanup.
        If the query is the same as the original question,
        rewrite it based on the task description.
        """
        original_q = self.analysis.original_query if self.analysis else ""
        task_query = task.query.strip()

        # If the planner already set a distinct retrieval query, trust it
        if task_query and task_query.lower() != original_q.lower():
            # Still inject company name if missing
            return _inject_company(task_query)

        # Otherwise, rewrite from task description
        periods = self.analysis.temporal_constraints if self.analysis else []
        return rewrite_for_retrieval(
            task.description,
            periods=periods,
            expand_domain=True,
        )
