"""
src/planning/analyzer_p6.py — Phase 6 tool-aware query analyzer.

Extends the Phase 5 QueryAnalyzer by detecting whether a query needs
documents, metrics, or calculations — and routing to the correct answer path.

DOES NOT replace Phase 5 analyzer. Wraps it.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.planning.models import QueryAnalysis, PlanningConfig
from src.planning.analyzer import QueryAnalyzer
from src.planning.models_p6 import AnswerPath, ToolAwareQueryAnalysis
from src.financial.aliases import CANONICAL_METRICS, resolve_alias


# Patterns that signal metric intent
_METRIC_PATTERNS = [
    r'\b(revenue|gross margin|net income|eps|earnings per share|r&d|capex|free cash flow|fcf|ebitda|operating income|operating margin|gross profit)\b',
    r'\b(how much|what (is|was|were) the|what did|how did .+ change)\b',
    r'\b(grew|declined|increased|decreased|change[d]?|from .+ to)\b',
    r'\$\d',           # dollar sign followed by digit
    r'\bfy\d{4}\b',    # FY2023, FY2024 etc.
    r'\b20(2[0-9])\b', # 2020-2029 year references
]

_CALC_PATTERNS = [
    r'\b(growth|cagr|percent change|how much .+ grow|how much .+ increase|how much .+ decline)\b',
    r'\b(by how much|what .+ multiple|how many times|ratio|basis points?)\b',
    r'\b(compare[d]? .+ to|versus|vs\.?|difference between)\b',
]

_FISCAL_RE = re.compile(r'\b(FY\d{4}|Q[1-4]\s*FY\d{4}|CY\d{4}|\b20[2-9]\d\b)', re.IGNORECASE)
_COMPANY_HINTS = ["nexacore", "quantumedge", "nxcr", "qedg", "meridian"]


class ToolAwareAnalyzer:
    """
    Phase 6 analyzer that wraps Phase 5 QueryAnalyzer.

    Returns ToolAwareQueryAnalysis instead of QueryAnalysis.
    Phase 4/5 behavior is completely unchanged.
    """

    def __init__(self, config: Optional[PlanningConfig] = None):
        self._p5_analyzer = QueryAnalyzer(config or PlanningConfig.default())

    def analyze(self, query: str) -> ToolAwareQueryAnalysis:
        """
        Run Phase 5 classification, then add Phase 6 tool routing.
        """
        # Phase 5 analysis (unchanged)
        p5: QueryAnalysis = self._p5_analyzer.analyze(query)

        q_lower = query.lower()

        # Detect metric mentions
        needs_metrics = bool(re.search('|'.join(_METRIC_PATTERNS), q_lower, re.IGNORECASE))
        needs_calc    = bool(re.search('|'.join(_CALC_PATTERNS), q_lower, re.IGNORECASE))
        needs_docs    = not needs_metrics or bool(re.search(
            r'\b(why|explain|describe|narrative|strategy|risk|context|reason|because|impact)\b',
            q_lower
        ))

        # Extract mentioned metrics
        metrics_mentioned = [
            m for m in CANONICAL_METRICS
            if m.replace("_", " ") in q_lower or m in q_lower
        ]

        # Also check aliases
        words = re.findall(r'\b[\w &]+\b', q_lower)
        for phrase_len in range(3, 0, -1):
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i:i + phrase_len])
                canonical = resolve_alias(phrase)
                if canonical and canonical not in metrics_mentioned:
                    metrics_mentioned.append(canonical)

        # Extract fiscal periods
        fiscal_periods = list({m.upper() for m in _FISCAL_RE.findall(query)})

        # Extract companies
        companies = [c for c in _COMPANY_HINTS if c in q_lower]
        # Also from Phase 5 entities
        for e in p5.entities:
            if e.lower() not in companies:
                companies.append(e)

        # Determine answer path
        answer_path = _determine_answer_path(
            needs_docs=needs_docs,
            needs_metrics=needs_metrics,
            needs_calc=needs_calc,
            requires_decomposition=p5.requires_decomposition,
        )

        return ToolAwareQueryAnalysis.from_phase5(
            analysis=p5,
            needs_documents=needs_docs,
            needs_metrics=needs_metrics,
            needs_calculation=needs_calc,
            metrics_mentioned=list(set(metrics_mentioned)),
            fiscal_periods=fiscal_periods,
            companies=list(set(companies)),
            answer_path=answer_path,
        )


def _determine_answer_path(
    needs_docs: bool,
    needs_metrics: bool,
    needs_calc: bool,
    requires_decomposition: bool,
) -> AnswerPath:
    """Route to the simplest sufficient path."""
    if needs_metrics and not needs_docs and not needs_calc and not requires_decomposition:
        return AnswerPath.METRICS_ONLY
    if needs_metrics and needs_calc and not needs_docs:
        return AnswerPath.METRICS_PLUS_CALC
    if needs_metrics or needs_calc:
        return AnswerPath.HYBRID
    return AnswerPath.DOCS_ONLY
