"""
src/planning/analyzer.py  ─  Query Analysis + Classification
══════════════════════════════════════════════════════════════
Phase 5: Determines query type and whether decomposition is needed.

CLASSIFIER STRATEGY: Hybrid — Rules + LLM
──────────────────────────────────────────
WHY HYBRID?

Rule-based classifiers are:
  ✓ Fast (no LLM call overhead)
  ✓ Deterministic (same query → same type)
  ✓ Transparent (reasoning is code, not hidden in model weights)
  ✗ Brittle for ambiguous phrasing

LLM-based classifiers are:
  ✓ Flexible (handles paraphrasing and novel structures)
  ✗ Slow (adds 1–2s latency per query)
  ✗ Non-deterministic (temperature > 0)
  ✗ Requires API call

STRATEGY:
  1. Run rule-based classifier first (~1ms)
  2. If rules produce high-confidence result → use it
  3. If ambiguous (confidence < RULE_CONFIDENCE_THRESHOLD) → call LLM
  4. LLM returns JSON QueryAnalysis

This gives the speed of rules for 80% of queries and LLM quality
for the 20% that rules can't confidently classify.

DOCUMENT: The choice of rule vs LLM for each query is logged in
QueryAnalysis.classifier_method so you can inspect it.

IMPORTANT DESIGN PRINCIPLE:
Simple queries should NOT be pushed toward complex classification.
When in doubt, classify as SIMPLE and let the planner handle the
single-retrieval path efficiently.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

from src.planning.models import (
    PlanningConfig,
    QueryAnalysis,
    QueryType,
    COMPLEX_QUERY_TYPES,
)

# ── Rule confidence threshold ─────────────────────────────────────────────────
# If rules produce confidence >= this, skip LLM call
RULE_CONFIDENCE_THRESHOLD = 0.75


# ── Keyword / pattern signals for each query type ─────────────────────────────
_COMPARISON_SIGNALS = [
    r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bdifference between\b",
    r"\bhow does .+ differ\b", r"\bcontrast\b", r"\bsimilarities? and differences?\b",
    r"\brelative to\b", r"\bagainst\b",
]

_TEMPORAL_SIGNALS = [
    r"\b(how|what).+changed?\b",
    r"\b202[0-9].+to.+202[0-9]\b",
    r"\bfrom\s+\d{4}\s+to\s+\d{4}\b",
    r"\byear[ -]over[ -]year\b",
    r"\bover time\b",
    r"\b(before|after)\s+(the\s+)?\d{4}\b",
    r"\bhistoric(al)?\b",
    r"\bevolution\b",
    r"\btrend\b",
    r"\bevolved?\b",
    r"\bfy20\d\d.+fy20\d\d\b",
    r"\bimprove.+from\s+fy\b",
    r"\bchanges?.+between\s+20\d\d\b",
    r"\bwhat.+resolved\b",
]

_MULTI_HOP_SIGNALS = [
    r"\bwhy\b.+\band\b.+\bwhat\b",
    r"\bwhat.+caused?.+and.+what\b",
    r"\bfactors?.+led to\b",
    r"\bimplicat(ion|e)\b.+\brisk\b",
    r"\bbecause of\b.+\bwhat\b",
    r"\bif.+then\b",
    r"\bconsequence\b",
    r"\bachieve.+and.+what.+threaten\b",
    r"\bdrove?.+and.+what.+risk\b",
    r"\bwhat.+drove?.+and.+how\b",
    r"\bwhat.+(drove?|caused?|led to|resulted in).+and\b",
    r"\bhow.+contributed?.+and.+what.+risk\b",
    r"\bfactors?.+caused?.+and.+steps\b",
    r"\brelationship between\b",
    r"\bdrove?.+revenue.+and.+what\b",
    r"\bpersonnel.+costs?.+grow\b",
    r"\bpartnership.+strateg.+and.+risk\b",
    r"\bhow.+(concentrat|expand|focus).+creat.+\b",
    r"\bsustainab\b",
    r"\bpath to profitab\b",
]

_CROSS_DOCUMENT_SIGNALS = [
    r"\bmanagement.+comment(ary|s)?\b",
    r"\bformal.+risk\b",
    r"\bofficial.+statement\b",
    r"\bcompare?.+with.+disclosure\b",
    r"\bversus.+report\b",
    r"\bannual report.+versus\b",
    r"\bdossier.+vs\b",
    r"\brisk memo\b.+\bvs\b",
    r"\bforward.looking.+with.+risk\b",
    r"\bmanagement.+vs.+risk\b",
    r"\bmanagement.+align.+risk\b",
    r"\bannual review.+risk\b",
    r"\bmanagement.+section.+risk\b",
    r"\bcharacterize.+capabilities.+risk\b",
    r"\bdescribe.+vs.+risk\b",
    r"\bproject(ion|s)?.+risk.+miss\b",
    r"\bhow does.+describe.+(overview|versus|compared)\b",
    r"\bproduct.market fit.+risk\b",
    r"\broadmap.+risk\b",
    r"\bwhat.+say about.+(expan|strateg|product).+risk\b",
    r"\brisk.+qualif\b",
    r"\bregulatory compliance.+approach.+risk\b",
    r"\bcompliance.+approach.+formal\b",
]

_MULTI_TOPIC_SIGNALS = [
    r"\band (the|each|their)\b.+(risk|strateg|priorit)\b",
    r"\bfor each\b",
    r"\bmultiple\b.+(area|aspect|topic)\b",
    r"\b(strateg|priorit).+\band\b.+(risk|challenge)\b",
    r"\bgrowth.+priorit.+risk\b",
]

# Very short or purely factual patterns → SIMPLE
_SIMPLE_SIGNALS = [
    r"\bwhat is\s+(the\s+)?(revenue|margin|fcf|arr|employees|ceo|cfo|founded)\b",
    r"\bwhat (was|were) (the\s+)?\w+'s?\s+fy20\d\d\s+\w+\b",
    r"\bwho (is|are)\b",
    r"\bwhen (was|did)\b",
    r"\bhow (many|much)\b",
    r"\bwhat (is|are) the (current\s+)?\w+\b",
]


def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_COMPILED = {
    QueryType.SIMPLE:         _compile(_SIMPLE_SIGNALS),
    QueryType.COMPARISON:     _compile(_COMPARISON_SIGNALS),
    QueryType.TEMPORAL:       _compile(_TEMPORAL_SIGNALS),
    QueryType.MULTI_HOP:      _compile(_MULTI_HOP_SIGNALS),
    QueryType.CROSS_DOCUMENT: _compile(_CROSS_DOCUMENT_SIGNALS),
    QueryType.MULTI_TOPIC:    _compile(_MULTI_TOPIC_SIGNALS),
}


def _count_matches(query: str, patterns: List[re.Pattern]) -> int:
    return sum(1 for p in patterns if p.search(query))


def _extract_entities(query: str) -> List[str]:
    """Heuristic entity extraction — proper nouns and product names."""
    # Known entity canonicalization (maps variants → canonical form)
    canonical_map = {
        "european": "Europe",
        "europe": "Europe",
        "emea": "EMEA",
        "united states": "United States",
        "u.s.": "US",
        " us ": "US",
        "north america": "North America",
        "germany": "Germany",
        "france": "France",
        "sentinel copilot": "Sentinel Copilot",
        "vaultguard": "VaultGuard",
        "vaultsync": "VaultSync",
        "orionvault": "OrionVault",
        "gdpr": "GDPR",
        "soc2": "SOC2",
    }
    entities = []
    q_lower = query.lower()

    # Extract canonical known entities in order of appearance
    for variant, canonical in canonical_map.items():
        if variant in q_lower and canonical not in entities:
            entities.append(canonical)

    # Also extract capitalized proper nouns not in known set
    tokens = query.split()
    for i, tok in enumerate(tokens):
        clean = tok.strip("?.,!\"'()")
        clean_nopos = clean.rstrip("'s").rstrip("'")
        if (i > 0 and clean_nopos and clean_nopos[0].isupper() and
                len(clean_nopos) >= 3 and clean_nopos not in entities):
            entities.append(clean_nopos)

    return list(dict.fromkeys(entities))  # deduplicate, preserve order


def _extract_topics(query: str) -> List[str]:
    """Simple topic extraction from keyword signals."""
    topic_map = {
        "revenue":         "financial performance",
        "margin":          "financial performance",
        "risk":            "risk analysis",
        "strategy":        "growth strategy",
        "expansion":       "growth strategy",
        "growth":          "growth strategy",
        "product":         "product portfolio",
        "sentinel":        "product portfolio",
        "customers?":      "customer relationships",
        "retention":       "customer relationships",
        "regulation":      "regulatory compliance",
        "gdpr":            "regulatory compliance",
        "compl":           "regulatory compliance",
        "competition":     "competitive landscape",
        "compet":          "competitive landscape",
        "employee":        "workforce",
        "team":            "workforce",
        "financ":          "financial performance",
        "fcf":             "financial performance",
        "cash":            "financial performance",
    }
    topics_found = set()
    q_lower = query.lower()
    for kw, topic in topic_map.items():
        if re.search(kw, q_lower):
            topics_found.add(topic)
    return list(topics_found)


def _extract_temporal(query: str) -> List[str]:
    """Extract year / fiscal year references."""
    patterns = [
        r"FY\s*20\d\d",
        r"\b20\d\d\b",
        r"\bQ[1-4]\s*FY?\s*20\d\d\b",
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, query, re.IGNORECASE))
    return list(dict.fromkeys(found))


def _rule_classify(query: str) -> tuple[QueryType, float, str]:
    """
    Rule-based classification.

    Returns (query_type, confidence, reasoning).
    Higher confidence = rules are surer.
    """
    scores: dict[QueryType, int] = {qt: 0 for qt in QueryType}

    for qt, patterns in _COMPILED.items():
        scores[qt] = _count_matches(query, patterns)

    # SIMPLE has highest priority if it matches strongly
    if scores[QueryType.SIMPLE] >= 1 and scores[QueryType.COMPARISON] == 0 \
            and scores[QueryType.TEMPORAL] == 0:
        return QueryType.SIMPLE, 0.9, "Simple factual pattern detected"

    # Remove SIMPLE from competition for complex types
    del scores[QueryType.SIMPLE]

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score == 0:
        # No signals → default SIMPLE
        return QueryType.SIMPLE, 0.7, "No complex signals found; defaulting to SIMPLE"

    total = sum(scores.values())
    confidence = best_score / total if total > 0 else 0.5
    reasoning = f"{best_type.value} pattern matched ({best_score} signals out of {total} total)"

    # Temporal + Comparison combo → TEMPORAL takes priority
    if scores.get(QueryType.TEMPORAL, 0) >= 1 and scores.get(QueryType.COMPARISON, 0) >= 1:
        return QueryType.TEMPORAL, 0.8, "Both temporal and comparison signals; classified as TEMPORAL"

    return best_type, min(confidence + 0.1, 1.0), reasoning


def _llm_classify(query: str, config: PlanningConfig) -> Optional[QueryAnalysis]:
    """
    LLM-based query classification (fallback when rules are uncertain).

    Returns a QueryAnalysis or None on failure.
    Uses the same OpenRouter endpoint as the generator.
    """
    try:
        import os
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        model   = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it")

        if not api_key:
            return None

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        system_prompt = """\
You are a query analyzer for a RAG system about company due-diligence documents.

Classify the user's question and output ONLY valid JSON with this schema:
{
  "query_type": "SIMPLE|MULTI_HOP|COMPARISON|TEMPORAL|CROSS_DOCUMENT|MULTI_TOPIC",
  "requires_decomposition": true|false,
  "entities": ["list of named entities"],
  "topics": ["list of topic areas"],
  "temporal_constraints": ["list of years or periods mentioned"],
  "document_type_hints": ["annual_review"|"risk_memo" if relevant, else []],
  "expected_subquery_count": integer,
  "reasoning": "one sentence explanation"
}

Definitions:
- SIMPLE: single factual question, one retrieval sufficient
- MULTI_HOP: requires chaining multiple facts
- COMPARISON: compares two entities, strategies, or positions
- TEMPORAL: compares two time periods
- CROSS_DOCUMENT: needs evidence from two structurally distinct sources
- MULTI_TOPIC: multiple independent research areas each needing separate retrieval

Rules:
- Be conservative. Default to SIMPLE if unclear.
- SIMPLE queries must have requires_decomposition=false and expected_subquery_count=1.
- Complex types: expected_subquery_count should be 2-6.
- Do NOT output markdown or explanation outside the JSON."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Query: {query}"},
            ],
            max_tokens=400,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        qt   = QueryType(data["query_type"])

        return QueryAnalysis(
            original_query=query,
            query_type=qt,
            requires_decomposition=qt in COMPLEX_QUERY_TYPES,
            entities=data.get("entities", []),
            topics=data.get("topics", []),
            temporal_constraints=data.get("temporal_constraints", []),
            document_type_hints=data.get("document_type_hints", []),
            expected_subquery_count=max(1, data.get("expected_subquery_count", 1)),
            classifier_method="llm",
            classifier_reasoning=data.get("reasoning", ""),
            confidence=0.85,
        )

    except Exception as exc:
        # Any failure → return None, caller falls back to rules
        import logging
        logging.getLogger(__name__).warning(f"[analyzer] LLM classify failed: {exc}")
        return None


class QueryAnalyzer:
    """
    Hybrid query analyzer.

    Usage:
        analyzer = QueryAnalyzer(config)
        analysis = analyzer.analyze("Compare US and European strategies")
        # analysis.query_type == QueryType.COMPARISON
        # analysis.requires_decomposition == True
    """

    def __init__(self, config: Optional[PlanningConfig] = None) -> None:
        self.config = config or PlanningConfig.default()

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Classify query and extract structure.

        Pipeline:
          1. Run rule-based classifier
          2. If confidence >= threshold → use rule result
          3. If confidence < threshold AND allow_llm_classifier → try LLM
          4. Merge: use LLM result if available, else rule result
        """
        query = query.strip()

        # Step 1: Rule-based classification
        rule_type, rule_confidence, rule_reasoning = _rule_classify(query)

        # Step 2: High-confidence rule result → use directly
        if rule_confidence >= RULE_CONFIDENCE_THRESHOLD or not self.config.allow_llm_classifier:
            return QueryAnalysis(
                original_query=query,
                query_type=rule_type,
                requires_decomposition=rule_type in COMPLEX_QUERY_TYPES,
                entities=_extract_entities(query),
                topics=_extract_topics(query),
                temporal_constraints=_extract_temporal(query),
                expected_subquery_count=_estimate_task_count(rule_type),
                classifier_method="rules",
                classifier_reasoning=rule_reasoning,
                confidence=rule_confidence,
            )

        # Step 3: Ambiguous → try LLM
        llm_result = _llm_classify(query, self.config)
        if llm_result:
            return llm_result

        # Step 4: LLM failed → use rule result despite lower confidence
        return QueryAnalysis(
            original_query=query,
            query_type=rule_type,
            requires_decomposition=rule_type in COMPLEX_QUERY_TYPES,
            entities=_extract_entities(query),
            topics=_extract_topics(query),
            temporal_constraints=_extract_temporal(query),
            expected_subquery_count=_estimate_task_count(rule_type),
            classifier_method="rules_fallback",
            classifier_reasoning=f"LLM unavailable. {rule_reasoning}",
            confidence=rule_confidence,
        )


def _estimate_task_count(qt: QueryType) -> int:
    """Conservative estimate of task count by query type."""
    return {
        QueryType.SIMPLE:         1,
        QueryType.MULTI_HOP:      3,
        QueryType.COMPARISON:     4,
        QueryType.TEMPORAL:       3,
        QueryType.CROSS_DOCUMENT: 3,
        QueryType.MULTI_TOPIC:    4,
    }[qt]
