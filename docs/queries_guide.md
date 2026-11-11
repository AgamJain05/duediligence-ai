# DueDiligenceAI — Queries Guide (Phase 1 to Phase 5)

This file lists all queries you can run against the DueDiligenceAI RAG system, organized by phase.
Each section explains what the phase enables, which mode to select, and the exact questions to ask.

The corpus used is the NexaCore Technologies / QuantumEdge Corp fictional document set
(generated from `docs/create_company_docs.md`).

---

## How to Run Queries

### Terminal CLI (all phases)
```bash
python app.py
```
You will get a REPL prompt. Type your question and press Enter.
The full trace is printed: retrieved chunks → prompt → answer → sources.

### Streamlit UI (all phases with mode selector)
```bash
streamlit run ui.py
```
Select the retrieval mode from the sidebar dropdown before submitting a query.

### Benchmark scripts (batch evaluation)
```bash
# Compare Phase 1 vs Phase 3 vs Phase 4 on all questions
python scripts/compare_phases.py

# Evaluate planning accuracy (Phase 5)
python scripts/evaluate_planning.py

# Retrieval metrics only
python scripts/evaluate_retrieval.py
```

### Debug/Inspect scripts
```bash
# See raw retrieved chunks for a query
python scripts/inspect_retrieval.py "What is NexaCore's FY2025 gross margin?"

# See the research plan for a complex query
python scripts/inspect_plan.py "How did NexaCore's China revenue change from FY2024 to FY2025?"

# See the full research trace end-to-end
python scripts/inspect_research.py "Compare NexaCore and QuantumEdge's profitability"
```

---

## Phase 1 — Baseline FAISS Vector Search

**Mode:** `VECTOR_ONLY`
**What it does:** Embeds your query with BAAI/bge-base-en-v1.5, runs exact cosine search against FAISS index.
**Best for:** Conceptual/semantic questions where the answer is in a single passage.
**Weakness:** Misses exact terminology (ticker symbols, product names, specific numbers).

**How to select:** In `ui.py` sidebar → Retrieval Mode → `VECTOR_ONLY`

### Factual Single-Lookup Queries
```
What is NexaCore's FY2025 net income?
What is NexaCore's gross margin for fiscal year 2025?
Who is the CEO of NexaCore Technologies?
How many employees does NexaCore have?
What is NexaCore's primary AI accelerator product?
What is the NexaCore X2 Accelerator and when was it launched?
What is NexaCore's ticker symbol?
Where is NexaCore Technologies headquartered?
Who is QuantumEdge's CEO?
What is the QuantumEdge EdgeNet Software Platform?
```

### Business/Strategy Queries
```
What is NexaCore's competitive position in the AI chip market?
Why does NexaCore have strong pricing power?
What segments does NexaCore compete in?
What is NexaCore's strategy for the data center market?
How does QuantumEdge differentiate itself from NexaCore?
What is QuantumCloud and why did QuantumEdge launch it?
```

### Risk and Concern Queries
```
What are the main risks facing NexaCore Technologies?
Why is NexaCore dependent on a single manufacturing partner?
What is SilicaFab and why does it matter?
What are the risks of AI chip export controls to China?
What is NexaCore's customer concentration risk?
Why is HBM4 memory a supply chain concern?
```

**Expected Recall@1 (Phase 1 baseline):** ~9–16% — many relevant chunks retrieved but ranking is imperfect.

---

## Phase 2 — Structure-Aware Chunking

**Mode:** `VECTOR_ONLY` (same retrieval, but index built with `--strategy structure_aware`)
**What it does:** Chunks respect section boundaries — a Risk Factor chunk stays within its section heading rather than splitting mid-paragraph across pages.
**Best for:** Section-specific questions, table-heavy queries, hierarchical content.

**How to build this index:**
```bash
python scripts/ingest.py --strategy structure_aware
```
Then in `ui.py` sidebar → Retrieval Mode → `VECTOR_ONLY` (the structure-aware index is loaded automatically if present).

### Section-Specific Queries (benefit from structure awareness)
```
What does the Risk Factors section say about Taiwan?
What does NexaCore's Letter to Shareholders say about FY2025?
What is stated in the Manufacturing and Supply Chain section?
What does the Litigation section of NexaCore's annual report cover?
What are the specific risk factors related to export controls?
What does the Financial Results section show for NexaCore's gross margin?
What does the Outlook section say about FY2026 guidance?
What does QuantumEdge say about the patent litigation against NexaCore?
```

### Table-Heavy Queries (benefit from non-split table chunks)
```
What is the segment revenue breakdown for NexaCore in FY2025?
What is the geographic revenue split for NexaCore FY2025?
Show me NexaCore's full income statement for FY2025.
What is QuantumEdge's revenue by business segment for FY2025?
Compare NexaCore's gross margin across FY2023, FY2024, and FY2025 using the financial table.
```

**Compare Phase 1 vs Phase 2:**
```bash
python scripts/compare_chunking.py
```
This runs the same 41 questions against both indexes and prints Recall@K side-by-side.

---

## Phase 3 — Hybrid Retrieval (BM25 + RRF + Cross-Encoder Reranking)

Phase 3 adds four retrieval modes. Use different modes depending on your query type.

---

### Mode: `KEYWORD_ONLY` (BM25)

**What it does:** Tokenizes query and corpus, scores by BM25Okapi term frequency. No semantic understanding.
**Best for:** Exact product names, ticker symbols, error codes, specific accounting terms.
**Weakness:** Misses synonyms and paraphrase.

**How to select:** In `ui.py` sidebar → `KEYWORD_ONLY`

```
NXCR FY2025 EPS
SilicaFab Taiwan agreement December 2027
HBM3e memory supply allocation
NexaCore X2 160 GB HBM3e
DataSync acquisition $2.3 billion
EAR Part 744 export restrictions
QuantumEdge QE-Ultra FY2026
GlobalChip South Korea manufacturing
NexaCore X3 192 GB HBM4
Project Diversify secondary site South Korea
DOJ antitrust civil investigative demand
patent infringement NexaCore HBM memory controller
```

---

### Mode: `HYBRID` (Semantic + BM25 via Reciprocal Rank Fusion)

**What it does:** Runs both FAISS vector search (top-20) AND BM25 keyword search (top-20), then merges rankings using Reciprocal Rank Fusion (RRF). Chunks appearing in both lists score highest.
**Best for:** Most general research questions — benefits from both exact terminology AND conceptual matching.
**Improvement over Phase 1:** +70% Recall@1 on benchmark.

**How to select:** In `ui.py` sidebar → `HYBRID`

```
What is NexaCore's supply chain risk related to Taiwan?
Why did NexaCore's China revenue decline in FY2025?
What is the gross margin difference between NexaCore and QuantumEdge?
How is NexaCore positioned against AMD and QuantumEdge?
What manufacturing capacity agreements does NexaCore have?
How does QuantumEdge's software business contribute to margins?
What is the status of the QuantumEdge patent lawsuit?
How does NexaCore's EPS compare to QuantumEdge's EPS?
What is SilicaFab's role in NexaCore's production?
Why is NexaCore at risk from US export regulations?
How does the HBM memory bottleneck affect both NexaCore and QuantumEdge?
```

---

### Mode: `HYBRID_RERANKED` (Hybrid + Cross-Encoder Reranking) ← Production Default

**What it does:** HYBRID mode (top-20 candidates from RRF), then cross-encoder model `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores each (query, chunk) pair jointly and selects final top-5.
**Best for:** Highest accuracy retrieval. Use when answer quality matters most.
**Tradeoff:** ~2.9 seconds vs ~470ms for VECTOR_ONLY. Worth it for research use cases.

**How to select:** In `ui.py` sidebar → `HYBRID_RERANKED` (this is the default)

### Full Research Questions (use HYBRID_RERANKED)
```
Analyze NexaCore's key business risks and their potential financial impact.
What drove NexaCore's gross margin expansion from 68% in FY2023 to 76% in FY2025?
Explain the competitive dynamics between NexaCore and QuantumEdge.
What is NexaCore's exposure to US-China trade tensions and how has it evolved?
Assess NexaCore's capital allocation strategy in FY2025.
What are the technology risks facing NexaCore's product roadmap?
How dependent is NexaCore on a small number of cloud customers?
What is the significance of NexaCore's DataSync acquisition?
How did NexaCore's R&D spending change and what does it indicate?
Summarize QuantumEdge's FY2025 performance and strategic direction.
What are the main differences in manufacturing strategy between NexaCore and QuantumEdge?
What evidence supports or contradicts the view that NexaCore has a durable competitive moat?
```

**Check the diagnostics panel** in the Streamlit UI to see the latency breakdown: embedding → semantic → keyword → fusion → rerank.

---

## Phase 4 — PostgreSQL + pgvector (5 Retrieval Modes)

Phase 4 requires a PostgreSQL database with pgvector installed and data ingested.

**Ingest into PostgreSQL first:**
```bash
python scripts/ingest_pg.py
```

**Check database connection:**
```bash
python scripts/inspect_db.py
```

---

### Mode: `POSTGRES_EXACT`

**What it does:** Brute-force cosine scan via `<=>` operator. Exact nearest neighbor, no HNSW index used. Baseline for HNSW comparison.
**Use for:** Verifying that HNSW results match exact search.

```
What is NexaCore's FY2025 revenue?
Who manufactures NexaCore X2 chips?
What is QuantumEdge's gross margin for FY2025?
```

---

### Mode: `POSTGRES_HNSW`

**What it does:** Approximate vector search using HNSW graph index (m=16, ef_search=100). ~2.2× faster than EXACT. This is the standard PostgreSQL vector retrieval mode.
**Best for:** General questions when you want production-speed PostgreSQL retrieval.

```
What is NexaCore's Data Center segment revenue in FY2025?
Explain NexaCore's risk related to SilicaFab.
What is QuantumCloud and how many customers does it have?
What is NexaCore's diluted EPS for FY2025?
What is NexaCore's free cash flow in FY2025?
What is the NexaCore X3 Accelerator and when did it launch?
How many employees does QuantumEdge have?
What is NexaCore's total debt in FY2025?
What is QuantumEdge's operating margin in FY2025?
```

---

### Mode: `POSTGRES_FILTERED`

**What it does:** HNSW vector search WITH relational WHERE filters applied in SQL. Filters: `company`, `document_type`, `fiscal_year`, `section`, `content_type`, `is_current`.
**Best for:** Targeted queries where you know exactly which document/year/section to search. Faster retrieval on large corpora by reducing candidate set.

**How to use filters in `ui.py`:** Select `POSTGRES_FILTERED` mode → use the sidebar filter dropdowns (Content Type, Section, Document Version).

#### Filter by company + fiscal year:
```
# NexaCore FY2023 only → filters: company=NexaCore, fiscal_year=2023
What was NexaCore's gross margin in FY2023?
What products did NexaCore have in FY2023?
What were the main risk factors for NexaCore in FY2023?

# NexaCore FY2025 annual report only → filters: company=NexaCore, document_type=annual_report, is_current=true
What is NexaCore's current share buyback program?
What is the status of Project Diversify?
What is NexaCore's FY2026 revenue guidance?
```

#### Filter by section:
```
# section=Risk Factors
What specific manufacturing risks does NexaCore face?
What export control risks are documented for NexaCore?
What litigation is NexaCore involved in?

# section=Financial Results
What is NexaCore's segment revenue breakdown for FY2025?
What drove NexaCore's gross margin improvement?

# document_type=earnings_call
What did the CEO say about China revenue on the earnings call?
What did analysts ask about QuantumEdge's competition?
```

#### Filter by content_type=table:
```
# content_type=table (retrieves only table chunks)
Show me NexaCore's revenue by geography.
Show me the financial comparison between NexaCore and QuantumEdge.
Show me NexaCore's segment revenue table.
```

---

### Mode: `POSTGRES_FTS`

**What it does:** PostgreSQL full-text search using GIN-indexed tsvector. Pure keyword matching using `ts_rank` and `to_tsquery`.
**Best for:** Exact phrase matching, specific product names, regulatory terms. Similar to BM25 but uses PostgreSQL's built-in FTS engine.

```
SilicaFab Taiwan agreement
HBM3e memory supply
EAR Part 744 export control
DataSync acquisition Seattle
NexaCore X3 Accelerator HBM4
QuantumCloud managed inference API
Project Diversify secondary manufacturing
DOJ antitrust investigation NexaCore
patent infringement HBM memory controller
GlobalChip South Korea QuantumEdge
Meridian Technology Research AI semiconductor
NexaCore X2 160 GB performance
```

---

### Mode: `POSTGRES_HYBRID`

**What it does:** Runs HNSW vector search + PostgreSQL FTS in parallel, then applies Reciprocal Rank Fusion to merge results. Best of both worlds at PostgreSQL layer.
**Best for:** General research questions in production PostgreSQL environment.

```
What is the competitive landscape for AI training chips?
How has NexaCore's profitability evolved over three years?
What are the risks associated with NexaCore's Taiwan supply chain?
Compare NexaCore and QuantumEdge on gross margin and growth.
What actions is NexaCore taking to diversify manufacturing?
How does export control policy affect NexaCore's China business?
What are the HBM memory supply constraints for the AI chip industry?
Explain the significance of the NexaCore vs QuantumEdge patent dispute.
```

---

### Version-Aware Temporal Queries (Phase 4 specific)

Phase 4 stores documents with `effective_date` and `is_current` flags. This enables these queries:

#### Query current documents only (is_current=True):
```
What is NexaCore's current FY2026 revenue guidance?
What is the current status of Project Diversify?
What is NexaCore's current share count?
Who is currently the CEO of NexaCore?
```

#### Query historical documents (filter by effective_date range):
```
# effective_date between 2024-01-01 and 2024-12-31 (FY2023 annual report era)
What was NexaCore's revenue when the AI boom was just beginning?
What risks did NexaCore identify in early 2024?
What was NexaCore's employee count in early 2024?

# effective_date = 2025-03-12 (FY2024 annual report)
What was DataSync's revenue contribution in the year of acquisition?
What was NexaCore's China revenue before export controls took full effect?
```

---

## Phase 5 — Query Planning and Multi-Step Orchestration (LangGraph)

Phase 5 automatically classifies queries and routes them through a LangGraph StateGraph. Complex queries are decomposed into parallel or sequential research tasks.

**How to run Phase 5 queries:**
```bash
# Terminal with full LangGraph trace
python app.py
# (Phase 5 is active automatically for complex queries)

# Inspect the research plan for a query before executing
python scripts/inspect_plan.py "Compare NexaCore and QuantumEdge's gross margins over two years"

# Inspect the full research trace
python scripts/inspect_research.py "How did NexaCore's China revenue change from FY2024 to FY2025 and why?"

# Evaluate planning accuracy across 50 benchmark questions
python scripts/evaluate_planning.py
```

---

### Query Type: `SIMPLE` → routes to `retrieve_simple` node

**Characteristics:** Single fact, single retrieval, no decomposition needed.
**Classification triggers:** Direct factual questions, single metric queries.
**LangGraph path:** `analyze_query` → `route_query` → `retrieve_simple` → `build_context` → `generate_answer`

```
What is NexaCore's FY2025 gross margin?
How many employees does NexaCore have as of FY2025?
What is QuantumEdge's EPS for FY2025?
Who is the CFO of NexaCore Technologies?
What is the NexaCore X2 Accelerator's memory capacity?
When did NexaCore acquire DataSync?
What is the ticker symbol for QuantumEdge?
What is NexaCore's total debt in FY2025?
How much did NexaCore spend on R&D in FY2025?
What is NexaCore's FY2026 revenue guidance?
```

---

### Query Type: `TEMPORAL` → creates date-filtered parallel tasks

**Characteristics:** Questions about change over time, year-over-year comparison, evolution of a metric.
**Classification triggers:** "change", "evolve", "FY2023 vs FY2025", "over time", "how did X grow", "trend".
**LangGraph path:** `analyze_query` → `route_query` → `plan_research` → `execute_tasks` (parallel, date-filtered) → `aggregate_evidence` → `build_context` → `generate_answer`
**Tasks generated:** One task per time period, each with `effective_date` filter.

```
How did NexaCore's gross margin change from FY2023 to FY2025?
How did NexaCore's China revenue evolve from FY2023 to FY2025?
Trace NexaCore's revenue growth from FY2023 to FY2025 and explain the drivers.
How did NexaCore's R&D spending change over three years as a percentage of revenue?
How did QuantumEdge's operating margin evolve from FY2024 to FY2025?
How has NexaCore's employee count changed over the past three years?
How did NexaCore's geographic revenue mix shift from FY2023 to FY2025?
How did NexaCore's risk profile change between FY2023 and FY2025?
How did NexaCore's customer concentration evolve from FY2023 to FY2025?
Describe the evolution of NexaCore's capital allocation strategy from FY2023 to FY2025.
```

---

### Query Type: `COMPARISON` → creates parallel tasks, one per comparison target

**Characteristics:** Side-by-side analysis of two companies, two products, or two strategies.
**Classification triggers:** "compare", "vs", "difference between", "how does X compare to Y", "which is better".
**LangGraph path:** `analyze_query` → `route_query` → `plan_research` → `execute_tasks` (parallel, one per company) → `aggregate_evidence` → `build_context` → `generate_answer`
**Tasks generated:** Task 1 retrieves NexaCore data, Task 2 retrieves QuantumEdge data, results merged.

```
Compare NexaCore and QuantumEdge's gross margins for FY2025.
Compare NexaCore and QuantumEdge's revenue growth rates.
How do NexaCore and QuantumEdge differ in their manufacturing strategies?
Compare NexaCore and QuantumEdge's profitability in FY2025.
How does NexaCore's R&D investment compare to QuantumEdge's?
Compare the competitive positioning of NexaCore X3 and QuantumEdge QE-Ultra.
Compare NexaCore and QuantumEdge's exposure to supply chain risk.
How do NexaCore and QuantumEdge differ in their software revenue strategies?
Compare NexaCore's EPS growth with QuantumEdge's EPS growth from FY2024 to FY2025.
Which company — NexaCore or QuantumEdge — is better positioned for the inference market?
Compare NexaCore's free cash flow to QuantumEdge's and explain the difference.
Compare NexaCore and QuantumEdge's customer concentration risk.
```

---

### Query Type: `MULTI_HOP` → creates sequential dependent tasks

**Characteristics:** Answer requires chaining two retrievals — the second lookup depends on knowing the result of the first.
**Classification triggers:** "which company", "what company makes", "who supplies", "find the [X] of [Y]".
**LangGraph path:** `analyze_query` → `plan_research` → `execute_tasks` (sequential, Task 2 depends on Task 1) → `aggregate_evidence` → `generate_answer`

```
Who manufactures NexaCore's X2 chips and where is that manufacturer located?
What company did NexaCore acquire in FY2024 and what technology did it contribute?
Which company supplies HBM3e memory to NexaCore and what is the risk of that dependency?
Who is SilicaFab and why does NexaCore's production depend on them?
DataSync was acquired by NexaCore — what product did DataSync become after the acquisition?
Which company filed a patent lawsuit against NexaCore and what technology does the dispute involve?
Where is NexaCore's secondary manufacturing site being evaluated under Project Diversify?
What is the HBM4 memory source for NexaCore's X3 chip and what fraction does each supplier provide?
QuantumEdge uses GlobalChip for manufacturing — where is GlobalChip located and how does this compare to NexaCore's supply risk?
Which two Korean suppliers provide HBM memory for NexaCore and what are their names?
```

---

### Query Type: `CROSS_DOCUMENT` → creates tasks filtered by document type

**Characteristics:** Requires synthesizing information from multiple different document types (annual report + earnings call + industry report).
**Classification triggers:** "corroborate", "confirm with", "according to both", "does the earnings call agree with", "what does the industry report say about".
**LangGraph path:** `plan_research` → `execute_tasks` (one task per document type) → `aggregate_evidence` → `generate_answer`

```
Corroborate NexaCore's FY2025 gross margin using both the annual report and the industry analysis.
Does the earnings call confirm the annual report's statement about China revenue?
What does the industry analysis say about NexaCore's market share and does it match the annual report?
Compare what NexaCore's annual report says about QuantumEdge vs what the industry report says.
Verify NexaCore's FY2025 revenue figure across multiple documents.
What does the Meridian Technology Research report say about NexaCore's Taiwan risk compared to what NexaCore itself reports?
Does the Q4 FY2025 earnings call guidance match the annual report's FY2026 outlook?
What do both QuantumEdge's annual report and NexaCore's annual report say about the patent dispute?
Confirm NexaCore's employee count using both the annual report and the earnings call.
What does each document say about the HBM memory supply constraint?
```

---

### Query Type: `MULTI_TOPIC` → creates independent parallel tasks per topic

**Characteristics:** One question asks about multiple unrelated topics simultaneously.
**Classification triggers:** Questions with "and", listing multiple separate topics.
**LangGraph path:** `plan_research` → `execute_tasks` (parallel, independent tasks) → `aggregate_evidence` → `generate_answer`

```
What are NexaCore's main risk factors and what is its FY2025 revenue guidance?
Summarize NexaCore's financial performance and explain its manufacturing risk.
What is QuantumEdge's competitive strategy and what is its gross margin?
What is NexaCore's China exposure and what is its share buyback program?
What are the HBM memory supply constraints and how does this affect both NexaCore's and QuantumEdge's production?
Explain NexaCore's DataSync acquisition and describe its export control challenge.
What is NexaCore's employee count and what is its R&D spending in FY2025?
Describe NexaCore's litigation situation and its FY2026 revenue outlook.
What is QuantumEdge's QuantumCloud service and what is the status of its patent claims against NexaCore?
Summarize the AI semiconductor market size and explain NexaCore's position within it.
```

---

## Validation and Adversarial Queries (Phase 5 — Grounding Layer)

These queries test whether the system correctly flags conflicting evidence, unsupported claims, and misleading analyst opinions.

### Conflict Detection Queries
These should trigger a warning: two sources give different values for the same metric.
```
What percentage of NexaCore's FY2025 revenue came from China?
```
*Expected behavior: The system should retrieve the annual report (8%) AND the earnings call (9%) and note the discrepancy: "NexaCore's FY2025 Annual Report states China revenue = 8% ($4.9B). The Q4 FY2025 Earnings Call transcript states China revenue = 9%. These figures conflict. The annual report is audited; treat the annual report figure as authoritative."*

### Analyst Opinion vs Primary Source
```
What is NexaCore's true adjusted gross margin according to analyst research?
Is NexaCore's 76% gross margin accurate or does analyst analysis suggest a lower figure?
```
*Expected behavior: The system should retrieve both NexaCore's audited 76% gross margin AND the Meridian Research "adjusted" 71% claim, and note: "NexaCore's FY2025 Annual Report (audited) states gross margin = 76.0%. Meridian Technology Research suggests an adjusted figure of ~71%, however this is an analyst estimate, not an audited figure."*

### Forward-Looking / Unsupported Claims
```
Will NexaCore maintain market leadership through 2030?
What will NexaCore's stock price be in 2027?
```
*Expected behavior: The system should answer that it cannot predict future stock prices or market outcomes, and that the earnings call contains only forward-looking management statements which are not evidence.*

### Version / Temporal Attack
```
What was NexaCore's gross margin in FY2023 even though the FY2025 report exists?
Use the oldest annual report to find NexaCore's employee count.
```
*Expected behavior with Phase 4 filtering: The system retrieves FY2023 data when specified. Without filtering, it may return current (FY2025) data — test whether the system correctly attributes which year its answer comes from.*

---

## Quick Reference — Mode Selection Guide

| Your Query Type | Recommended Mode | Why |
|----------------|-----------------|-----|
| Single factual lookup | `HYBRID_RERANKED` | Best accuracy for most queries |
| Exact product name / ticker / code | `KEYWORD_ONLY` or `POSTGRES_FTS` | BM25/FTS beats semantic for exact terms |
| Conceptual / paraphrase | `VECTOR_ONLY` or `POSTGRES_HNSW` | Semantic understands synonyms |
| General research question | `HYBRID_RERANKED` | Best overall, worth the latency |
| Filtered by company/year/section | `POSTGRES_FILTERED` | Add WHERE clause to narrow corpus |
| Year-over-year change | Phase 5 `TEMPORAL` | Auto-decomposes into date-filtered tasks |
| Compare two companies | Phase 5 `COMPARISON` | Auto-runs parallel tasks per company |
| Multi-step reasoning | Phase 5 `MULTI_HOP` | Chains dependent retrievals |
| Multi-document synthesis | Phase 5 `CROSS_DOCUMENT` | Retrieves from each document type |
| Multi-topic in one query | Phase 5 `MULTI_TOPIC` | Parallel independent tasks |

---

## Benchmark Question Sets

### Retrieval Benchmark (41 questions — `data/evaluation/retrieval_questions.json`)

Covers SIMPLE questions with manually labeled ground-truth chunk IDs. Used by:
```bash
python scripts/evaluate_retrieval.py
python scripts/compare_phases.py
```

Sample questions from this set:
```
What is NexaCore's FY2025 gross margin?
Who manufactures NexaCore's X2 chip?
What was NexaCore's China revenue in FY2024?
When did NexaCore launch the X2 Accelerator?
What is SilicaFab's contract expiry date?
What was NexaCore's FY2025 R&D expense?
What is NexaCore's diluted EPS for FY2025?
What is the name of NexaCore's data pipeline product post-DataSync acquisition?
What is the HBM4 memory source for NexaCore X3?
What is QuantumEdge's FY2025 revenue?
```

### Planning Benchmark (50 questions — `data/evaluation/planning_questions.json`)

10 questions per query type (SIMPLE, MULTI_HOP, COMPARISON, TEMPORAL, CROSS_DOCUMENT, MULTI_TOPIC). Used by:
```bash
python scripts/evaluate_planning.py
```

---

## Example End-to-End Research Session

This shows a complete research session using Phase 5 for a full due diligence query:

```
# Start the system
python app.py

# Ask a full due diligence question
> Analyze NexaCore Technologies' financial performance, key business risks, and competitive position as of FY2025.
```

**What Phase 5 does with this query:**
1. `analyze_query` → classifies as `MULTI_TOPIC` (financial performance + risks + competition = 3 topics)
2. `plan_research` → creates 3 parallel tasks:
   - Task 1: "NexaCore FY2025 financial performance" (retrieves from Financial Results sections)
   - Task 2: "NexaCore business risks FY2025" (retrieves from Risk Factors sections)
   - Task 3: "NexaCore competitive position FY2025" (retrieves from Business Overview + industry report)
3. `execute_tasks` → all 3 run in parallel using `ThreadPoolExecutor`
4. `aggregate_evidence` → deduplicates chunks, scores by frequency across tasks
5. `build_context` → assembles top-15 chunks with source citations
6. `generate_answer` → LLM writes structured report

**Expected output format:**
```
Financial Performance:
NexaCore achieved FY2025 revenue of $61.2B (+107% YoY)... [Source: NexaCore FY2025 Annual Report, Financial Results]

Key Business Risks:
1. Manufacturing Concentration: 100% of X-series chips fabricated by SilicaFab in Taiwan... [Source: NexaCore FY2025 Annual Report, Risk Factors]
2. Export Controls: China revenue fell from 20% to 8% due to EAR Part 744... [Source: NexaCore FY2025 Annual Report, Risk Factors]

Competitive Position:
NexaCore holds 72% market share in AI training accelerators... [Source: AI Semiconductor Industry Report 2025, Meridian Technology Research]
```
