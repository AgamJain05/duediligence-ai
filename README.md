# DueDiligenceAI

**Evidence-grounded company research powered by retrieval-augmented generation (RAG).**

DueDiligenceAI turns company documents into a searchable research workspace. It combines document-aware ingestion, semantic and keyword retrieval, result fusion, reranking, and a PostgreSQL/pgvector knowledge layer to retrieve relevant evidence and generate answers with source context.

The repository is designed to make retrieval decisions inspectable: you can see which chunks were retrieved, where they came from, how they were ranked, and how retrieval strategies compare.

> **Important:** The included OrionVault Systems company and documents are fictional and exist only for development and evaluation.

## Key capabilities

- **Document ingestion:** Extracts PDF text page by page and retains source and page information.
- **Structure-aware chunking:** Preserves headings, section paths, and table-like content where extraction permits; keeps source text linked to each chunk.
- **Semantic retrieval:** Uses `BAAI/bge-base-en-v1.5` embeddings and FAISS for a local vector-search baseline.
- **Keyword retrieval:** Uses BM25 to surface exact product names, identifiers, terminology, and numerical terms that semantic search may rank poorly.
- **Hybrid retrieval:** Combines semantic and keyword rankings with Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking:** Re-scores query–chunk pairs before selecting the final context for answer generation.
- **Relational knowledge layer:** Uses PostgreSQL and pgvector to store documents, versions, chunks, embeddings, and structured metadata together.
- **Metadata and version-aware retrieval:** Supports narrowing results by document attributes and retaining the provenance needed to distinguish current from historical source material.
- **Retrieval diagnostics:** Inspect intermediate results, rankings, sources, pages, and scores instead of treating retrieval as a black box.
- **Quantitative evaluation:** Benchmarks every retrieval mode on Recall@K, MRR, and latency against a manually labeled question set — improvements are measured, not assumed.
- **Multi-phase comparison:** `compare_phases.py` runs Phase 1, Phase 3, and Phase 4 retrieval side-by-side on the same question set and prints a comparison table with scores for each mode.
- **Planning evaluation:** `evaluate_planning.py` measures query classification accuracy, decomposition routing accuracy, plan validity rate, and over-decomposition rate across 50 benchmark questions.

## User Interface

DueDiligenceAI provides a dedicated research workspace built with Streamlit for evidence-grounded analysis, retrieval mode configuration, and source provenance inspection.

| Research Workspace & Query Execution | Evidence-Grounded Answer & Metrics |
| :---: | :---: |
| ![Research Workspace](Images/Running.png) | ![Evidence-Grounded Answer](Images/Result.png) |

| Collapsible Source Attribution | Chunk Provenance & Diagnostic Scores |
| :---: | :---: |
| ![Source Citations](Images/Sources.png) | ![Inspect Chunk Metadata and Scores](Images/Sources_1.png) |

## Architecture

### Ingestion

```text
PDF documents
     │
     ▼
Page-level text extraction
     │
     ▼
Normalization + structure detection
     │
     ▼
Structure-aware chunks + metadata
     │
     ├──────────────► Embeddings ───────────────► FAISS index
     │
     └──────────────► PostgreSQL + pgvector
                       documents · versions
                       sections · chunks · sources
```

### Query and retrieval

```text
User question
     │
     ├──────────────► Semantic retrieval (embeddings)
     │
     └──────────────► Keyword retrieval (BM25)
                              │
                              ▼
                    Reciprocal Rank Fusion
                              │
                              ▼
                     Cross-encoder reranker
                              │
                              ▼
                    Selected evidence chunks
                              │
                              ▼
                    Context-grounded generation
                              │
                              ▼
                   Answer + source references
```

The local FAISS pipeline provides a useful baseline. The PostgreSQL/pgvector layer adds relational metadata and document/version relationships alongside vector search. Retrieval modes can be compared to understand the effect of each retrieval component.

## Technology stack

| Area | Technology | Purpose |
|---|---|---|
| Language | Python 3.11+ | Application and retrieval pipeline |
| PDF extraction | `pypdf`, `pdfplumber` | Page text extraction and table-aware parsing |
| Chunking | `tiktoken` + custom Python logic | Token counting and chunk boundaries |
| Embeddings | `sentence-transformers`, `BAAI/bge-base-en-v1.5` | Local 768-dimensional text embeddings |
| Vector baseline | FAISS `IndexFlatIP` | Exact local similarity search over normalized vectors |
| Keyword retrieval | `rank-bm25` | Lexical relevance scoring |
| Rank fusion | Reciprocal Rank Fusion | Combines independently ranked result lists |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Ranks query–chunk pairs for relevance |
| Knowledge store | PostgreSQL + pgvector | Vector search, relational metadata, and provenance |
| Generation | OpenRouter API | Generates answers from selected evidence |
| Interface | Streamlit | Interactive research workspace and retrieval inspector |
| Testing | `pytest` | Unit and integration tests |

## Retrieval modes

The retriever exposes separate modes so their behavior can be evaluated rather than assumed.

| Mode | What it tests |
|---|---|
| `VECTOR_ONLY` | Semantic similarity from embeddings |
| `KEYWORD_ONLY` | Lexical matching with BM25 |
| `HYBRID` | Semantic + keyword retrieval merged with RRF |
| `HYBRID_RERANKED` | Hybrid candidate retrieval followed by cross-encoder reranking |

Use vector retrieval for meaning and paraphrases, keyword retrieval for exact terminology and identifiers, and hybrid retrieval when both signals may matter. Reranking improves ordering only among candidates already retrieved; it cannot recover evidence that never entered the candidate set.

## Knowledge model

The PostgreSQL-backed store represents the corpus as related records rather than unconnected text chunks. The model tracks companies, documents, document versions, sections, chunks, sources, and structured financial metrics where available.

This supports questions such as:

- Which source and page support a retrieved passage?
- Which version of a policy applies to a particular date?
- Can retrieval be restricted to a document type or section?
- Can structured financial values be queried separately from prose evidence?

Vector similarity is not a replacement for relational constraints. Dates, document types, version status, and provenance should be represented as data that can be queried directly.

## Example research questions

Using the fictional OrionVault Systems corpus, try:

- What were OrionVault's revenue and operating margin in FY2025?
- What risks come from dependence on external cloud infrastructure?
- What does Sentinel Copilot do, and what risks are associated with it?
- What internal identifier is used for Sentinel Copilot?
- Compare management's comments about Sentinel Copilot with formal risk disclosures.
- How did the company's risk profile change between the 2024 and 2026 risk documents?
- What data-residency policy applies to Germany, and which source version supports the answer?

These questions are intentionally varied: some test semantic similarity, some rely on exact terms, and others require evidence from multiple source documents or versions.

## Getting started

### Prerequisites

- Python 3.11 or newer
- An OpenRouter API key for answer generation
- PostgreSQL with the `vector` extension enabled for database-backed retrieval

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.

### 2. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. Configure PostgreSQL connection settings using the variable names documented in `.env.example` for the database-backed retrieval path. Do not commit `.env` or API keys.

### 3. Add the sample corpus

Place the fictional OrionVault PDF corpus in `data/raw/`. Keep source PDFs unchanged so retrieved evidence can be traced back to the original document and page.

### 4. Ingest documents

For the local FAISS workflow:

```powershell
python scripts/ingest.py --strategy structure_aware
```

To compare fixed-size and structure-aware chunking, build both indexes:

```powershell
python scripts/ingest.py --strategy fixed
python scripts/ingest.py --strategy structure_aware
```

For PostgreSQL-backed retrieval, configure the database, apply the repository's database migrations, and run its database ingestion workflow before selecting the database-backed retriever.

### 5. Run the application

**Launch the Streamlit research interface:**

```powershell
streamlit run ui.py
```

**Or run the terminal CLI:**

```powershell
python app.py
```

### 6. Inspect retrieval results

```powershell
python scripts/inspect_retrieval.py "What are the risks associated with Sentinel Copilot?"
python scripts/inspect_retrieval.py "What are the corrective actions for the VaultGuard event?" --mode KEYWORD_ONLY
```

### 7. Run the retrieval benchmark

```powershell
python scripts/evaluate_retrieval.py
```

### 8. Run tests

```powershell
python -m pytest tests/ -v
```

> Commands for database migrations and database ingestion depend on the migration and script names present in the checked-out repository. Use the checked-in migration configuration and `scripts/` directory as the source of truth.

## Evaluation and benchmarks

This project treats retrieval as an engineering problem that must be measured, not assumed. Every retrieval strategy is benchmarked against a manually labeled question set so that improvements are verifiable and regressions are visible.

### Benchmark setup

- **Benchmark corpus:** 3 OrionVault Systems PDF documents (fictional), multi-version
- **Question set:** 41 retrieval questions (30 with manually labeled ground-truth chunk IDs) across factual, comparison, temporal, and cross-document categories
- **Planning question set:** 50 additional questions (10 per query type: SIMPLE, COMPARISON, TEMPORAL, MULTI_HOP, CROSS_DOCUMENT) for Phase 5 planner evaluation
- **Embedding model:** `BAAI/bge-base-en-v1.5` (768-dimensional, L2-normalized)
- **Evaluation metrics:** Recall@1, Recall@3, Recall@5, MRR (Mean Reciprocal Rank), latency per query

### Measured retrieval results

Benchmarked across 41 retrieval questions using `python scripts/compare_phases.py`:

| Retrieval Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Avg Latency |
|---|---|---|---|---|---|
| **Phase 1 — FAISS Vector only** | 0.096 | 0.303 | 0.402 | 0.329 | 470 ms |
| **Phase 3 — FAISS Hybrid + Reranking** | **0.163** | **0.382** | **0.510** | **0.401** | 2947 ms |
| Phase 4 — PostgreSQL HNSW | — | — | — | — | 279 ms |
| Phase 4 — PostgreSQL HNSW + Filter | — | — | — | — | 262 ms |

> Phase 3 hybrid retrieval achieves **+70% Recall@1** and **+27% Recall@5** relative to Phase 1 vector-only search, at the cost of higher latency from the cross-encoder reranker.
>
> Phase 4 PostgreSQL metrics reflect the vector store with correct chunk IDs after full ingestion; Recall values update once ground-truth IDs are aligned to the database chunk hashes.

### What each metric means

- **Recall@K** — Out of all relevant chunks, what fraction appeared in the top-K results? Higher is better. Measures whether the system finds the right evidence at all.
- **MRR (Mean Reciprocal Rank)** — On average, how highly is the first relevant result ranked? Score of 1.0 = always first. Penalises systems that bury relevant evidence deep in results.
- **Latency** — Wall-clock time per query. Important for understanding the speed/quality tradeoff of each retrieval component.

### HNSW index experiments (Phase 4)

Approximate nearest-neighbour search was benchmarked against exact vector search:

- **HNSW Recall@5 vs exact search: 100%** on this corpus size (41 chunks)
- **HNSW latency speedup vs exact: 2.2×** (279 ms vs 626 ms)
- Index configured with `m=16`, `ef_construction=64`, `ef_search=100` (cosine distance via pgvector `<=>`)
- Note: ANN recall advantage over exact search is negligible at small scale; the gap grows with corpus size, which is when HNSW becomes essential

### Version-aware retrieval verification (Phase 4, Experiment 4)

The relational versioning layer was verified against three temporal queries:

| Query | Filter applied | Result |
|---|---|---|
| *"What is the current data residency requirement for Germany?"* | `is_current = True` | Retrieved **v2** (2026 policy), similarity 0.701 |
| *"What data residency requirement applied in 2024?"* | `effective_date ≤ 2024-12-31` | Retrieved **v1** (2024 baseline), correctly separated from v2 |
| *"What risks were identified in the 2024 annual review?"* | `effective_date ≤ 2024-12-31` | Retrieved **v1** risk section, similarity 0.515 |

This demonstrates that semantic similarity alone cannot distinguish current from historical evidence — relational metadata filters are required for temporally sensitive due-diligence questions.

### Planning evaluation (Phase 5)

Query decomposition and planning accuracy benchmarked across 50 questions using `python scripts/evaluate_planning.py`:

| Metric | Score |
|---|---|
| **Classification Accuracy** | 82% |
| **Routing Accuracy** (simple vs complex path) | 90% |
| **Task Count Accuracy** | 84% |
| **Plan Validity Rate** | 100% |
| **Over-decomposition Rate** | 6% ↓ lower is better |

- Simple queries (e.g. *"What is FY2025 revenue?"*) correctly routed to **single-retrieval path** in 100% of cases — no unnecessary decomposition
- Comparison queries (e.g. *"Compare US and European strategies"*) correctly split into **4 parallel retrieval tasks** with no cross-dependencies
- Temporal queries (e.g. *"How did risk profile change from 2024 to 2026?"*) correctly generate **date-filtered tasks** using `effective_date_lte` and `is_current` SQL filters

### How to reproduce

```powershell
# Full phase comparison (Phase 1 vs Phase 3 vs Phase 4)
python scripts/compare_phases.py

# Run a specific experiment
python scripts/compare_phases.py --experiment 2   # Exact vs HNSW
python scripts/compare_phases.py --experiment 4   # Historical vs current version

# Phase 5 planning evaluation
python scripts/evaluate_planning.py

# Phase 3 retrieval benchmark
python scripts/evaluate_retrieval.py
```

## Project structure

```text
duediligence-ai/
├── data/
│   ├── raw/                  # Source PDFs
│   ├── index/                # Local indexes and chunk metadata
│   └── evaluation/           # Retrieval benchmark data
├── src/
│   ├── ingestion/            # Parsing, normalization, structure, chunking
│   ├── retrieval/             # Semantic, keyword, fusion, reranking
│   ├── db/                    # PostgreSQL connection, models, repositories
│   ├── embeddings.py          # Local embedding model
│   ├── vector_store.py        # FAISS index operations
│   ├── generator.py           # Prompt construction and answer generation
│   └── rag_pipeline.py        # End-to-end retrieval and generation
├── scripts/
│   ├── ingest.py
│   ├── inspect_document.py
│   ├── inspect_chunks.py
│   ├── inspect_retrieval.py
│   └── evaluate_retrieval.py
├── tests/
├── ui.py                     # Streamlit research workspace
├── app.py                    # Terminal CLI application
├── requirements.txt
├── .env.example
└── README.md
```

The exact repository may contain additional database migration files, tests, and utilities; those are omitted from the overview for readability.

## Design decisions

- **Keep retrieval inspectable.** Intermediate candidates and scores are available for debugging and error analysis.
- **Use lexical and semantic signals together.** They solve different retrieval problems and should be evaluated independently.
- **Rerank a candidate pool.** A reranker is a second-stage ranker, not a substitute for retrieval.
- **Keep structured data structured.** Dates, versions, identifiers, and financial values should not exist only inside unstructured text.
- **Preserve provenance.** Retrieved evidence should retain its source, version, page, section, and chunk identity.
- **Measure before optimizing.** Retrieval changes should be compared against the same labeled queries instead of being described as improvements without evidence.

## Limitations

- PDF layout and table extraction remain imperfect, especially for complex or merged cells.
- OCR is not guaranteed for scanned PDFs.
- Retrieval quality depends on document coverage, chunk quality, metadata accuracy, and the embedding/reranking models.
- Reranking cannot recover a relevant chunk that was omitted from the candidate pool.
- A retrieved passage is evidence, not proof that every generated claim is correct; source checking is still necessary for high-stakes decisions.
- The included company and corpus are fictional. The project demonstrates engineering patterns, not real investment advice or verified company research.

