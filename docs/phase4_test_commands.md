# Phase 4 Test Commands

> Run all commands from the project root directory.

```bash
## Setup
alembic upgrade 0001
python scripts/ingest_pg.py

## Unit Tests
pytest tests/db/ -v
pytest tests/db/test_schema.py -v
pytest tests/db/test_versioning.py -v
pytest tests/db/test_exact_vs_hnsw.py -v
pytest tests/db/test_retrieval_db.py -v
pytest tests/db/test_financials.py -v
pytest tests/db/test_ingestion.py -v

## Inspect
python scripts/inspect_db.py
python scripts/inspect_chunks.py
python scripts/inspect_retrieval.py "What is NexaCore's FY2025 revenue?"

## Retrieval modes
# In Streamlit: select POSTGRES_EXACT, POSTGRES_HNSW, POSTGRES_FILTERED, POSTGRES_FTS, POSTGRES_HYBRID
streamlit run ui.py

## Benchmark Phase 4 vs Phase 1 and Phase 3
python scripts/compare_phases.py

## Latency comparison
# POSTGRES_HNSW expected ~279ms (2.2x faster than FAISS exact ~470ms)
python scripts/evaluate_retrieval.py
```
