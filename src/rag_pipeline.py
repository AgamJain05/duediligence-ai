"""
rag_pipeline.py  ─  End-to-end RAG pipeline orchestrator
══════════════════════════════════════════════════════════
Ties together retrieval + generation into a single traceable call.
Returns a FULL TRACE dict so every intermediate step can be inspected.

ASCII Architecture Diagram
──────────────────────────

  ┌─────────────────────────────────────────────────────────────────┐
  │                  DueDiligence-AI  Phase 1                       │
  │                  RAG Pipeline Flow                              │
  └─────────────────────────────────────────────────────────────────┘

  USER QUESTION
       │
       ▼
  ┌─────────────────────────────┐
  │  embed_query(query)         │  embeddings.py
  │  BGE-base-en-v1.5 (local)  │  No API key needed
  │  → float32 vector (1×768)  │
  └─────────────┬───────────────┘
                │  query vector
                ▼
  ┌─────────────────────────────┐
  │  FAISS IndexFlatIP.search() │  vector_store.py
  │  Cosine similarity (exact)  │  data/index/faiss.index
  │  → [(idx, score), ...]      │
  └─────────────┬───────────────┘
                │  integer indices
                ▼
  ┌─────────────────────────────┐
  │  Chunk lookup               │  retriever.py
  │  chunks[idx]                │  data/index/chunks.json
  │  → [{text,source,page,...}] │
  └─────────────┬───────────────┘
                │  retrieved chunks (with text)
                ▼
  ┌─────────────────────────────┐
  │  build_prompt()             │  generator.py
  │  SYSTEM | CONTEXT | QUERY  │
  │  → message list             │
  └─────────────┬───────────────┘
                │  structured messages
                ▼
  ┌─────────────────────────────┐
  │  OpenRouter LLM             │  generator.py
  │  llama-3.3-70b (free)       │  OPENROUTER_API_KEY
  │  → answer string            │
  └─────────────┬───────────────┘
                │
                ▼
  ANSWER  +  FULL TRACE  (returned to app.py for display)

Pipeline position: called by app.py for every user question.
Phase 1: no query rewriting, no reranking, no conversation history.
"""

from __future__ import annotations

from typing import List, Dict

import faiss

from src.retriever import retrieve
from src.generator import answer_query


def run_pipeline(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    top_k: int = 5,
) -> Dict:
    """
    Receives : query          — raw user question string
               index          — loaded FAISS index (from vector_store.load_index)
               chunks         — loaded chunk store (from retriever.load_chunks)
               top_k          — number of chunks to retrieve
    Returns  : trace dict:
               {
                 "query":            str  — the original question
                 "retrieved_chunks": list — chunks found by retriever
                 "prompt_messages":  list — exact messages sent to LLM
                 "answer":           str  — LLM response
                 "model_used":       str  — which LLM was used
               }

    Why return the full trace? This is Phase 1 — the purpose is to
    understand RAG mechanics.  Every step of the pipeline is visible
    in the returned dict.  app.py displays all of it.

    No magic, no abstraction layers.  You can follow the data from
    the raw query string all the way to the final answer.
    """
    # ── Step 1 & 2: Embed query + vector search ────────────────────────────────
    retrieved_chunks = retrieve(query, index, chunks, top_k=top_k)

    # ── Edge case: no chunks retrieved ────────────────────────────────────────
    # This can happen if the index is empty or if the query is completely
    # out-of-distribution (no semantic similarity to any stored chunk).
    if not retrieved_chunks:
        return {
            "query":            query,
            "retrieved_chunks": [],
            "prompt_messages":  [],
            "answer":           (
                "No relevant context was found in the document store.\n"
                "Make sure you have run  `python scripts/ingest.py`  "
                "and that your PDFs contain text on the topic you are asking about."
            ),
            "model_used": "N/A (retrieval returned 0 results)",
        }

    # ── Step 3: Generate answer from retrieved context ────────────────────────
    generation = answer_query(query, retrieved_chunks)

    return {
        "query":            query,
        "retrieved_chunks": retrieved_chunks,
        "prompt_messages":  generation["prompt_messages"],
        "answer":           generation["answer"],
        "model_used":       generation["model_used"],
    }
