"""
app.py  ─  Interactive CLI for the DueDiligence-AI RAG system
═══════════════════════════════════════════════════════════════
Usage:
    cd duediligence-ai
    python app.py

The REPL (Read-Eval-Print Loop) loads the FAISS index and chunk store ONCE
at startup, then answers repeated questions without restarting.

For every question it displays the FULL pipeline trace:
  1. Original question
  2. Retrieved chunks  (text preview, source, page, similarity score)
  3. Prompt summary    (system instruction + model used)
  4. Generated answer
  5. Source citations

This verbose display is intentional — Phase 1 is about understanding
what the retriever finds and what the LLM actually receives.

Type  'quit' | 'exit' | 'q'  to stop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env before any src imports that read env vars ───────────────────────
load_dotenv()

# ── Ensure project root is on sys.path so `src` is importable ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import load_index
from src.retriever    import load_chunks
from src.rag_pipeline import run_pipeline

# ── Configuration ─────────────────────────────────────────────────────────────
TOP_K     = int(os.getenv("TOP_K", "5"))
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# ── Display helpers ────────────────────────────────────────────────────────────

def _sep(char: str = "-", width: int = 64) -> None:
    """Print a horizontal separator line."""
    print(char * width)


def _banner() -> None:
    print()
    print("+" + "=" * 62 + "+")
    print("|    DueDiligence-AI  |  Phase 1: RAG Foundations              |")
    print("|    Embeddings: BAAI/bge-base-en-v1.5  (local, free)          |")
    print("|    LLM:        OpenRouter  (google/gemma-4-31b-it)           |")
    print("+" + "=" * 62 + "+")
    print()


def display_results(result: dict) -> None:
    """
    Receives : pipeline trace dict from run_pipeline()
    Returns  : nothing  (prints full trace to stdout)

    Why verbose? Learning RAG requires seeing what the retriever found,
    what the prompt contained, and how those affected the final answer.
    """
    query   = result["query"]
    chunks  = result["retrieved_chunks"]
    answer  = result["answer"]
    model   = result["model_used"]
    prompt  = result.get("prompt_messages", [])

    # ── 1. Question ────────────────────────────────────────────────────────────
    _sep("=")
    print(f"QUESTION:  {query}")
    _sep()

    # ── 2. Retrieved chunks ────────────────────────────────────────────────────
    print(f"\nRETRIEVED CONTEXT  ({len(chunks)} chunk(s), top_k={TOP_K}):\n")

    if not chunks:
        print("  [none - the retriever found no matching chunks]\n")
    else:
        for i, chunk in enumerate(chunks, start=1):
            # Show first 350 characters of chunk text as a preview
            preview = chunk["text"][:350].replace("\n", " ").strip()
            ellipsis = "..." if len(chunk["text"]) > 350 else ""

            print(f"  [{i}]")
            print(f"       source  = {chunk['source']}")
            print(f"       page    = {chunk['page_num']}")
            print(f"       score   = {chunk['score']:.4f}  (cosine similarity)")
            print(f"       id      = {chunk['chunk_id']}")
            print(f"       text    = {preview}{ellipsis}")
            print()

    # ── 3. Prompt summary ──────────────────────────────────────────────────────
    _sep()
    print("\nPROMPT SENT TO LLM:\n")
    if prompt:
        sys_msg = prompt[0]["content"]
        # Show first 300 chars of system message
        print(f"  [system]  {sys_msg[:300]}...")
        print(f"\n  [user]    <retrieved context + question>")
        print(f"             Context chunks: {len(chunks)}")
        print(f"             Model: {model}")
    else:
        print("  [no prompt - pipeline short-circuited before generation]")

    # ── 4. Answer ──────────────────────────────────────────────────────────────
    _sep()
    print(f"\nANSWER:\n")
    print(answer)
    print()

    # ── 5. Source citations ────────────────────────────────────────────────────
    if chunks:
        _sep()
        print("\nSOURCES:\n")
        seen: set = set()
        for chunk in chunks:
            key = (chunk["source"], chunk["page_num"])
            if key not in seen:
                print(f"  *  {chunk['source']}   (page {chunk['page_num']})")
                seen.add(key)
        print()

    _sep("=")
    print()


def main() -> None:
    _banner()

    # ── Load index + chunks once at startup ────────────────────────────────────
    try:
        print(f"[startup] Loading FAISS index and chunk store from {INDEX_DIR}...")
        index  = load_index(INDEX_DIR)
        chunks = load_chunks(INDEX_DIR)
        print(f"[startup] Ready - {index.ntotal} vectors, {len(chunks)} chunks.\n")
    except FileNotFoundError as exc:
        print(f"\n[ERROR]{exc}")
        sys.exit(1)

    print("Type your question and press Enter.")
    print("Type  'quit'  or  'exit'  to stop.\n")

    # ── REPL loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        short = (query[:60] + "…") if len(query) > 60 else query
        print(f"\n[pipeline] Running for: '{short}'\n")

        result = run_pipeline(query, index, chunks, top_k=TOP_K)
        display_results(result)


if __name__ == "__main__":
    main()
