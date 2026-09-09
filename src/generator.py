"""
generator.py  ─  LLM answer generation
═══════════════════════════════════════
Phase 1: Single-turn generation with retrieved context injected into prompt.

LLM backend: OpenRouter  (https://openrouter.ai)
  • OpenAI-compatible REST API  →  we use the openai Python library
  • Free tier: google/gemma-4-31b-it
      - 70B parameter model, strong instruction following
      - 131,072 token context window
      - ~50 RPD on free tier (no credit purchase needed)
  • Set OPENROUTER_API_KEY in .env (free signup at openrouter.ai)
  • Set OPENROUTER_MODEL to switch models without code changes

PROMPT STRUCTURE
────────────────
We use the standard chat messages format with THREE logical sections:

    messages[0]  role=system   →  SYSTEM INSTRUCTION
                                   (role, rules, citation requirement)
    messages[1]  role=user     →  RETRIEVED CONTEXT + USER QUESTION
                                   (injected evidence + the actual query)

Keeping system instruction and user content separate makes it easy to:
  • Audit what the model receives
  • Swap LLMs without changing logic
  • Extend to conversation history later (Phase 2+)

Pipeline position:
    [query, retrieved_chunks]  →  build_prompt()  →  [messages]
    [messages]                 →  answer_query()  →  {answer, prompt, model}
"""

from __future__ import annotations

import os
from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL       = "google/gemma-4-31b-it"
MAX_TOKENS          = 1024
TEMPERATURE         = 0.1   # Low = factual, high = creative; we want factual


def _get_client() -> OpenAI:
    """
    Receives : nothing  (reads OPENROUTER_API_KEY from environment)
    Returns  : configured openai.OpenAI client pointing at OpenRouter

    Why it exists: OpenRouter is OpenAI-API-compatible.  By pointing the
    openai library at a different base_url, we get free model access with
    zero code changes to the calling logic.

    Raises EnvironmentError if the API key is missing so the user gets a
    clear error message rather than a cryptic HTTP 401.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "\n[generator] OPENROUTER_API_KEY is not set.\n"
            "  1. Sign up free at https://openrouter.ai\n"
            "  2. Copy your key into .env:  OPENROUTER_API_KEY=sk-or-...\n"
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> List[Dict]:
    """
    Receives : user query string
               list of retrieved chunk dicts [{source, page_num, text, score}, ...]
    Returns  : list of message dicts in OpenAI chat format:
               [{"role": "system", "content": ...},
                {"role": "user",   "content": ...}]

    Why it exists: The LLM receives only what is in this prompt.
    Building the prompt explicitly (rather than via a framework abstraction)
    means you can inspect exactly what the model will see before it answers.

    Context injection: each retrieved chunk is formatted with its source
    and page number so the LLM can cite them in its answer.
    """
    # ── Build the context block from retrieved chunks ──────────────────────────
    context_parts: List[str] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        header = (
            f"[Source {i}] "
            f"File: {chunk['source']} | "
            f"Page: {chunk['page_num']} | "
            f"Similarity: {chunk['score']:.3f}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")

    context_block = "\n\n" + ("─" * 60 + "\n\n").join(context_parts)

    # ── System instruction ─────────────────────────────────────────────────────
    system_content = (
        "You are a meticulous financial and business analyst specializing in "
        "company due diligence reports.\n\n"
        "RULES:\n"
        "1. Answer the user's question using ONLY the context passages provided. "
        "Do NOT use any knowledge outside these passages.\n"
        "2. If the context does not contain sufficient information to answer the "
        "question, respond EXACTLY with: "
        "'The provided documents do not contain sufficient information to answer "
        "this question.'\n"
        "3. Always cite your sources. After each factual claim, note the file name "
        "and page number in parentheses, e.g. (OrionVault_Dossier.pdf, p.4).\n"
        "4. Do not invent numbers, names, dates, or any facts not present in the "
        "context.\n"
        "5. Be concise and structured. Use bullet points where appropriate."
    )

    # ── User message: context + question ──────────────────────────────────────
    user_content = (
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"{'═' * 60}\n\n"
        f"QUESTION: {query}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


def answer_query(
    query: str,
    retrieved_chunks: List[Dict],
    model: str | None = None,
    max_tokens: int = MAX_TOKENS,
) -> Dict:
    """
    Receives : user query string
               retrieved_chunks from retriever.py
               optional model override (falls back to OPENROUTER_MODEL env var)
               max_tokens for the LLM response
    Returns  : dict:
               {
                 "answer":          str   — the LLM's answer
                 "prompt_messages": list  — the exact messages sent to the LLM
                 "model_used":      str   — which model was called
               }

    Why it exists: This is the "G" in RAG — generation.
    It takes the retrieved evidence and asks the LLM to synthesize an answer.

    Returning prompt_messages alongside the answer is intentional:
    app.py can display the prompt for educational inspection of the pipeline.

    API errors are caught and returned as error strings rather than crashing
    the REPL, so the user can fix the issue and ask again.
    """
    model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    prompt_messages = build_prompt(query, retrieved_chunks)

    try:
        client   = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
        )
        answer = response.choices[0].message.content.strip()

    except EnvironmentError as e:
        # Missing API key — give a friendly error
        answer = str(e)

    except Exception as e:
        # Network errors, model errors, rate limits, etc.
        answer = (
            f"[ERROR] LLM generation failed: {type(e).__name__}: {e}\n\n"
            "  Check that your OPENROUTER_API_KEY is valid and that the model "
            f"'{model}' is available on your OpenRouter account."
        )

    return {
        "answer":          answer,
        "prompt_messages": prompt_messages,
        "model_used":      model,
    }
