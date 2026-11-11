#!/usr/bin/env python3
"""
scripts/inspect_tools.py — List tools and optionally invoke them.

Usage:
    python scripts/inspect_tools.py                          # list all tools
    python scripts/inspect_tools.py --tool calculate_metric  # show tool details
    python scripts/inspect_tools.py --tool calculate_metric --args '{"operation":"percent_change","inputs":[{"label":"old","value":"29.5","unit":"USD"},{"label":"new","value":"61.2","unit":"USD"}]}'
    python scripts/inspect_tools.py --tool search_financial_data --args '{"company":"NexaCore","metric":"revenue","fiscal_year":2025}'
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Inspect and invoke Phase 6 tools")
    parser.add_argument("--tool",  default=None, help="Tool name")
    parser.add_argument("--args",  default=None, help="JSON args for tool invocation")
    args = parser.parse_args()

    try:
        from src.db.connection import SessionLocal
        db_session = SessionLocal()
    except Exception:
        db_session = None

    try:
        faiss_retriever = None
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.models import HYBRID_RERANKED
        import os
        index_dir = str(Path(__file__).resolve().parent.parent / "data" / "index")
        if (Path(index_dir) / "faiss.index").exists():
            faiss_retriever = HybridRetriever(mode=HYBRID_RERANKED, index_dir=index_dir, final_top_k=5)
    except Exception:
        pass

    from src.tools.registry import build_default_registry, list_tools, get_tool, get_schema
    build_default_registry(db_session=db_session, faiss_retriever=faiss_retriever)

    if args.tool is None:
        print("Registered tools:")
        for name in list_tools():
            schema = get_schema(name)
            print(f"\n  {name}")
            print(f"    {schema['description'][:120]}")
        return

    tool = get_tool(args.tool)
    if tool is None:
        print(f"Tool not found: {args.tool}")
        sys.exit(1)

    schema = get_schema(args.tool)
    print(f"Tool: {schema['name']}")
    print(f"Description: {schema['description']}")

    if args.args:
        kwargs = json.loads(args.args)
        print(f"\nInvoking with: {kwargs}")
        result = tool.safe_run(**kwargs)
        print(f"\nStatus: {result.status}")
        if result.error:
            print(f"Error: {result.error}")
        print(f"Data: {json.dumps(result.model_dump_json_safe()['data'], indent=2)}")
        if result.provenance:
            print(f"Provenance: {len(result.provenance)} records")
        print(f"Latency: {result.latency_ms:.1f}ms")

    if db_session:
        db_session.close()


if __name__ == "__main__":
    main()
