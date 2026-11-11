"""
src/tools/registry.py — Tool registry.

All Phase 6 tools register here. The planner uses the registry to validate
that a tool name exists before creating a plan task.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.tools.base import BaseTool

_REGISTRY: Dict[str, BaseTool] = {}


def register(tool: BaseTool) -> None:
    _REGISTRY[tool.name] = tool


def get_tool(name: str) -> Optional[BaseTool]:
    return _REGISTRY.get(name)


def list_tools() -> List[str]:
    return list(_REGISTRY.keys())


def get_schema(name: str) -> Optional[Dict]:
    """Return a simple description dict for a tool (used by inspect_tools.py)."""
    tool = _REGISTRY.get(name)
    if tool is None:
        return None
    return {
        "name":        tool.name,
        "description": tool.description,
    }


def build_default_registry(db_session=None, faiss_retriever=None) -> None:
    """
    Instantiate and register all Phase 6 tools.
    Call once at application startup.

    Args:
        db_session:      SQLAlchemy session for DB-backed tools (financial_data, source_retrieval).
        faiss_retriever: HybridRetriever instance for document_search FAISS fallback.
    """
    from src.tools.calculator import CalculatorTool
    from src.tools.document_search import DocumentSearchTool
    from src.tools.financial_data import FinancialDataTool
    from src.tools.period_compare import PeriodCompareTool
    from src.tools.source_retrieval import SourceRetrievalTool

    register(CalculatorTool())
    register(DocumentSearchTool(faiss_retriever=faiss_retriever, db_session=db_session))
    register(FinancialDataTool(db_session=db_session))
    register(PeriodCompareTool(db_session=db_session))
    register(SourceRetrievalTool(db_session=db_session))
