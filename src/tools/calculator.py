"""
src/tools/calculator.py — Deterministic calculator using decimal.Decimal.

WHITELISTED OPERATIONS ONLY. No eval(), no exec(), no dynamic code.
Every computation returns its formula string and full input provenance.
The LLM must NEVER perform arithmetic — it calls this tool instead.
"""
from __future__ import annotations

import math
from decimal import Decimal, DivisionByZero, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool, ToolError, ToolResult, Provenance
from src.tools.reconciliation import reconcile_derived

SUPPORTED_OPERATIONS = [
    "percent_change", "difference", "ratio", "margin",
    "share_of_total", "cagr", "sum", "multiple", "basis_point_change",
]

_TWO_DP = Decimal("0.01")
_FOUR_DP = Decimal("0.0001")


def _d(v: Any) -> Decimal:
    """Convert to Decimal. Raises InvalidOperation on failure."""
    if isinstance(v, Decimal):
        return v
    return Decimal(str(v))


class CalculatorTool(BaseTool):
    name = "calculate_metric"
    description = (
        "Deterministic arithmetic for financial metrics using Decimal precision. "
        "Supported operations: percent_change, difference, ratio, margin, "
        "share_of_total, cagr, sum, multiple, basis_point_change. "
        "Returns formula, inputs with provenance, and result. "
        "NEVER use the LLM to compute numbers — always call this tool."
    )

    def run(
        self,
        operation: str,
        inputs: List[Dict[str, Any]],
        reported_value: Optional[str] = None,
        tolerance: Optional[str] = None,
    ) -> ToolResult:
        """
        Args:
            operation:      One of SUPPORTED_OPERATIONS.
            inputs:         List of {"label": str, "value": str/Decimal, "unit": str, "provenance": dict}.
            reported_value: Optional reported figure to reconcile against computed result.
            tolerance:      Optional tolerance for reconciliation (e.g. "0.15" for 0.15 pp).
        """
        if operation not in SUPPORTED_OPERATIONS:
            return ToolResult(
                tool=self.name,
                status="ERROR",
                error=ToolError(
                    code="UNSUPPORTED_OPERATION",
                    message=f"Operation '{operation}' not supported. Use one of: {SUPPORTED_OPERATIONS}",
                ),
            )

        # Parse and validate inputs
        parsed_inputs = []
        provenance_list = []
        for inp in inputs:
            try:
                val = _d(inp["value"])
            except (InvalidOperation, KeyError) as e:
                return ToolResult(
                    tool=self.name, status="ERROR",
                    error=ToolError(code="INVALID_INPUT", message=f"Cannot parse input value: {inp}. {e}"),
                )
            parsed_inputs.append({
                "label": inp.get("label", ""),
                "value": val,
                "unit":  inp.get("unit", ""),
            })
            if inp.get("provenance"):
                try:
                    provenance_list.append(Provenance(**inp["provenance"]))
                except Exception:
                    provenance_list.append(Provenance(source_type="computation"))

        try:
            result_val, formula, result_unit = _dispatch(operation, parsed_inputs)
        except DivisionByZero:
            return ToolResult(
                tool=self.name, status="ERROR",
                error=ToolError(code="DIVISION_BY_ZERO", message="Division by zero in calculation."),
            )
        except ValueError as e:
            return ToolResult(
                tool=self.name, status="ERROR",
                error=ToolError(code="CALCULATION_ERROR", message=str(e)),
            )

        rounded = result_val.quantize(_TWO_DP, rounding=ROUND_HALF_UP)

        # Reconciliation check
        reconciliation = None
        if reported_value is not None:
            tol = Decimal(str(tolerance)) if tolerance else Decimal("0.15")
            try:
                reported_d = _d(reported_value)
                reconciliation = reconcile_derived(reported_d, rounded, tol, result_unit)
            except Exception:
                pass

        data: Dict[str, Any] = {
            "operation": operation,
            "inputs":    [{"label": p["label"], "value": str(p["value"]), "unit": p["unit"]}
                          for p in parsed_inputs],
            "formula":   formula,
            "result":    str(rounded),
            "unit":      result_unit,
            "rounding":  "2dp",
        }
        if reconciliation:
            data["reconciliation"] = reconciliation

        return ToolResult(
            tool=self.name,
            status="SUCCESS",
            data=data,
            provenance=provenance_list,
        )


def _dispatch(operation: str, inputs: List[Dict]) -> tuple[Decimal, str, str]:
    """Route to the correct arithmetic function. Returns (result, formula_str, unit)."""
    vals = [inp["value"] for inp in inputs]
    labels = [inp.get("label", f"v{i}") for i, inp in enumerate(inputs)]
    units = [inp.get("unit", "") for inp in inputs]

    if operation == "percent_change":
        if len(vals) < 2:
            raise ValueError("percent_change requires 2 inputs: old, new")
        old, new = vals[0], vals[1]
        if old == 0:
            raise DivisionByZero()
        result = (new - old) / old * Decimal("100")
        formula = f"({new} - {old}) / {old} * 100"
        return result, formula, "PERCENT"

    if operation == "difference":
        if len(vals) < 2:
            raise ValueError("difference requires 2 inputs: a, b")
        result = vals[1] - vals[0]
        formula = f"{vals[1]} - {vals[0]}"
        return result, formula, units[1] if units[1] else units[0]

    if operation == "ratio":
        if len(vals) < 2:
            raise ValueError("ratio requires 2 inputs: numerator, denominator")
        if vals[1] == 0:
            raise DivisionByZero()
        result = vals[0] / vals[1]
        formula = f"{vals[0]} / {vals[1]}"
        return result, formula, "MULTIPLE"

    if operation == "margin":
        if len(vals) < 2:
            raise ValueError("margin requires 2 inputs: numerator, denominator")
        if vals[1] == 0:
            raise DivisionByZero()
        result = vals[0] / vals[1] * Decimal("100")
        formula = f"({vals[0]} / {vals[1]}) * 100"
        return result, formula, "PERCENT"

    if operation == "share_of_total":
        if len(vals) < 2:
            raise ValueError("share_of_total requires 2 inputs: part, total")
        if vals[1] == 0:
            raise DivisionByZero()
        result = vals[0] / vals[1] * Decimal("100")
        formula = f"({vals[0]} / {vals[1]}) * 100"
        return result, formula, "PERCENT"

    if operation == "cagr":
        if len(vals) < 2:
            raise ValueError("cagr requires inputs: start_value, end_value; and years in a 3rd input")
        if len(inputs) < 3:
            raise ValueError("cagr requires 3 inputs: start, end, years")
        start, end, years = vals[0], vals[1], vals[2]
        if start <= 0:
            raise ValueError("CAGR: start value must be positive")
        if years <= 0:
            raise ValueError("CAGR: years must be positive")
        # CAGR = (end/start)^(1/years) - 1
        ratio = float(end / start)
        cagr_float = (ratio ** (1 / float(years)) - 1) * 100
        result = Decimal(str(round(cagr_float, 6)))
        formula = f"((({end}/{start})^(1/{years})) - 1) * 100"
        return result, formula, "PERCENT"

    if operation == "sum":
        result = sum(vals, Decimal("0"))
        formula = " + ".join(str(v) for v in vals)
        return result, formula, units[0] if units else ""

    if operation == "multiple":
        if len(vals) < 2:
            raise ValueError("multiple requires 2 inputs: a, b")
        if vals[1] == 0:
            raise DivisionByZero()
        result = vals[0] / vals[1]
        formula = f"{vals[0]} / {vals[1]}"
        return result, formula, "MULTIPLE"

    if operation == "basis_point_change":
        if len(vals) < 2:
            raise ValueError("basis_point_change requires 2 inputs: old_pct, new_pct")
        result = (vals[1] - vals[0]) * Decimal("100")
        formula = f"({vals[1]} - {vals[0]}) * 100"
        return result, formula, "BASIS_POINTS"

    raise ValueError(f"Unknown operation: {operation}")
