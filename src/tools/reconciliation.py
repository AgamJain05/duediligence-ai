"""
src/tools/reconciliation.py — Reported vs derived value reconciliation.

Financial figures are rounded in documents. Recomputation from raw inputs
often differs by rounding. This utility classifies the difference and always
returns BOTH values — it never overwrites the reported figure.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict


def reconcile_derived(
    reported: Decimal,
    computed: Decimal,
    tolerance: Decimal,
    unit: str = "",
) -> Dict:
    """
    Compare reported value to computed value.

    Args:
        reported:  The value as stated in the source document.
        computed:  The value derived by our calculator.
        tolerance: Acceptable difference (absolute, in same unit).
        unit:      Unit string for display.

    Returns:
        {
          "reported": str,
          "computed": str,
          "difference": str,
          "tolerance": str,
          "classification": "WITHIN_TOLERANCE" | "OUTSIDE_TOLERANCE",
          "unit": str,
          "note": str,
        }

    Never overwrites reported with computed.
    OUTSIDE_TOLERANCE is surfaced as a warning, not an exception.
    """
    diff = abs(computed - reported)
    within = diff <= tolerance

    return {
        "reported":        str(reported),
        "computed":        str(computed),
        "difference":      str(diff),
        "tolerance":       str(tolerance),
        "classification":  "WITHIN_TOLERANCE" if within else "OUTSIDE_TOLERANCE",
        "unit":            unit,
        "note": (
            "Computed and reported values match within tolerance."
            if within else
            f"Computed {computed} differs from reported {reported} by {diff} "
            f"(tolerance={tolerance}). Using reported value."
        ),
    }
