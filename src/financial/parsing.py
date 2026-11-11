"""
src/financial/parsing.py — Deterministic value parser for financial strings.

Handles: "$61.2B", "76.0%", "$8.84", "+107%", "18,500", "$3.1B", "72%"
No LLM. Pure regex + Decimal.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import NamedTuple, Optional


class ParsedValue(NamedTuple):
    value:    Decimal
    unit:     str   # USD | PERCENT | USD_PER_SHARE | COUNT | MULTIPLE
    scale:    str   # UNITS | THOUSANDS | MILLIONS | BILLIONS
    original: str


_SCALE_KEYS = {
    "b": "BILLIONS", "bn": "BILLIONS", "billion": "BILLIONS",
    "m": "MILLIONS", "mn": "MILLIONS", "million": "MILLIONS",
    "k": "THOUSANDS", "thousand": "THOUSANDS",
}

_PERCENT_RE = re.compile(
    r"^([+-]?)\s*(\d[\d,]*)(?:\.(\d+))?\s*%$",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r"^\$?([+-]?)\s*(\d[\d,]*)(?:\.(\d+))?\s*(B|BN|Billion|M|MN|Million|K|Thousand)?\s*$",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"^([+-]?)\s*(\d[\d,]*)(?:\.(\d+))?\s*$",
)


def _make_decimal(sign: str, integer: str, frac: Optional[str]) -> Decimal:
    s = integer.replace(",", "")
    if frac:
        s += "." + frac
    if sign == "-":
        s = "-" + s
    return Decimal(s)


def parse_value(raw: str) -> Optional[ParsedValue]:
    """
    Parse a raw financial string.
    Returns None on failure — never raises.
    """
    s = raw.strip()
    if not s:
        return None

    # 1. Percent
    m = _PERCENT_RE.match(s)
    if m:
        sign, integer, frac = m.groups()
        try:
            val = _make_decimal(sign, integer, frac)
        except InvalidOperation:
            return None
        return ParsedValue(val, "PERCENT", "UNITS", raw)

    # 2. Currency / number with optional scale suffix
    m = _CURRENCY_RE.match(s)
    if m:
        sign, integer, frac, suffix = m.groups()
        try:
            val = _make_decimal(sign, integer, frac)
        except InvalidOperation:
            return None

        scale = "UNITS"
        if suffix:
            scale = _SCALE_KEYS.get(suffix.lower(), "UNITS")

        has_dollar = "$" in s
        if has_dollar:
            # Heuristic: USD_PER_SHARE when no scale suffix and small value with decimal
            if scale == "UNITS" and val < Decimal("1000") and "." in s:
                unit = "USD_PER_SHARE"
            else:
                unit = "USD"
        else:
            unit = "COUNT"

        return ParsedValue(val, unit, scale, raw)

    # 3. Plain integer/decimal count
    m = _COUNT_RE.match(s)
    if m:
        sign, integer, frac = m.groups()
        try:
            val = _make_decimal(sign, integer, frac)
        except InvalidOperation:
            return None
        return ParsedValue(val, "COUNT", "UNITS", raw)

    return None
