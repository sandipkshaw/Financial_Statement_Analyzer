"""Formatting helpers for monetary values shown in agent responses."""

from __future__ import annotations

import re


_CURRENCY_SYMBOLS = {
    "inr": "₹",
    "indian rupee": "₹",
    "rupee": "₹",
    "usd": "$",
    "us dollar": "$",
    "dollar": "$",
    "eur": "€",
    "euro": "€",
    "gbp": "£",
    "pound": "£",
}


def format_amount(value: float, unit: str = "", currency: str | None = None) -> str:
    """Format an amount already expressed in the unit declared by *unit*."""
    normalized = unit.casefold().replace(".", "").strip()
    symbol = _currency_symbol(currency or unit)

    # Trend observations use crore as their source unit, while dossiers can
    # describe it as "crore", "crores", or "cr".
    if symbol and re.search(r"\bcrores?\b|\bcr\b", normalized):
        absolute = abs(value)
        formatted = f"{absolute:,.2f}".rstrip("0").rstrip(".")
        sign = "-" if value < 0 else ""
        return f"{sign}{symbol}{formatted} crore"

    formatted = f"{abs(value):,.2f}".rstrip("0").rstrip(".")
    sign = "-" if value < 0 else ""
    return f"{sign}{symbol}{formatted}" if symbol else f"{sign}{formatted}"


def _currency_symbol(value: str) -> str:
    text = value.casefold().strip()
    for token in ("₹", "$", "€", "£"):
        if token in text:
            return token
    for name, symbol in _CURRENCY_SYMBOLS.items():
        if re.search(rf"\b{re.escape(name)}\b", text):
            return symbol
    return ""
