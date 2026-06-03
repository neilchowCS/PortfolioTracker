"""Shared formatting helpers for the UI layer."""


def fmt_acct(name: str) -> str:
    """Pretty-print an account name: schwab_roth_ira → Schwab Roth Ira."""
    if not name:
        return ""
    return name.replace("_", " ").title()


def fmtd(val: float, decimals: int = 2, sign: bool = False) -> str:
    """Format a dollar amount with the sign before the $ symbol.
    -311.00 → -$311.00, not $-311.00.
    If sign=True, always show + or - prefix.
    """
    if sign:
        prefix = "+" if val >= 0 else "-"
    else:
        prefix = "-" if val < 0 else ""
    fmt = f",.{decimals}f"
    return f"{prefix}${abs(val):{fmt}}"
