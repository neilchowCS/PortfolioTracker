"""CSV import parsers for Schwab, Fidelity, and E*Trade brokerage exports.

Each parser returns:
    rows: list[dict]  — parsed transactions (same schema as TRANSACTION_COLUMNS minus id)
    skipped: list[dict]  — rows that could not be mapped, with file/line info

Account name is derived from the CSV filename (without extension).

Transfer type:
    A "transfer" transaction records shares arriving in an account with no
    corresponding cash outflow.  The market value (price × shares) is stored
    in `amount` so the portfolio can count it as a contribution for net-worth
    purposes while also tracking the shares.
"""

from __future__ import annotations

import csv
import io
import re
import os
from datetime import datetime

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(raw: str) -> str | None:
    """Try common date formats, return ISO string or None."""
    raw = raw.strip()
    # Strip " as of ..." suffix (Schwab uses this)
    raw = re.split(r"\s+as\s+of\s+", raw, flags=re.IGNORECASE)[0].strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _parse_float(raw: str) -> float:
    """Parse a dollar/number string, stripping $, commas, quotes, parens."""
    if not raw:
        return 0.0
    raw = raw.replace("$", "").replace(",", "").replace('"', "").strip()
    if not raw or raw == "--":
        return 0.0
    # Handle accounting-style negatives: (1234.56) → -1234.56
    neg = False
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
        neg = True
    try:
        val = float(raw)
        return -val if neg else val
    except ValueError:
        return 0.0


def _account_from_filename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _is_junk_row(raw: dict) -> bool:
    """Return True if a row looks like a disclaimer, footer, or empty line.
    Catches: completely empty rows, rows whose text is clearly boilerplate
    legalese, and rows where a single field contains a long paragraph (>80
    chars) which is never a real transaction.
    """
    vals = [str(v).strip() for v in raw.values() if v]
    non_empty = [v for v in vals if v]
    if not non_empty:
        return True
    # A single field with a long paragraph is a disclaimer, not a transaction
    if len(non_empty) <= 2 and max(len(v) for v in non_empty) > 80:
        return True
    joined = " ".join(non_empty).lower()
    junk_phrases = ("informational purposes", "member sipc",
                    "data and information", "date downloaded",
                    "not intended", "insurance products",
                    "morgan stanley", "withholding taxes",
                    "bank deposit program", "official account statements")
    return any(p in joined for p in junk_phrases)


# Money-market / cash-equivalent tickers — treated as cash, not stock positions
_CASH_TICKERS = {"SWVXX", "FDRXX", "SPAXX", "FCASH", "CORE"}


# ── Schwab ────────────────────────────────────────────────────────────────────

_SCHWAB_ACTION_MAP = {
    "buy":               "buy",
    "sell":              "sell",
    "reinvest shares":   "buy",
    "stock split":       "split",
    "moneylink transfer":  "contribution",
    "funds received":      "contribution",
    "promotional award":   "contribution",
    "journal":             "contribution",
    "credit interest":     "contribution",
    "cash dividend":       "contribution",
    "qualified dividend":  "contribution",
    "qual div reinvest":   "contribution",
    "adr mgmt fee":        "withdrawal",
    "foreign tax paid":    "withdrawal",
}


def parse_schwab(filepath: str) -> tuple[list[dict], list[dict]]:
    account = _account_from_filename(filepath)
    rows: list[dict] = []
    skipped: list[dict] = []

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for lineno, raw in enumerate(reader, start=2):
            action_raw = (raw.get("Action") or "").strip().lower()
            date_str = _parse_date(raw.get("Date", ""))
            if not date_str:
                if not _is_junk_row(raw):
                    skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                    "reason": f"Bad date: {raw.get('Date','')}", "raw": dict(raw)})
                continue

            mapped = _SCHWAB_ACTION_MAP.get(action_raw)

            # Security Transfer — ignore (user must manually enter)
            if action_raw == "security transfer":
                skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                "reason": "Security transfer (no cost basis) — enter manually",
                                "raw": dict(raw)})
                continue

            if mapped is None:
                if not _is_junk_row(raw):
                    skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                    "reason": f"Unknown action: {raw.get('Action','')}",
                                    "raw": dict(raw)})
                continue

            ticker = (raw.get("Symbol") or "").strip().upper()
            qty = abs(_parse_float(raw.get("Quantity", "")))
            price = abs(_parse_float(raw.get("Price", "")))
            amount = abs(_parse_float(raw.get("Amount", "")))

            if mapped == "split":
                # Schwab: Quantity = new shares added, need to parse ratio from Description
                desc = raw.get("Description", "")
                # Format: "VANGUARD INFO TECH ETF SPLIT RATIO  8:1"  — not available,
                # but Quantity is the net new shares and Price is the post-split price.
                # We store shares=qty (the added shares count) — but we need old:new.
                # Schwab doesn't give ratio directly. We'll parse from description if possible.
                # For now store as-is; the user will see it and can edit.
                # We'll try to extract ratio from description
                ratio_match = re.search(r"(\d+):(\d+)", desc)
                if ratio_match:
                    split_new = float(ratio_match.group(1))
                    split_old = float(ratio_match.group(2))
                else:
                    split_new = qty
                    split_old = 1.0
                rows.append({
                    "type": "split", "ticker": ticker,
                    "price": split_old, "shares": split_new,
                    "amount": 0, "date": date_str, "account": account,
                })
                continue

            if mapped in ("buy", "sell"):
                if not ticker or qty == 0:
                    skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                    "reason": f"Missing ticker/qty for {mapped}", "raw": dict(raw)})
                    continue
                # Money-market tickers → silently skip (just cash moving to/from
                # money market, not new money in or out).  Interest/dividends
                # from these tickers are still captured via the dividend action.
                if ticker in _CASH_TICKERS:
                    continue
                rows.append({
                    "type": mapped, "ticker": ticker,
                    "price": price, "shares": qty,
                    "amount": round(price * qty, 4),
                    "date": date_str, "account": account,
                })
            elif mapped in ("contribution", "withdrawal"):
                if amount == 0:
                    continue  # skip zero-amount rows
                rows.append({
                    "type": mapped, "ticker": "", "price": 0,
                    "shares": 0, "amount": amount,
                    "date": date_str, "account": account,
                })

    return rows, skipped


# ── E*Trade ───────────────────────────────────────────────────────────────────

_ETRADE_ACTION_MAP = {
    "bought":             "buy",
    "sold":               "sell",
    "stock split":        "split",
    "contribution":       "contribution",
    "conversion":         "contribution",
    "interest income":    "contribution",
    "dividend":           "contribution",
    "qualified dividend": "contribution",
    "online transfer":    "_online_transfer",  # check sign
}


def parse_etrade(filepath: str) -> tuple[list[dict], list[dict]]:
    account = _account_from_filename(filepath)
    rows: list[dict] = []
    skipped: list[dict] = []

    # E*Trade CSVs have header junk — find the real header row
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Activity/Trade Date"):
            header_idx = i
            break
    if header_idx is None:
        return [], [{"file": os.path.basename(filepath), "line": 0,
                     "reason": "Could not find E*Trade header row", "raw": {}}]

    reader = csv.DictReader(lines[header_idx:])
    for lineno, raw in enumerate(reader, start=header_idx + 2):
        action_raw = (raw.get("Activity Type") or "").strip().lower()
        date_str = _parse_date(raw.get("Activity/Trade Date", ""))
        if not date_str:
            if not _is_junk_row(raw):
                skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                "reason": f"Bad date: {raw.get('Activity/Trade Date','')}",
                                "raw": dict(raw)})
            continue

        ticker = (raw.get("Symbol") or "").strip().upper()
        if ticker == "--":
            ticker = ""

        # Transfer of stock — ignore per user request (cash transfers are OK)
        if action_raw == "transfer":
            if ticker:
                skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                "reason": f"Stock transfer ({ticker}) — enter manually",
                                "raw": dict(raw)})
                continue
            # Cash transfer with no ticker — skip as unrecognized
            skipped.append({"file": os.path.basename(filepath), "line": lineno,
                            "reason": f"Transfer without ticker", "raw": dict(raw)})
            continue

        mapped = _ETRADE_ACTION_MAP.get(action_raw)
        if mapped is None:
            if not action_raw or _is_junk_row(raw):
                continue
            skipped.append({"file": os.path.basename(filepath), "line": lineno,
                            "reason": f"Unknown action: {raw.get('Activity Type','')}",
                            "raw": dict(raw)})
            continue

        qty = abs(_parse_float(raw.get("Quantity #", "")))
        price_val = abs(_parse_float(raw.get("Price $", "")))
        amount = _parse_float(raw.get("Amount $", ""))

        if mapped == "split":
            desc = raw.get("Description", "")
            ratio_match = re.search(r"(\d+):(\d+)", desc)
            if ratio_match:
                split_new = float(ratio_match.group(1))
                split_old = float(ratio_match.group(2))
            else:
                split_new = qty
                split_old = 1.0
            rows.append({
                "type": "split", "ticker": ticker,
                "price": split_old, "shares": split_new,
                "amount": 0, "date": date_str, "account": account,
            })
            continue

        if mapped in ("buy", "sell"):
            if not ticker or qty == 0:
                skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                "reason": f"Missing ticker/qty for {mapped}", "raw": dict(raw)})
                continue
            if ticker in _CASH_TICKERS:
                continue
            rows.append({
                "type": mapped, "ticker": ticker,
                "price": price_val, "shares": qty,
                "amount": round(price_val * qty, 4),
                "date": date_str, "account": account,
            })
        elif mapped == "_online_transfer":
            # Positive amount = contribution, negative = withdrawal
            if amount == 0:
                continue
            txn_type = "contribution" if amount > 0 else "withdrawal"
            rows.append({
                "type": txn_type, "ticker": "", "price": 0,
                "shares": 0, "amount": abs(amount),
                "date": date_str, "account": account,
            })
        elif mapped in ("contribution", "withdrawal"):
            amt = abs(amount)
            if amt == 0:
                continue
            rows.append({
                "type": mapped, "ticker": "", "price": 0,
                "shares": 0, "amount": amt,
                "date": date_str, "account": account,
            })

    return rows, skipped


# ── Fidelity ─────────────────────────────────────────────────────────────────

# Fidelity type 2 funds — non-searchable
# Funds treated as cash (just contribution/withdrawal, no ticker)
_FIDELITY_T2_CASH_FUNDS = {"STABLE VALUE", "BROKERAGELINK"}
# Funds whose growth tracks VOO — stored with original ticker/shares,
# portfolio calculations use VOO’s growth rate to determine current value.
VOO_PROXY_FUNDS = {"LIFEPATH IDX 2065 F"}
VOO_PROXY_TICKER = "VOO"

_FIDELITY_T2_ACTION_MAP = {
    "contributions":         "contribution_buy",
    "exchange in":           "buy",
    "exchange out":          "sell",
    "withdrawals":           "withdrawal",
    "change in market value": None,  # skip — growth handled via proxy price
    "transfers":             None,   # skip zero-value transfers
}


def _parse_fidelity_type1(reader, filepath: str, account: str,
                          start_line: int) -> tuple[list[dict], list[dict]]:
    """Parse Fidelity brokerage-style rows (Type 1)."""
    rows: list[dict] = []
    skipped: list[dict] = []

    for lineno, raw in enumerate(reader, start=start_line):
        date_str = _parse_date(raw.get("Run Date", ""))
        if not date_str:
            if not any(v.strip() for v in raw.values() if v):
                continue
            # Footer / disclaimer — stop parsing
            break

        action_raw = (raw.get("Action") or "").strip()
        ticker = (raw.get("Symbol") or "").strip().upper()
        price_val = abs(_parse_float(raw.get("Price ($)", "")))
        qty = abs(_parse_float(raw.get("Quantity", "")))
        amount = _parse_float(raw.get("Amount ($)", ""))

        action_lower = action_raw.lower()

        # Cash transfers / rollovers — treat as contribution
        if "transferred" in action_lower or "rollover" in action_lower:
            if abs(amount) < 0.01:
                continue
            txn_type = "contribution" if amount > 0 else "withdrawal"
            rows.append({
                "type": txn_type, "ticker": "", "price": 0,
                "shares": 0, "amount": abs(amount),
                "date": date_str, "account": account,
            })
            continue

        # Dividend / reinvestment → contribution + buy
        if "dividend" in action_lower:
            if abs(amount) < 0.01:
                continue
            rows.append({
                "type": "contribution", "ticker": "", "price": 0,
                "shares": 0, "amount": abs(amount),
                "date": date_str, "account": account,
            })
            continue

        if "reinvestment" in action_lower:
            # Cash-equivalent reinvestment (e.g. FDRXX) — the dividend row
            # already captured the income as a contribution; the reinvestment
            # is just buying more money-market shares.  Skip entirely.
            if ticker in _CASH_TICKERS:
                continue
            if ticker and qty > 0 and price_val > 0:
                amt = round(price_val * qty, 4)
                rows.append({
                    "type": "contribution", "ticker": "", "price": 0,
                    "shares": 0, "amount": amt,
                    "date": date_str, "account": account,
                })
                rows.append({
                    "type": "buy", "ticker": ticker,
                    "price": price_val, "shares": qty,
                    "amount": amt,
                    "date": date_str, "account": account,
                })
            continue

        # Buy/Sell
        if "bought" in action_lower:
            if not ticker or qty == 0:
                skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                "reason": f"Missing ticker/qty for buy", "raw": dict(raw)})
                continue
            if ticker in _CASH_TICKERS:
                continue
            rows.append({
                "type": "buy", "ticker": ticker,
                "price": price_val, "shares": qty,
                "amount": abs(amount) if amount else round(price_val * qty, 4),
                "date": date_str, "account": account,
            })
            continue

        if "sold" in action_lower:
            if not ticker or qty == 0:
                skipped.append({"file": os.path.basename(filepath), "line": lineno,
                                "reason": f"Missing ticker/qty for sell", "raw": dict(raw)})
                continue
            if ticker in _CASH_TICKERS:
                continue
            rows.append({
                "type": "sell", "ticker": ticker,
                "price": price_val, "shares": qty,
                "amount": abs(amount) if amount else round(price_val * qty, 4),
                "date": date_str, "account": account,
            })
            continue

        # Unrecognized
        if not _is_junk_row(raw):
            skipped.append({"file": os.path.basename(filepath), "line": lineno,
                            "reason": f"Unknown action: {action_raw}",
                            "raw": dict(raw)})

    return rows, skipped


def _parse_fidelity_type2(reader, filepath: str, account: str,
                          start_line: int) -> tuple[list[dict], list[dict]]:
    """Parse Fidelity 401k-style rows (Type 2).
    Cash-like funds (STABLE VALUE, BROKERAGELINK) → contribution/withdrawal only.
    Other non-searchable funds → proxied to VOO.
    """
    rows: list[dict] = []
    skipped: list[dict] = []

    for lineno, raw in enumerate(reader, start=start_line):
        date_str = _parse_date(raw.get("Date", ""))
        if not date_str:
            if not any(v.strip() for v in raw.values() if v):
                continue
            break

        fund = (raw.get("Investment") or "").strip().upper()
        action_raw = (raw.get("Transaction Type") or "").strip().lower()
        amount = _parse_float(raw.get("Amount ($)", ""))
        shares_units = abs(_parse_float(raw.get("Shares/Unit", "")))

        mapped = _FIDELITY_T2_ACTION_MAP.get(action_raw)
        if mapped is None:
            # Explicitly skipped types (market value changes, zero transfers)
            continue

        is_voo_fund = fund in VOO_PROXY_FUNDS

        # Non-VOO funds (STABLE VALUE, BROKERAGELINK, etc.):
        # These are pass-through cash funds.  Money flows through them
        # to the brokerage (Type 1) where it is captured as rollover /
        # transfer contributions.  Skip everything for these funds so
        # the 401k account only tracks the LIFEPATH investment position.
        if not is_voo_fund:
            continue

        # Investment funds → keep original fund ticker, track via VOO growth
        ticker = fund

        if mapped in ("contribution_buy", "buy"):
            # Shares arrive — no cash impact.
            # "Contributions" = payroll into fund; "Exchange In" = rebalance.
            if shares_units == 0 or abs(amount) < 0.01:
                continue
            price_per = abs(amount) / shares_units if shares_units else 0
            rows.append({
                "type": "transfer", "ticker": ticker,
                "price": round(price_per, 4), "shares": shares_units,
                "amount": 0,
                "date": date_str, "account": account,
            })
            continue

        if mapped in ("sell", "withdrawal"):
            # Shares leave — no cash impact.
            # "Exchange Out" = rebalance; "Withdrawals" = money leaving plan.
            if shares_units == 0 or abs(amount) < 0.01:
                continue
            price_per = abs(amount) / shares_units if shares_units else 0
            rows.append({
                "type": "sell", "ticker": ticker,
                "price": round(price_per, 4), "shares": shares_units,
                "amount": 0,
                "date": date_str, "account": account,
            })
            continue

        if not _is_junk_row(raw):
            skipped.append({"file": os.path.basename(filepath), "line": lineno,
                            "reason": f"Unknown type2 action: {action_raw}",
                            "raw": dict(raw)})

    return rows, skipped


def parse_fidelity(filepath: str) -> tuple[list[dict], list[dict]]:
    """Parse a Fidelity CSV that may contain Type 1 and/or Type 2 sections."""
    account = _account_from_filename(filepath)
    all_rows: list[dict] = []
    all_skipped: list[dict] = []

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Find Type 1 header (Run Date,Action,Symbol,...)
    t1_idx = None
    # Find Type 2 header (Date,Investment,Transaction Type,...)
    t2_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("run date,"):
            t1_idx = i
        if stripped.startswith("date,investment,"):
            t2_idx = i

    if t1_idx is not None:
        # Read until blank section or t2 header
        end = t2_idx if t2_idx is not None else len(lines)
        section = lines[t1_idx:end]
        reader = csv.DictReader(section)
        r, s = _parse_fidelity_type1(reader, filepath, account, t1_idx + 2)
        all_rows.extend(r)
        all_skipped.extend(s)

    if t2_idx is not None:
        section = lines[t2_idx:]
        reader = csv.DictReader(section)
        r, s = _parse_fidelity_type2(reader, filepath, account, t2_idx + 2)
        all_rows.extend(r)
        all_skipped.extend(s)

    if t1_idx is None and t2_idx is None:
        all_skipped.append({"file": os.path.basename(filepath), "line": 0,
                            "reason": "Could not identify Fidelity CSV format", "raw": {}})

    return all_rows, all_skipped


# ── Folder-based import ──────────────────────────────────────────────────────

_IMPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "imports")
_BROKER_PARSERS = {
    "schwab":  parse_schwab,
    "fidelity": parse_fidelity,
    "etrade":  parse_etrade,
}


def import_all_from_folders() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Scan imports/schwab/, imports/fidelity/, imports/etrade/ folders.
    Each CSV in a broker folder is parsed with that broker's parser.
    Account name = filename without .csv extension.

    Returns:
        all_rows:    {filename: [parsed_row_dicts, ...]}
        all_skipped: {filename: [skipped_info_dicts, ...]}
    """
    all_rows: dict[str, list[dict]] = {}
    all_skipped: dict[str, list[dict]] = {}

    for broker, parser in _BROKER_PARSERS.items():
        folder = os.path.join(_IMPORTS_DIR, broker)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".csv"):
                continue
            filepath = os.path.join(folder, fname)
            # Use broker/filename as key to avoid collisions across folders
            key = f"{broker}/{fname}"
            try:
                rows, skipped = parser(filepath)
            except Exception as e:
                rows = []
                skipped = [{"file": key, "line": 0,
                            "reason": f"Parse error: {e}", "raw": {}}]
            all_rows[key] = rows
            all_skipped[key] = skipped

    return all_rows, all_skipped
