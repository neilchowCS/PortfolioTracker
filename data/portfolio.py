import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from data.data_manager import load_transactions, get_account_names, get_existing_tickers
from data.ticker_lookup import get_ticker_name
from data.csv_import import VOO_PROXY_FUNDS, VOO_PROXY_TICKER

# Intra-day ordering: splits first, then contributions, buys/transfers, sells, withdrawals
_TYPE_ORDER = {"split": 0, "contribution": 1, "buy": 2, "transfer": 2, "sell": 3, "withdrawal": 4}


def _txn_amount(t, fallback: float) -> float:
    """Return the transaction's stored amount, or *fallback* if absent.
    amount=0 is treated as intentional (e.g. exchange with no cash impact).
    """
    raw = t.get("amount")
    if raw is None or (isinstance(raw, str) and raw == ""):
        return fallback
    if isinstance(raw, float) and pd.isna(raw):
        return fallback
    return float(raw)


def _sort_by_date_type(df: pd.DataFrame) -> pd.DataFrame:
    """Sort transactions by date, then by type priority within the same day."""
    df = df.copy()
    df["_sort"] = df["type"].map(_TYPE_ORDER).fillna(5)
    return df.sort_values(["date", "_sort"]).drop(columns="_sort").reset_index(drop=True)

# ── Price fetching (cached via yfinance) ──────────────────────────────────────

import os, json, time

_PRICE_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "prices")
os.makedirs(_PRICE_CACHE_DIR, exist_ok=True)


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _price_cache_path(symbol: str) -> str:
    return os.path.join(_PRICE_CACHE_DIR, f"{symbol.upper()}.json")


def _load_price_cache(symbol: str) -> dict | None:
    """Load cached price data. Valid if fetched today (same calendar date)."""
    path = _price_cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            entry = json.load(f)
        if entry.get("date") == _today_str():
            return entry
    except Exception:
        pass
    return None


def _save_price_cache(symbol: str, current: float, history: dict):
    path = _price_cache_path(symbol)
    try:
        with open(path, "w") as f:
            json.dump({
                "date": _today_str(),
                "ts": time.time(),
                "current": current,
                "history": history,
            }, f)
    except Exception:
        pass


def _fetch_and_cache(symbol: str) -> dict:
    """Fetch price data from yfinance and cache it. Returns cache dict."""
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        price = info.get("regularMarketPrice") or info.get("currentPrice") or 0.0
        hist = tk.history(period="max")
        history = {}
        if not hist.empty:
            history = {d.strftime("%Y-%m-%d"): round(v, 4) for d, v in hist["Close"].items()}
        _save_price_cache(symbol, price, history)
        return {"date": _today_str(), "current": price, "history": history}
    except Exception:
        return {"date": _today_str(), "current": 0.0, "history": {}}


def get_current_price(symbol: str) -> float:
    """Get the current market price for a ticker. Returns 0.0 on failure.
    Only hits the API if no cache exists for today.
    """
    if symbol.upper() == "CASH":
        return 1.0  # CASH is not a tradable security
    if symbol.upper() in VOO_PROXY_FUNDS:
        return 0.0  # proxy tickers valued via VOO growth in compute_holdings
    cached = _load_price_cache(symbol)
    if cached:
        return cached.get("current", 0.0)
    return _fetch_and_cache(symbol).get("current", 0.0)


def get_price_history(symbol: str) -> dict[str, float]:
    """Get full daily close price history as {date_str: price}.
    Only hits the API if no cache exists for today.
    """
    cached = _load_price_cache(symbol)
    if cached and cached.get("history"):
        return cached["history"]
    return _fetch_and_cache(symbol).get("history", {})


# ── Benchmark indices ─────────────────────────────────────────────────────────

BENCHMARKS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "SPMO": "SPMO",
    "VGT": "VGT",
}


def get_benchmark_series(symbol: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns [date, price] for a benchmark ticker.
    Uses the same date-based cache as stock prices.
    """
    history = get_price_history(symbol)
    if not history:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(list(history.items()), columns=["date", "price"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    return df


# ── Holdings computation ──────────────────────────────────────────────────────

def compute_holdings(account_filter: list[str] | None = None) -> pd.DataFrame:
    """Compute current holdings per ticker per account, including cash balances.
    Returns DataFrame with columns: account, ticker, shares, avg_cost, current_price,
    market_value, cost_basis, unrealized_gain.
    Cash is represented as ticker "CASH" with shares=1 and market_value=cash balance.
    """
    txns = load_transactions()
    if txns.empty:
        return pd.DataFrame()

    if account_filter:
        txns = txns[txns["account"].isin(account_filter)]

    trade_txns = txns[txns["type"].isin(["buy", "sell", "split", "transfer"])].copy()

    rows = []

    if not trade_txns.empty:
        trade_txns["price"] = pd.to_numeric(trade_txns["price"], errors="coerce").fillna(0)
        trade_txns["shares"] = pd.to_numeric(trade_txns["shares"], errors="coerce").fillna(0)
        trade_txns = _sort_by_date_type(trade_txns)

        # Pre-fetch VOO price history if any proxy tickers exist
        _voo_history = None
        _voo_current = None
        if trade_txns["ticker"].str.upper().isin(VOO_PROXY_FUNDS).any():
            _voo_history = get_price_history(VOO_PROXY_TICKER)
            _voo_current = get_current_price(VOO_PROXY_TICKER)

        for (acct, ticker), grp in trade_txns.groupby(["account", "ticker"]):
            is_proxy = str(ticker).upper() in VOO_PROXY_FUNDS
            # Process in date order: accumulate lots, apply splits
            # lots: [shares, cost_per_share, buy_date_str (proxy only)]
            lots = []
            for _, txn in grp.iterrows():
                if txn["type"] in ("buy", "transfer"):
                    buy_date = pd.to_datetime(txn["date"]).strftime("%Y-%m-%d") if pd.notna(txn["date"]) else ""
                    lots.append([txn["shares"], txn["price"], buy_date])
                elif txn["type"] == "sell":
                    remaining = txn["shares"]
                    while remaining > 0 and lots:
                        lot = lots[0]
                        used = min(remaining, lot[0])
                        lot[0] -= used
                        remaining -= used
                        if lot[0] <= 1e-9:
                            lots.pop(0)
                elif txn["type"] == "split":
                    # shares col = new, price col = old → ratio = new/old
                    split_old = txn["price"]
                    split_new = txn["shares"]
                    if split_old > 0:
                        ratio = split_new / split_old
                        for lot in lots:
                            lot[0] *= ratio
                            lot[1] /= ratio
            total_shares = sum(l[0] for l in lots)
            total_cost = sum(l[0] * l[1] for l in lots)
            avg_cost = total_cost / total_shares if total_shares > 0 else 0.0

            if is_proxy and _voo_history and _voo_current:
                # Market value = sum of each lot's cost × VOO growth since buy
                market_value = 0.0
                for lot_shares, lot_price, lot_date in lots:
                    lot_cost = lot_shares * lot_price
                    voo_at_buy = _lookup_price(_voo_history, lot_date) if lot_date else _voo_current
                    if voo_at_buy > 0:
                        market_value += lot_cost * (_voo_current / voo_at_buy)
                    else:
                        market_value += lot_cost
                current_price = market_value / total_shares if total_shares > 0 else 0.0
            else:
                current_price = get_current_price(str(ticker))
                market_value = total_shares * current_price

            cost_basis = total_cost
            unrealized = market_value - cost_basis
            rows.append({
                "account": acct,
                "ticker": str(ticker),
                "shares": round(total_shares, 4),
                "avg_cost": round(avg_cost, 4),
                "current_price": round(current_price, 4),
                "market_value": round(market_value, 2),
                "cost_basis": round(cost_basis, 2),
                "unrealized_gain": round(unrealized, 2),
            })

    # Compute cash balance per account: contributions - withdrawals + sells - buys
    # Note: transfer type does NOT affect cash (shares arrive with no cash outflow)
    for acct in txns["account"].unique():
        acct_txns = txns[txns["account"] == acct]
        cash = 0.0
        for _, t in acct_txns.iterrows():
            if t["type"] == "contribution":
                cash += float(t.get("amount", 0) or 0)
            elif t["type"] == "withdrawal":
                cash -= float(t.get("amount", 0) or 0)
            elif t["type"] == "buy":
                shares = float(t["shares"]) if pd.notna(t.get("shares")) and t["shares"] != "" else 0
                price = float(t["price"]) if pd.notna(t.get("price")) and t["price"] != "" else 0
                cash -= _txn_amount(t, shares * price)
            elif t["type"] == "sell":
                shares = float(t["shares"]) if pd.notna(t.get("shares")) and t["shares"] != "" else 0
                price = float(t["price"]) if pd.notna(t.get("price")) and t["price"] != "" else 0
                cash += _txn_amount(t, shares * price)
            # transfer: no cash impact (shares arrive without cash outflow)
        if abs(cash) >= 0.01:
            rows.append({
                "account": acct,
                "ticker": "CASH",
                "shares": 1.0,
                "avg_cost": round(cash, 2),
                "current_price": round(cash, 2),
                "market_value": round(cash, 2),
                "cost_basis": round(cash, 2),
                "unrealized_gain": 0.0,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_tax_lots(account_filter: list[str] | None = None) -> pd.DataFrame:
    """Compute remaining FIFO tax lots with holding-period classification.
    Returns DataFrame: account, ticker, shares, cost, buy_date, holding_days, is_long_term,
    current_price, unrealized_gain.
    Long-term = held > 365 days.
    """
    txns = load_transactions()
    if txns.empty:
        return pd.DataFrame()

    if account_filter:
        txns = txns[txns["account"].isin(account_filter)]

    trade_txns = _sort_by_date_type(txns[txns["type"].isin(["buy", "sell", "split", "transfer"])].copy())
    trade_txns["price"] = pd.to_numeric(trade_txns["price"], errors="coerce").fillna(0)
    trade_txns["shares"] = pd.to_numeric(trade_txns["shares"], errors="coerce").fillna(0)

    today = pd.Timestamp.now()
    rows = []
    for (acct, ticker), grp in trade_txns.groupby(["account", "ticker"]):
        lots = []  # FIFO: list of [shares_remaining, cost, buy_date]
        for _, txn in grp.iterrows():
            if txn["type"] in ("buy", "transfer"):
                lots.append([txn["shares"], txn["price"], pd.Timestamp(txn["date"])])
            elif txn["type"] == "sell":
                remaining = txn["shares"]
                while remaining > 0 and lots:
                    lot = lots[0]
                    used = min(remaining, lot[0])
                    lot[0] -= used
                    remaining -= used
                    if lot[0] <= 1e-9:
                        lots.pop(0)
            elif txn["type"] == "split":
                split_old = txn["price"]
                split_new = txn["shares"]
                if split_old > 0:
                    ratio = split_new / split_old
                    for lot in lots:
                        lot[0] *= ratio
                        lot[1] /= ratio
        cp = get_current_price(str(ticker))
        for lot in lots:
            if lot[0] <= 1e-9:
                continue
            holding_days = (today - lot[2]).days
            unrealized = lot[0] * (cp - lot[1])
            rows.append({
                "account": acct,
                "ticker": str(ticker),
                "shares": round(lot[0], 4),
                "cost": round(lot[1], 4),
                "buy_date": lot[2].strftime("%Y-%m-%d"),
                "holding_days": holding_days,
                "is_long_term": holding_days > 365,
                "current_price": round(cp, 4),
                "unrealized_gain": round(unrealized, 2),
            })
    return pd.DataFrame(rows)


def compute_realized_gains(account_filter: list[str] | None = None) -> pd.DataFrame:
    """Compute realized gains per ticker per account using FIFO.
    Returns DataFrame with columns: account, ticker, realized_gain.
    """
    txns = load_transactions()
    if txns.empty:
        return pd.DataFrame()

    if account_filter:
        txns = txns[txns["account"].isin(account_filter)]

    trade_txns = _sort_by_date_type(txns[txns["type"].isin(["buy", "sell", "split", "transfer"])].copy())
    trade_txns["price"] = pd.to_numeric(trade_txns["price"], errors="coerce").fillna(0)
    trade_txns["shares"] = pd.to_numeric(trade_txns["shares"], errors="coerce").fillna(0)

    rows = []
    for (acct, ticker), grp in trade_txns.groupby(["account", "ticker"]):
        lots = []  # FIFO buy lots: list of [shares_remaining, cost]
        realized = 0.0
        for _, txn in grp.iterrows():
            if txn["type"] in ("buy", "transfer"):
                lots.append([txn["shares"], txn["price"]])
            elif txn["type"] == "sell":
                sell_shares = txn["shares"]
                sell_price = txn["price"]
                remaining = sell_shares
                while remaining > 0 and lots:
                    lot = lots[0]
                    used = min(remaining, lot[0])
                    realized += used * (sell_price - lot[1])
                    lot[0] -= used
                    remaining -= used
                    if lot[0] <= 1e-9:
                        lots.pop(0)
            elif txn["type"] == "split":
                split_old = txn["price"]
                split_new = txn["shares"]
                if split_old > 0:
                    ratio = split_new / split_old
                    for lot in lots:
                        lot[0] *= ratio
                        lot[1] /= ratio
        rows.append({
            "account": acct,
            "ticker": str(ticker),
            "realized_gain": round(realized, 2),
        })
    return pd.DataFrame(rows)


# ── Account worth over time ───────────────────────────────────────────────────

def compute_account_worth_over_time(
    accounts: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compute daily account worth over time.
    Returns DataFrame with columns: date, account, worth.
    One row per date per account, plus a 'Total' pseudo-account.
    """
    txns = load_transactions()
    if txns.empty:
        return pd.DataFrame()

    all_accounts = accounts or get_account_names()
    txns = txns[txns["account"].isin(all_accounts)]

    if txns.empty:
        return pd.DataFrame()

    # Determine date range
    min_date = txns["date"].min()
    max_date = pd.Timestamp.now()
    if start_date:
        min_date = max(min_date, pd.Timestamp(start_date))
    if end_date:
        max_date = min(max_date, pd.Timestamp(end_date))

    date_range = pd.date_range(min_date, max_date, freq="D")
    if len(date_range) == 0:
        return pd.DataFrame()

    # Get all tickers and their price histories
    tickers = txns[txns["type"].isin(["buy", "sell", "transfer"])]["ticker"].dropna().unique()
    price_histories = {}
    has_proxy = False
    for t in tickers:
        t_str = str(t)
        if not t_str:
            continue
        if t_str.upper() in VOO_PROXY_FUNDS:
            has_proxy = True
            continue  # proxy tickers don't have their own price history
        price_histories[t_str] = get_price_history(t_str)
    # Fetch VOO history once if any proxy tickers exist
    voo_ph = get_price_history(VOO_PROXY_TICKER) if has_proxy else {}

    # Build daily state
    rows = []
    for acct in all_accounts:
        acct_txns = _sort_by_date_type(txns[txns["account"] == acct])
        cash = 0.0  # contributions - withdrawals + sells - buys (cash flow)
        holdings = {}  # ticker -> shares (non-proxy)
        proxy_lots = {}  # ticker -> [[shares, cost_per_share, buy_date_str], ...]

        txn_idx = 0
        for day in date_range:
            day_str = day.strftime("%Y-%m-%d")

            # Apply transactions up to this day
            while txn_idx < len(acct_txns):
                t = acct_txns.iloc[txn_idx]
                if pd.Timestamp(t["date"]) > day:
                    break
                if t["type"] == "contribution":
                    cash += float(t["amount"])
                elif t["type"] == "withdrawal":
                    cash -= float(t["amount"])
                elif t["type"] == "buy":
                    shares = float(t["shares"])
                    price = float(t["price"])
                    ticker = str(t["ticker"])
                    if ticker.upper() in VOO_PROXY_FUNDS:
                        buy_date = pd.to_datetime(t["date"]).strftime("%Y-%m-%d") if pd.notna(t["date"]) else day_str
                        proxy_lots.setdefault(ticker, []).append([shares, price, buy_date])
                    else:
                        holdings[ticker] = holdings.get(ticker, 0) + shares
                    cash -= _txn_amount(t, shares * price)
                elif t["type"] == "sell":
                    shares = float(t["shares"])
                    price = float(t["price"])
                    ticker = str(t["ticker"])
                    if ticker.upper() in VOO_PROXY_FUNDS:
                        remaining = shares
                        lots = proxy_lots.get(ticker, [])
                        while remaining > 0 and lots:
                            used = min(remaining, lots[0][0])
                            lots[0][0] -= used
                            remaining -= used
                            if lots[0][0] <= 1e-9:
                                lots.pop(0)
                    else:
                        holdings[ticker] = holdings.get(ticker, 0) - shares
                    cash += _txn_amount(t, shares * price)
                elif t["type"] == "transfer":
                    shares = float(t["shares"])
                    ticker = str(t["ticker"])
                    price = float(t["price"]) if pd.notna(t.get("price")) and t["price"] != "" else 0
                    if ticker.upper() in VOO_PROXY_FUNDS:
                        buy_date = pd.to_datetime(t["date"]).strftime("%Y-%m-%d") if pd.notna(t["date"]) else day_str
                        proxy_lots.setdefault(ticker, []).append([shares, price, buy_date])
                    else:
                        holdings[ticker] = holdings.get(ticker, 0) + shares
                    # No cash impact for transfers
                elif t["type"] == "split":
                    split_new = float(t["shares"])
                    split_old = float(t["price"])
                    ticker = str(t["ticker"])
                    if split_old > 0:
                        ratio = split_new / split_old
                        if ticker.upper() in VOO_PROXY_FUNDS:
                            for lot in proxy_lots.get(ticker, []):
                                lot[0] *= ratio
                                lot[1] /= ratio
                        elif ticker in holdings:
                            holdings[ticker] *= ratio
                txn_idx += 1

            # Value holdings at day's close
            stock_value = 0.0
            for ticker, shares in holdings.items():
                if shares == 0:
                    continue
                ph = price_histories.get(ticker, {})
                price = _lookup_price(ph, day_str)
                stock_value += shares * price

            # Value proxy lots using VOO growth
            voo_today = _lookup_price(voo_ph, day_str) if voo_ph else 0
            for ticker, lots in proxy_lots.items():
                for lot_shares, lot_price, lot_buy_date in lots:
                    if lot_shares <= 0:
                        continue
                    lot_cost = lot_shares * lot_price
                    voo_at_buy = _lookup_price(voo_ph, lot_buy_date) if voo_ph else 0
                    if voo_at_buy > 0 and voo_today > 0:
                        stock_value += lot_cost * (voo_today / voo_at_buy)
                    else:
                        stock_value += lot_cost

            total_worth = cash + stock_value
            rows.append({"date": day, "account": acct, "worth": round(total_worth, 2)})

    df = pd.DataFrame(rows)

    # Add Total row
    if not df.empty and len(all_accounts) > 1:
        totals = df.groupby("date")["worth"].sum().reset_index()
        totals["account"] = "Total"
        df = pd.concat([df, totals], ignore_index=True)

    return df


def _lookup_price(history: dict[str, float], date_str: str) -> float:
    """Find the price on or most recently before date_str."""
    if date_str in history:
        return history[date_str]
    # Walk backwards up to 10 days
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    for i in range(1, 11):
        prev = (dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if prev in history:
            return history[prev]
    return 0.0


# ── Return calculations ───────────────────────────────────────────────────────

def compute_returns(
    accounts: list[str] | None = None,
    days: int | None = None,
) -> dict:
    """Compute overall return and return excluding contributions/withdrawals.
    Returns dict with keys: total_return, total_return_pct,
    investment_return, investment_return_pct, contributions, withdrawals,
    start_worth, end_worth.
    """
    worth_df = compute_account_worth_over_time(accounts)
    if worth_df.empty:
        return {}

    # Use Total if available, else sum
    if "Total" in worth_df["account"].values:
        series = worth_df[worth_df["account"] == "Total"].sort_values("date")
    else:
        series = worth_df.groupby("date")["worth"].sum().reset_index().sort_values("date")
        series["account"] = "combined"

    if days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        series = series[series["date"] >= cutoff]

    if len(series) < 2:
        return {}

    start_worth = series.iloc[0]["worth"]
    end_worth = series.iloc[-1]["worth"]

    # Sum contributions and withdrawals in the period
    txns = load_transactions()
    if accounts:
        txns = txns[txns["account"].isin(accounts)]
    if days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        txns = txns[txns["date"] >= cutoff]

    contribs = txns[txns["type"] == "contribution"]["amount"].astype(float).sum()
    withdrawals = txns[txns["type"] == "withdrawal"]["amount"].astype(float).sum()
    net_flow = contribs - withdrawals

    total_return = end_worth - start_worth
    total_return_pct = (total_return / start_worth * 100) if start_worth != 0 else 0

    # Investment return = total return minus net contributions
    investment_return = total_return - net_flow
    inv_base = start_worth if start_worth != 0 else 1
    investment_return_pct = (investment_return / inv_base * 100)

    return {
        "start_worth": round(start_worth, 2),
        "end_worth": round(end_worth, 2),
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "investment_return": round(investment_return, 2),
        "investment_return_pct": round(investment_return_pct, 2),
        "contributions": round(contribs, 2),
        "withdrawals": round(withdrawals, 2),
    }


# ── Ticker summary ────────────────────────────────────────────────────────────

def compute_ticker_summary(account_filter: list[str] | None = None) -> pd.DataFrame:
    """Sorted list of tickers by current market value, with realized + unrealized gains.
    Includes tickers no longer held (0 shares).
    Returns DataFrame: ticker, name, shares, market_value, unrealized_gain,
    realized_gain, total_gain.
    """
    holdings = compute_holdings(account_filter)
    realized = compute_realized_gains(account_filter)

    # Separate CASH for proper aggregation
    cash_rows = holdings[holdings["ticker"] == "CASH"] if not holdings.empty else pd.DataFrame()
    stock_rows = holdings[holdings["ticker"] != "CASH"] if not holdings.empty else pd.DataFrame()

    # Aggregate across accounts
    if not stock_rows.empty:
        h_agg = stock_rows.groupby("ticker").agg({
            "shares": "sum",
            "market_value": "sum",
            "unrealized_gain": "sum",
        }).reset_index()
    else:
        h_agg = pd.DataFrame(columns=["ticker", "shares", "market_value", "unrealized_gain"])

    # Add CASH as a single row
    if not cash_rows.empty:
        cash_agg = pd.DataFrame([{
            "ticker": "CASH",
            "shares": 0.0,
            "market_value": cash_rows["market_value"].sum(),
            "unrealized_gain": 0.0,
        }])
        h_agg = pd.concat([h_agg, cash_agg], ignore_index=True)

    if not realized.empty:
        r_agg = realized.groupby("ticker")["realized_gain"].sum().reset_index()
    else:
        r_agg = pd.DataFrame(columns=["ticker", "realized_gain"])

    # Merge
    if h_agg.empty and r_agg.empty:
        return pd.DataFrame()

    summary = pd.merge(h_agg, r_agg, on="ticker", how="outer").fillna(0)
    summary["total_gain"] = summary["unrealized_gain"] + summary["realized_gain"]
    summary["name"] = summary["ticker"].apply(lambda t: get_ticker_name(str(t)))
    summary = summary.sort_values("market_value", ascending=False).reset_index(drop=True)
    summary = summary[["ticker", "name", "shares", "market_value", "unrealized_gain", "realized_gain", "total_gain"]]
    return summary
