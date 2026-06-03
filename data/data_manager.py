import os
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.dirname(__file__)
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")
ACCOUNTS_FILE = os.path.join(DATA_DIR, "accounts.csv")
CLASSIFICATIONS_FILE = os.path.join(DATA_DIR, "classifications.csv")

CLASSIFICATION_COLUMNS = ["ticker", "category"]
CATEGORIES = ["very_long_term", "long_term", "short_term"]

TRANSACTION_COLUMNS = [
    "id", "type", "ticker", "price", "shares", "amount", "date", "account", "source"
]
# type: buy, sell, contribution, withdrawal, split, transfer
# For buy/sell: ticker, price, shares are filled; amount = price * shares
# For contribution/withdrawal: amount is filled; ticker, price, shares are empty
# For transfer: ticker, shares, amount filled; treated as contribution + buy (no cash impact)
# source: 'manual' for hand-entered, filename for CSV-imported rows

ACCOUNT_COLUMNS = ["name", "taxable_type"]
# taxable_type: e.g. "taxable", "roth_ira", "traditional_ira", "401k", etc.


def _next_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    return int(df["id"].max()) + 1


# ── Accounts ─────────────────────────────────────────────────────────────────

def load_accounts() -> pd.DataFrame:
    if os.path.exists(ACCOUNTS_FILE):
        df = pd.read_csv(ACCOUNTS_FILE, dtype=str)
        for col in ACCOUNT_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[ACCOUNT_COLUMNS]
    return pd.DataFrame(columns=ACCOUNT_COLUMNS)


def save_accounts(df: pd.DataFrame):
    df.to_csv(ACCOUNTS_FILE, index=False)


def add_account(name: str, taxable_type: str) -> pd.DataFrame:
    df = load_accounts()
    if name in df["name"].values:
        raise ValueError(f"Account '{name}' already exists.")
    new_row = pd.DataFrame([{"name": name, "taxable_type": taxable_type}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_accounts(df)
    return df


def update_account(old_name: str, new_name: str, taxable_type: str) -> pd.DataFrame:
    df = load_accounts()
    mask = df["name"] == old_name
    if not mask.any():
        raise ValueError(f"Account '{old_name}' not found.")
    df.loc[mask, "name"] = new_name
    df.loc[mask, "taxable_type"] = taxable_type
    save_accounts(df)
    # Also update transactions that reference this account
    if old_name != new_name:
        txns = load_transactions()
        txns.loc[txns["account"] == old_name, "account"] = new_name
        save_transactions(txns)
    return df


def delete_account(name: str) -> pd.DataFrame:
    df = load_accounts()
    df = df[df["name"] != name].reset_index(drop=True)
    save_accounts(df)
    return df


def get_account_names() -> list[str]:
    df = load_accounts()
    return df["name"].tolist()


def get_existing_tickers() -> list[str]:
    """Return sorted list of unique tickers from saved transactions."""
    df = load_transactions()
    if df.empty:
        return []
    tickers = df["ticker"].dropna().astype(str)
    tickers = tickers[tickers != ""].str.upper().unique().tolist()
    return sorted(tickers)


# ── Transactions ──────────────────────────────────────────────────────────────

def load_transactions() -> pd.DataFrame:
    if os.path.exists(TRANSACTIONS_FILE):
        df = pd.read_csv(TRANSACTIONS_FILE)
        for col in TRANSACTION_COLUMNS:
            if col not in df.columns:
                df[col] = "manual" if col == "source" else ""
        # Backfill empty source values
        df["source"] = df["source"].fillna("manual").replace("", "manual")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["id"] = df["id"].astype(int)
        for c in ("price", "shares", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        _type_order = {"split": 0, "contribution": 1, "buy": 2, "transfer": 2, "sell": 3, "withdrawal": 4}
        df["_sort"] = df["type"].map(_type_order).fillna(5)
        df = df.sort_values(["date", "_sort"]).drop(columns="_sort").reset_index(drop=True)
        return df[TRANSACTION_COLUMNS]
    return pd.DataFrame(columns=TRANSACTION_COLUMNS)


def save_transactions(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(TRANSACTIONS_FILE, index=False)


def add_transaction(
    txn_type: str,
    account: str,
    ticker: str = "",
    price: float = 0.0,
    shares: float = 0.0,
    amount: float = 0.0,
    date: str | datetime = "",
) -> pd.DataFrame:
    df = load_transactions()
    new_id = _next_id(df)
    if isinstance(date, str):
        date = pd.to_datetime(date)
    if txn_type in ("buy", "sell"):
        amount = round(price * shares, 4)
    row = {
        "id": new_id,
        "type": txn_type,
        "ticker": ticker.upper() if ticker else "",
        "price": price if txn_type in ("buy", "sell", "split", "transfer") else 0,
        "shares": shares if txn_type in ("buy", "sell", "split", "transfer") else 0,
        "amount": amount if txn_type != "split" else 0,
        "date": date,
        "account": account,
        "source": "manual",
    }
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    save_transactions(df)
    return df


def update_transaction(
    txn_id: int,
    txn_type: str,
    account: str,
    ticker: str = "",
    price: float = 0.0,
    shares: float = 0.0,
    amount: float = 0.0,
    date: str | datetime = "",
) -> pd.DataFrame:
    df = load_transactions()
    mask = df["id"] == txn_id
    if not mask.any():
        raise ValueError(f"Transaction {txn_id} not found.")
    if isinstance(date, str):
        date = pd.to_datetime(date)
    if txn_type in ("buy", "sell"):
        amount = round(price * shares, 4)
        df.loc[mask, "ticker"] = ticker.upper()
        df.loc[mask, "price"] = price
        df.loc[mask, "shares"] = shares
    elif txn_type == "split":
        df.loc[mask, "ticker"] = ticker.upper()
        df.loc[mask, "price"] = price   # old shares
        df.loc[mask, "shares"] = shares  # new shares
        amount = 0
    elif txn_type == "transfer":
        df.loc[mask, "ticker"] = ticker.upper()
        df.loc[mask, "price"] = price
        df.loc[mask, "shares"] = shares
    else:
        df.loc[mask, "ticker"] = ""
        df.loc[mask, "price"] = 0
        df.loc[mask, "shares"] = 0
    df.loc[mask, "type"] = txn_type
    df.loc[mask, "amount"] = amount
    df.loc[mask, "date"] = date
    df.loc[mask, "account"] = account
    save_transactions(df)
    return df


def delete_transaction(txn_id: int) -> pd.DataFrame:
    df = load_transactions()
    df = df[df["id"] != txn_id].reset_index(drop=True)
    save_transactions(df)
    return df


# ── Ticker Classifications ────────────────────────────────────────────────────

def load_classifications() -> pd.DataFrame:
    if os.path.exists(CLASSIFICATIONS_FILE):
        df = pd.read_csv(CLASSIFICATIONS_FILE, dtype=str)
        for col in CLASSIFICATION_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[CLASSIFICATION_COLUMNS]
    return pd.DataFrame(columns=CLASSIFICATION_COLUMNS)


def save_classifications(df: pd.DataFrame):
    df.to_csv(CLASSIFICATIONS_FILE, index=False)


def set_classification(ticker: str, category: str):
    df = load_classifications()
    ticker = ticker.upper().strip()
    mask = df["ticker"] == ticker
    if mask.any():
        df.loc[mask, "category"] = category
    else:
        new_row = pd.DataFrame([{"ticker": ticker, "category": category}])
        df = pd.concat([df, new_row], ignore_index=True)
    save_classifications(df)


def get_classification(ticker: str) -> str:
    df = load_classifications()
    mask = df["ticker"] == ticker.upper().strip()
    if mask.any():
        return df.loc[mask, "category"].iloc[0]
    return ""
