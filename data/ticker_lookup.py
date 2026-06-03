import os
import json
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory cache so repeated calls within the same Streamlit session are free
_mem_cache: dict[str, str] = {}


def _name_cache_path(symbol: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in symbol.upper())
    return os.path.join(CACHE_DIR, f"name_{safe}.json")


def get_ticker_name(symbol: str) -> str:
    """Return the short name for a ticker, or the symbol itself on failure.
    Cached permanently on disk — only hits the API once per ticker, ever.
    """
    symbol = symbol.upper().strip()
    if not symbol:
        return ""
    if symbol == "CASH":
        return "Cash"

    # VOO proxy funds — use fund name directly, no API lookup
    from data.csv_import import VOO_PROXY_FUNDS
    if symbol in VOO_PROXY_FUNDS:
        _mem_cache[symbol] = symbol
        return symbol

    # 1. In-memory
    if symbol in _mem_cache:
        return _mem_cache[symbol]

    # 2. Disk
    path = _name_cache_path(symbol)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                entry = json.load(f)
            name = entry.get("name", symbol)
            _mem_cache[symbol] = name
            return name
        except Exception:
            pass

    # 3. API (only for brand-new tickers)
    try:
        info = yf.Ticker(symbol).info
        name = info.get("shortName", symbol)
    except Exception:
        name = symbol

    _mem_cache[symbol] = name
    try:
        with open(path, "w") as f:
            json.dump({"name": name}, f)
    except Exception:
        pass
    return name
