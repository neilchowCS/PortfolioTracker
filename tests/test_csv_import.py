"""Tests for data/csv_import.py — brokerage CSV parsers.

Tests parse the actual example data files in example_data/.
Run:  python -m pytest tests/test_csv_import.py -v
"""
import os
import tempfile
import pytest

from data.csv_import import parse_schwab, parse_etrade, parse_fidelity

_EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example_data")
_SCHWAB = os.path.join(_EXAMPLE_DIR, "schwab.csv")
_ETRADE = os.path.join(_EXAMPLE_DIR, "etrade.csv")
_FIDELITY = os.path.join(_EXAMPLE_DIR, "fidelity.csv")


def _write_csv(content: str, name: str = "test.csv") -> str:
    """Write content to a named temp file, return path."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ── Schwab (example_data/schwab.csv) ─────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(_SCHWAB), reason="example_data/schwab.csv missing")
class TestSchwabExample:
    @pytest.fixture(autouse=True)
    def _parse(self):
        self.rows, self.skipped = parse_schwab(_SCHWAB)

    def test_buy_parsed(self):
        buys = [r for r in self.rows if r["type"] == "buy"]
        # Buy AAAA + Reinvest Shares FFFF = 2 buys
        assert len(buys) == 2
        aaaa = [b for b in buys if b["ticker"] == "AAAA"]
        assert len(aaaa) == 1
        assert aaaa[0]["shares"] == 10.5
        assert aaaa[0]["price"] == 50.0

    def test_sell_parsed(self):
        sells = [r for r in self.rows if r["type"] == "sell"]
        assert len(sells) == 1
        assert sells[0]["ticker"] == "BBBB"
        assert sells[0]["shares"] == 30

    def test_split_parsed(self):
        splits = [r for r in self.rows if r["type"] == "split"]
        assert len(splits) == 1
        assert splits[0]["ticker"] == "DDDD"
        # "as of" date stripped — should parse 04/21/2026
        assert splits[0]["date"] == "2026-04-21"

    def test_contributions(self):
        """Cash Div, Qual Div, Credit Interest, Promo Award, Funds Received,
        MoneyLink Transfer, Journal, Qual Div Reinvest = 8 contributions."""
        contribs = [r for r in self.rows if r["type"] == "contribution"]
        assert len(contribs) == 8
        # MoneyLink should be $5,000
        ml = [c for c in contribs if c["amount"] == 5000.0]
        assert len(ml) == 1

    def test_withdrawals(self):
        """ADR Mgmt Fee + Foreign Tax Paid = 2 withdrawals."""
        wds = [r for r in self.rows if r["type"] == "withdrawal"]
        assert len(wds) == 2

    def test_security_transfer_skipped(self):
        xfers = [s for s in self.skipped if "Security transfer" in s["reason"]]
        assert len(xfers) == 1

    def test_account_from_filename(self):
        for r in self.rows:
            assert r["account"] == "schwab"

    def test_no_unknown_actions(self):
        unknown = [s for s in self.skipped if "Unknown action" in s["reason"]]
        assert len(unknown) == 0, f"Unexpected unknowns: {unknown}"


# ── E*Trade (example_data/etrade.csv) ────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(_ETRADE), reason="example_data/etrade.csv missing")
class TestETradeExample:
    @pytest.fixture(autouse=True)
    def _parse(self):
        self.rows, self.skipped = parse_etrade(_ETRADE)

    def test_buys(self):
        buys = [r for r in self.rows if r["type"] == "buy"]
        # Bought DDDD + Bought AAAA = 2
        assert len(buys) == 2
        tickers = {b["ticker"] for b in buys}
        assert tickers == {"DDDD", "AAAA"}

    def test_sells(self):
        sells = [r for r in self.rows if r["type"] == "sell"]
        assert len(sells) == 1
        assert sells[0]["ticker"] == "BBBB"

    def test_split(self):
        splits = [r for r in self.rows if r["type"] == "split"]
        assert len(splits) == 1
        assert splits[0]["ticker"] == "DDDD"
        assert splits[0]["shares"] == 8  # 8:1 ratio new
        assert splits[0]["price"] == 1   # 8:1 ratio old

    def test_contributions(self):
        """Interest Income, Qualified Dividend, Conversion, Dividend,
        Contribution = 5 contributions."""
        contribs = [r for r in self.rows if r["type"] == "contribution"]
        assert len(contribs) == 5
        # ACH Deposit = $3000
        ach = [c for c in contribs if c["amount"] == 3000.0]
        assert len(ach) == 1

    def test_online_transfer_withdrawal(self):
        """Two negative Online Transfers = 2 withdrawals."""
        wds = [r for r in self.rows if r["type"] == "withdrawal"]
        assert len(wds) == 2

    def test_stock_transfer_skipped(self):
        """Transfer of JJJJ should be skipped."""
        xfers = [s for s in self.skipped if "Stock transfer" in s["reason"]]
        assert len(xfers) == 1

    def test_two_digit_year_parsed(self):
        """E*Trade uses 2-digit years (06/02/26 = 2026-06-02)."""
        buys = [r for r in self.rows if r["type"] == "buy"]
        assert buys[0]["date"] == "2026-06-02"

    def test_footer_rows_ignored(self):
        """Footer disclaimer rows should not generate unknowns."""
        unknown = [s for s in self.skipped if "Unknown action" in s["reason"]]
        assert len(unknown) == 0, f"Unexpected unknowns: {unknown}"


# ── Fidelity (example_data/fidelity.csv — mixed type1 + type2) ───────────────

@pytest.mark.skipif(not os.path.exists(_FIDELITY), reason="example_data/fidelity.csv missing")
class TestFidelityExample:
    @pytest.fixture(autouse=True)
    def _parse(self):
        self.rows, self.skipped = parse_fidelity(_FIDELITY)

    # ── Type 1 (brokerage section) ──────────────────────────────────

    def test_t1_sell(self):
        sells = [r for r in self.rows if r["type"] == "sell"]
        bbbb = [s for s in sells if s["ticker"] == "BBBB"]
        assert len(bbbb) == 1
        assert bbbb[0]["shares"] == 190

    def test_t1_buy(self):
        buys = [r for r in self.rows if r["type"] == "buy"]
        dddd = [b for b in buys if b["ticker"] == "DDDD"]
        assert len(dddd) == 1
        assert dddd[0]["shares"] == 4

    def test_t1_reinvestment_cash_ticker(self):
        """FDRXX reinvestment → skipped (dividend already captured the income)."""
        fdrxx_buys = [r for r in self.rows if r["type"] == "buy" and r["ticker"] == "FDRXX"]
        assert len(fdrxx_buys) == 0
        # Only DIVIDEND RECEIVED emits a contribution; REINVESTMENT is skipped
        contribs = [r for r in self.rows if r["type"] == "contribution" and r["amount"] == 4.91]
        assert len(contribs) == 1

    def test_t1_transfer_contribution(self):
        """'TRANSFERRED FROM TO BROKERAGE OPTION' → contribution."""
        contribs = [r for r in self.rows if r["type"] == "contribution"]
        xfer_contribs = [c for c in contribs if c["amount"] == 1500.0]
        assert len(xfer_contribs) == 1

    # ── Type 2 (401k section) ───────────────────────────────────────

    def test_t2_no_cash_transactions(self):
        """Type 2 should emit no contributions or withdrawals — all cash-free."""
        t2_contribs = [r for r in self.rows if r["type"] == "contribution"
                       and r["ticker"] == ""]
        # Only Type 1 contributions exist (FDRXX dividend + transfer)
        for c in t2_contribs:
            assert c["amount"] != 500.0 and c["amount"] != 400.0, \
                "Type 2 should not emit contributions"

    def test_t2_lifepath_transfers(self):
        """LIFEPATH Contributions + Exchange In → transfers (no cash impact)."""
        xfers = [r for r in self.rows if r["type"] == "transfer"
                 and r["ticker"] == "LIFEPATH IDX 2065 F"]
        # 2 Contributions (05/08, 04/24) + 1 Exchange In (05/20) + 1 Contribution (04/10) = 4
        assert len(xfers) == 4
        for x in xfers:
            assert x["amount"] == 0  # no cash impact

    def test_t2_cash_funds_skipped(self):
        """BROKERAGELINK and STABLE VALUE rows produce no transactions."""
        all_tickers = {r.get("ticker", "") for r in self.rows}
        assert "BROKERAGELINK" not in all_tickers
        assert "STABLE VALUE" not in all_tickers

    def test_t2_market_value_changes_ignored(self):
        """Change in Market Value rows should produce no transactions."""
        # Amounts like 3.8, 0, 5.43, 0.51 — none should appear as rows
        mv_amounts = {1.5, 2.0, 0.51}
        mv_rows = [r for r in self.rows if r.get("amount") in mv_amounts]
        assert len(mv_rows) == 0

    def test_t2_zero_transfers_ignored(self):
        """Transfers with 0 amount should be silently skipped."""
        # Only check contribution/withdrawal; exchanges legitimately have amount=0
        zero_cash = [r for r in self.rows if r.get("amount") == 0
                     and r["type"] in ("contribution", "withdrawal")]
        assert len(zero_cash) == 0

    def test_no_unknown_actions(self):
        unknown = [s for s in self.skipped if "Unknown" in s.get("reason", "")]
        assert len(unknown) == 0, f"Unexpected unknowns: {unknown}"


# ── Skipped row schema ───────────────────────────────────────────────────────

class TestSkippedRowSchema:
    """Every skipped row dict must have file, line, reason keys."""

    @pytest.mark.skipif(not os.path.exists(_SCHWAB), reason="missing")
    def test_schwab_skipped_keys(self):
        _, skipped = parse_schwab(_SCHWAB)
        for s in skipped:
            assert "file" in s and "line" in s and "reason" in s, f"Bad skipped: {s}"

    @pytest.mark.skipif(not os.path.exists(_ETRADE), reason="missing")
    def test_etrade_skipped_keys(self):
        _, skipped = parse_etrade(_ETRADE)
        for s in skipped:
            assert "file" in s and "line" in s and "reason" in s, f"Bad skipped: {s}"

    @pytest.mark.skipif(not os.path.exists(_FIDELITY), reason="missing")
    def test_fidelity_skipped_keys(self):
        _, skipped = parse_fidelity(_FIDELITY)
        for s in skipped:
            assert "file" in s and "line" in s and "reason" in s, f"Bad skipped: {s}"


# ── Edge cases (synthetic data) ──────────────────────────────────────────────

class TestEdgeCases:
    def test_bad_date_skipped(self):
        csv = (
            "Date,Action,Symbol,Description,Quantity,Price,Fees & Comm,Amount\n"
            "not-a-date,Buy,XXXX,Sample,10,$100.00,$0.00,-$1000.00\n"
        )
        path = _write_csv(csv)
        rows, skipped = parse_schwab(path)
        assert len(rows) == 0
        assert len(skipped) == 1
        assert "Bad date" in skipped[0]["reason"]

    def test_missing_etrade_header(self):
        path = _write_csv("some,random,header\n1,2,3\n", "bad.csv")
        rows, skipped = parse_etrade(path)
        assert len(rows) == 0
        assert len(skipped) == 1
        assert "Could not find" in skipped[0]["reason"]

    def test_unrecognized_fidelity_format(self):
        path = _write_csv("Random,Columns,Here\n1,2,3\n", "bad.csv")
        rows, skipped = parse_fidelity(path)
        assert len(rows) == 0
        assert len(skipped) == 1
        assert "Could not identify" in skipped[0]["reason"]

    def test_parse_float_parens(self):
        """Accounting-style (1234.56) should parse as negative."""
        from data.csv_import import _parse_float
        assert _parse_float("($5,000.00)") == -5000.0
        assert _parse_float("$5,000.00") == 5000.0
        assert _parse_float("-$3,599.99") == -3599.99
        assert _parse_float("") == 0.0
        assert _parse_float("--") == 0.0
