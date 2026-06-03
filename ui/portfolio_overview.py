import streamlit as st
import pandas as pd

from data.data_manager import load_accounts, load_classifications, get_account_names
from data.portfolio import compute_holdings, compute_tax_lots, get_current_price
from data.ticker_lookup import get_ticker_name
from ui.fmt import fmtd, fmt_acct


def _lot_table(lots_df: pd.DataFrame, show_account: bool = False):
    """Render a tax-lot table inside an expander."""
    if lots_df.empty:
        st.caption("No lot data available.")
        return
    disp = lots_df.copy()
    disp["term"] = disp["is_long_term"].map({True: "Long Term", False: "Short Term"})
    disp["gain_pct"] = disp.apply(
        lambda r: (r["unrealized_gain"] / (r["shares"] * r["cost"]) * 100)
        if r["shares"] * r["cost"] else 0, axis=1
    )
    cols = []
    if show_account:
        disp["account"] = disp["account"].apply(fmt_acct)
        cols.append("account")
    cols += ["buy_date", "term", "shares", "cost", "current_price", "unrealized_gain", "gain_pct", "holding_days"]

    rename_map = {
        "account": "Account",
        "buy_date": "Buy Date",
        "term": "Term",
        "shares": "Shares",
        "cost": "Cost Basis",
        "current_price": "Price",
        "unrealized_gain": "Gain $",
        "gain_pct": "Gain %",
        "holding_days": "Days Held",
    }
    disp = disp[cols].rename(columns=rename_map)

    col_config = {
        "Cost Basis": st.column_config.NumberColumn(format="$%.4f"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Gain $": st.column_config.NumberColumn(format="$%.2f"),
        "Gain %": st.column_config.NumberColumn(format="%.2f%%"),
        "Shares": st.column_config.NumberColumn(format="%.4f"),
    }
    st.dataframe(disp, use_container_width=True, hide_index=True, column_config=col_config)


def render():
    st.header("Portfolio Overview")

    holdings = compute_holdings()
    if holdings.empty:
        st.info("No holdings yet. Add buy transactions in the Data Entry tab.")
        return

    tax_lots = compute_tax_lots()

    accounts_df = load_accounts()
    acct_type_map = dict(zip(accounts_df["name"], accounts_df["taxable_type"]))

    class_df = load_classifications()
    class_map = dict(zip(class_df["ticker"], class_df["category"])) if not class_df.empty else {}

    # Separate CASH from stock holdings for aggregation
    cash_holdings = holdings[holdings["ticker"] == "CASH"]
    stock_holdings = holdings[holdings["ticker"] != "CASH"]

    # ── Aggregate across all accounts ─────────────────────────────────────────
    agg = stock_holdings.groupby("ticker").agg({
        "shares": "sum",
        "market_value": "sum",
        "cost_basis": "sum",
        "unrealized_gain": "sum",
        "avg_cost": "mean",
        "current_price": "first",
    }).reset_index() if not stock_holdings.empty else pd.DataFrame(
        columns=["ticker", "shares", "market_value", "cost_basis", "unrealized_gain", "avg_cost", "current_price"]
    )

    # Add CASH as a single aggregated row
    if not cash_holdings.empty:
        total_cash = cash_holdings["market_value"].sum()
        cash_row = pd.DataFrame([{
            "ticker": "CASH", "shares": 0, "market_value": total_cash,
            "cost_basis": total_cash, "unrealized_gain": 0.0,
            "avg_cost": 0, "current_price": 0,
        }])
        agg = pd.concat([agg, cash_row], ignore_index=True)

    total_value = agg["market_value"].sum()
    agg["pct_portfolio"] = (agg["market_value"] / total_value * 100) if total_value else 0
    agg["name"] = agg["ticker"].apply(get_ticker_name)
    agg["category"] = agg["ticker"].map(class_map).fillna("")
    agg["category_label"] = agg["category"].apply(
        lambda x: x.replace("_", " ").title() if x else "—"
    )
    agg["gain_pct"] = agg.apply(
        lambda r: (r["unrealized_gain"] / r["cost_basis"] * 100) if r["cost_basis"] else 0, axis=1
    )

    # Tax-free account types (gains are never taxed)
    TAX_FREE_TYPES = {"roth_ira", "401k_roth", "hsa"}
    tax_free_accts = set(accounts_df[accounts_df["taxable_type"].isin(TAX_FREE_TYPES)]["name"])
    taxable_accts = set(accounts_df[accounts_df["taxable_type"] == "taxable"]["name"])

    # Compute tax-free gain per ticker (from roth/hsa accounts)
    tax_free_holdings = holdings[holdings["account"].isin(tax_free_accts)]
    tf_agg = tax_free_holdings.groupby("ticker")["unrealized_gain"].sum()
    agg["tax_free_gain"] = agg["ticker"].map(tf_agg).fillna(0)

    # Compute LT/ST split per ticker (taxable accounts only)
    if not tax_lots.empty:
        taxable_lots = tax_lots[tax_lots["account"].isin(taxable_accts)]
        lt_agg = taxable_lots[taxable_lots["is_long_term"]].groupby("ticker")["unrealized_gain"].sum()
        st_agg = taxable_lots[~taxable_lots["is_long_term"]].groupby("ticker")["unrealized_gain"].sum()
        agg["lt_gain"] = agg["ticker"].map(lt_agg).fillna(0)
        agg["st_gain"] = agg["ticker"].map(st_agg).fillna(0)
    else:
        agg["lt_gain"] = 0
        agg["st_gain"] = 0

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Portfolio Value", fmtd(total_value))
    total_cost = agg["cost_basis"].sum()
    total_gain = agg["unrealized_gain"].sum()
    total_gain_pct = (total_gain / total_cost * 100) if total_cost else 0
    gain_color = "green" if total_gain >= 0 else "red"
    m2.markdown(
        f"**Total Gains (Unrealized)**<br>"
        f'<span style="color:{gain_color}">{fmtd(total_gain, sign=True)} '
        f'({total_gain_pct:+.2f}%)</span>',
        unsafe_allow_html=True,
    )
    m3.metric("Positions", f"{len(agg[agg['shares'] > 0])}")

    st.divider()

    # ── Main holdings table ───────────────────────────────────────────────────
    st.subheader("All Holdings")

    active = agg[(agg["shares"] > 0) | (agg["ticker"] == "CASH")].sort_values("market_value", ascending=False)

    for _, row in active.iterrows():
        ticker = row["ticker"]
        gain_sign = "+" if row["unrealized_gain"] >= 0 else "-"
        gain_color = "green" if row["unrealized_gain"] >= 0 else "red"

        if ticker == "CASH":
            header = (
                f"**CASH**  |  "
                f"{fmtd(row['market_value'])}  |  "
                f"**{row['pct_portfolio']:.1f}%**"
            )
        else:
            header = (
                f"**{ticker}** — {row['name']}  |  "
                f"{row['shares']:.4f} shares @ {fmtd(row['current_price'])}  |  "
                f"{fmtd(row['market_value'])}  |  "
                f"**{row['pct_portfolio']:.1f}%**  |  "
                f"{row['category_label']}"
            )

        with st.expander(header):
            if ticker == "CASH":
                st.caption("Uninvested cash from contributions minus purchases.")
                cash_by_acct = holdings[holdings["ticker"] == "CASH"].copy()
                cash_by_acct["% of Portfolio"] = (
                    cash_by_acct["market_value"] / total_value * 100
                ) if total_value else 0
                cash_by_acct["account"] = cash_by_acct["account"].apply(fmt_acct)
                cash_table = cash_by_acct[["account", "market_value", "% of Portfolio"]].rename(
                    columns={"account": "Account", "market_value": "Value"}
                )
                st.dataframe(
                    cash_table,
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Value": st.column_config.NumberColumn(format="$%,.2f"),
                        "% of Portfolio": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )
            else:
                gain_parts = []
                if row["lt_gain"] != 0:
                    gain_parts.append(("Long Term Gain", row["lt_gain"]))
                if row["st_gain"] != 0:
                    gain_parts.append(("Short Term Gain", row["st_gain"]))
                if row["tax_free_gain"] != 0:
                    gain_parts.append(("Tax Free Gain", row["tax_free_gain"]))

                cols = st.columns(2 + len(gain_parts))
                cols[0].metric("Avg Cost", fmtd(row['avg_cost'], decimals=4))
                cols[1].markdown(
                    f"**Total Gain**<br>"
                    f'<span style="color:{gain_color}">{fmtd(row["unrealized_gain"], sign=True)} '
                    f'({row["gain_pct"]:+.2f}%)</span>',
                    unsafe_allow_html=True,
                )
                for i, (label, val) in enumerate(gain_parts):
                    color = "green" if val >= 0 else "red"
                    cols[2 + i].markdown(
                        f'**{label}**: <span style="color:{color}">{fmtd(val, sign=True)}</span>',
                        unsafe_allow_html=True,
                    )

                # Per-account breakdown for this ticker
                ticker_holdings = holdings[
                    (holdings["ticker"] == ticker) & (holdings["shares"] > 0)
                ].copy()
                if len(ticker_holdings) > 1:
                    st.markdown("**Account Breakdown**")
                    ticker_holdings["Account"] = ticker_holdings["account"].apply(fmt_acct)
                    ticker_holdings["% of Position"] = (
                        ticker_holdings["market_value"] / row["market_value"] * 100
                    ) if row["market_value"] else 0
                    acct_table = ticker_holdings[["Account", "shares", "market_value", "unrealized_gain", "% of Position"]].rename(
                        columns={"shares": "Shares", "market_value": "Value", "unrealized_gain": "Gain"}
                    )
                    st.dataframe(
                        acct_table,
                        use_container_width=True, hide_index=True,
                        column_config={
                            "Shares": st.column_config.NumberColumn(format="%.4f"),
                            "Value": st.column_config.NumberColumn(format="$%,.2f"),
                            "Gain": st.column_config.NumberColumn(format="$%+,.2f"),
                            "% of Position": st.column_config.NumberColumn(format="%.1f%%"),
                        },
                    )

                # Tax lots with account name
                ticker_lots = tax_lots[tax_lots["ticker"] == ticker] if not tax_lots.empty else pd.DataFrame()
                _lot_table(ticker_lots, show_account=True)

    # ── Per-account breakdown ─────────────────────────────────────────────────
    st.subheader("By Account")

    account_names = get_account_names()
    for acct in account_names:
        acct_holdings = holdings[
            (holdings["account"] == acct) & (holdings["shares"] > 0)
        ].copy()
        if acct_holdings.empty:
            continue

        acct_total = acct_holdings["market_value"].sum()
        acct_type_raw = acct_type_map.get(acct, "")
        acct_type = acct_type_raw.replace("_", " ").title()
        is_tax_free = acct_type_raw in TAX_FREE_TYPES

        with st.expander(f"{fmt_acct(acct)} ({acct_type}) — {fmtd(acct_total)}"):
            acct_holdings["pct_acct"] = (
                (acct_holdings["market_value"] / acct_total * 100) if acct_total else 0
            )
            acct_holdings["pct_portfolio"] = (
                (acct_holdings["market_value"] / total_value * 100) if total_value else 0
            )
            acct_holdings["name"] = acct_holdings["ticker"].apply(get_ticker_name)
            acct_holdings["gain_pct"] = acct_holdings.apply(
                lambda r: (r["unrealized_gain"] / r["cost_basis"] * 100) if r["cost_basis"] else 0,
                axis=1,
            )

            for _, h in acct_holdings.sort_values("market_value", ascending=False).iterrows():
                tkr = h["ticker"]
                if tkr == "CASH":
                    st.markdown(
                        f"**CASH**  |  {fmtd(h['market_value'])}  |  "
                        f"{h['pct_acct']:.1f}% acct / {h['pct_portfolio']:.1f}% portfolio"
                    )
                    continue
                g_sign = "+" if h["unrealized_gain"] >= 0 else "-"
                g_col = "green" if h["unrealized_gain"] >= 0 else "red"
                gain_label = "Tax Free Gain" if is_tax_free else "Gain"
                row_header = (
                    f"**{tkr}** — {h['name']}  |  "
                    f"{h['shares']:.4f} shares  |  "
                    f"{fmtd(h['market_value'])}  |  "
                    f"{h['pct_acct']:.1f}% acct / {h['pct_portfolio']:.1f}% portfolio"
                )
                with st.expander(row_header):
                    rc1, rc2 = st.columns(2)
                    rc1.metric("Avg Cost", fmtd(h['avg_cost'], decimals=4))
                    rc2.markdown(
                        f"**{gain_label}**: "
                        f'<span style="color:{g_col}">{fmtd(h["unrealized_gain"], sign=True)} '
                        f'({h["gain_pct"]:+.2f}%)</span>',
                        unsafe_allow_html=True,
                    )
                    # Lots for this ticker in this account only
                    acct_lots = tax_lots[
                        (tax_lots["ticker"] == tkr) & (tax_lots["account"] == acct)
                    ] if not tax_lots.empty else pd.DataFrame()
                    _lot_table(acct_lots, show_account=False)
