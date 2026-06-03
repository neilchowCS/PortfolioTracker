import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.data_manager import get_account_names, load_transactions
from ui.fmt import fmtd, fmt_acct
from data.portfolio import (
    compute_account_worth_over_time,
    compute_returns,
    compute_ticker_summary,
    get_benchmark_series,
    BENCHMARKS,
)


def _color_val(val: float, fmt: str = ",.2f") -> str:
    color = "green" if val >= 0 else "red"
    return f'<span style="color:{color}">{fmtd(val, sign=True)}</span>'


def _color_pct(val: float) -> str:
    color = "green" if val >= 0 else "red"
    return f'<span style="color:{color}">{val:+.2f}%</span>'


UNIT_MAP = {"D": 1, "M": 30, "Y": 365}


def _data_version() -> int:
    return st.session_state.get("_data_version", 0)


@st.cache_data(show_spinner="Computing portfolio worth...")
def _cached_worth(accounts: tuple[str, ...], data_version: int) -> pd.DataFrame:
    return compute_account_worth_over_time(accounts=list(accounts))


@st.cache_data(show_spinner="Computing returns...")
def _cached_returns(accounts: tuple[str, ...], days: int | None, data_version: int) -> dict:
    return compute_returns(accounts=list(accounts), days=days)


@st.cache_data(show_spinner="Computing ticker summary...", ttl=60)
def _cached_ticker_summary(accounts: tuple[str, ...], data_version: int) -> pd.DataFrame:
    return compute_ticker_summary(account_filter=list(accounts))


@st.cache_data(show_spinner="Fetching benchmark data...")
def _cached_benchmark(symbol: str, start_str: str | None, end_str: str | None) -> pd.DataFrame:
    return get_benchmark_series(symbol, start_date=start_str, end_date=end_str)


def render():
    st.header("Portfolio Dashboard")

    txns = load_transactions()
    if txns.empty:
        st.info("No transactions yet. Add some in the Data Entry tab.")
        return

    all_accounts = get_account_names()
    if not all_accounts:
        st.info("No accounts found.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    st.subheader("Filters")
    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])

    with f1:
        period_num = st.number_input("Period", min_value=1, value=1, step=1, key="dash_period_num")
    with f2:
        period_unit = st.selectbox("Unit", ["D", "M", "Y"], index=2, key="dash_period_unit")
    days = int(period_num) * UNIT_MAP[period_unit]

    with f3:
        account_options = ["All Accounts"] + all_accounts
        selected_raw = st.multiselect(
            "Accounts",
            account_options,
            default=["All Accounts"],
            format_func=lambda x: x if x == "All Accounts" else fmt_acct(x),
            key="dash_accounts",
        )
    if "All Accounts" in selected_raw:
        selected_accounts = all_accounts
    else:
        selected_accounts = selected_raw

    with f4:
        selected_benchmarks = st.multiselect(
            "Comparisons",
            list(BENCHMARKS.keys()),
            default=[],
            key="dash_benchmarks",
        )

    if not selected_accounts:
        st.warning("Select at least one account.")
        return

    # ── Returns metrics ───────────────────────────────────────────────────────
    ver = _data_version()
    acct_key = tuple(sorted(selected_accounts))
    returns = _cached_returns(acct_key, days, ver)

    if returns:
        st.subheader("Returns")
        cols = st.columns(4 + len(selected_benchmarks))
        cols[0].metric("Current Worth", fmtd(returns['end_worth']))
        cols[1].markdown(
            f"**Overall Return**<br>{_color_val(returns['total_return'])}<br>"
            f"{_color_pct(returns['total_return_pct'])}",
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            f"**Investment Return**<br>{_color_val(returns['investment_return'])}<br>"
            f"{_color_pct(returns['investment_return_pct'])}<br>"
            f'<span style="font-size:0.8em;color:gray">excl. contributions</span>',
            unsafe_allow_html=True,
        )
        cols[3].metric(
            "Net Contributions",
            fmtd(returns['contributions'] - returns['withdrawals']),
        )

        # Benchmark returns
        cutoff_ts = pd.Timestamp.now() - pd.Timedelta(days=days)
        cutoff_str = cutoff_ts.strftime("%Y-%m-%d")
        for i, bname in enumerate(selected_benchmarks):
            bsym = BENCHMARKS[bname]
            bdf = _cached_benchmark(bsym, cutoff_str, None)
            if len(bdf) >= 2:
                b_start = bdf.iloc[0]["price"]
                b_end = bdf.iloc[-1]["price"]
                b_ret_pct = ((b_end - b_start) / b_start * 100) if b_start else 0
                cols[4 + i].markdown(
                    f"**{bname}**<br>{_color_pct(b_ret_pct)}",
                    unsafe_allow_html=True,
                )
            else:
                cols[4 + i].markdown(f"**{bname}**<br>N/A", unsafe_allow_html=True)

    # ── Worth over time chart ─────────────────────────────────────────────────
    st.subheader("Account Worth Over Time")

    worth_df = _cached_worth(acct_key, ver)

    if worth_df.empty:
        st.info("Not enough data to chart.")
    else:
        cutoff_ts = pd.Timestamp.now() - pd.Timedelta(days=days)
        worth_df = worth_df[worth_df["date"] >= cutoff_ts]

        # Let user choose which account lines to show
        available_series = sorted(worth_df["account"].unique())
        show_series = st.multiselect(
            "Show on chart",
            available_series,
            default=["Total"] if "Total" in available_series else available_series,
            format_func=lambda x: x if x == "Total" else fmt_acct(x),
            key="dash_chart_series",
        )

        chart_data = worth_df[worth_df["account"].isin(show_series)]

        if not chart_data.empty:
            fig = go.Figure()

            # Get the starting worth of the first shown account series for benchmark normalization
            acct_start_worth = None
            for acct_name in show_series:
                adf = chart_data[chart_data["account"] == acct_name].sort_values("date")
                if not adf.empty:
                    acct_start_worth = adf.iloc[0]["worth"]
                    break

            # Add account lines (always in $)
            for acct_name in show_series:
                adf = chart_data[chart_data["account"] == acct_name].sort_values("date")
                if adf.empty:
                    continue
                fig.add_trace(go.Scatter(x=adf["date"], y=adf["worth"], mode="lines", name=fmt_acct(acct_name) if acct_name != "Total" else acct_name))

            # Add benchmark lines, normalized to start at account's starting worth
            cutoff_str = cutoff_ts.strftime("%Y-%m-%d")
            for bname in selected_benchmarks:
                bsym = BENCHMARKS[bname]
                bdf = _cached_benchmark(bsym, cutoff_str, None)
                if bdf.empty or acct_start_worth is None:
                    continue
                b_base = bdf.iloc[0]["price"]
                if b_base:
                    y_vals = (bdf["price"] / b_base) * acct_start_worth
                else:
                    y_vals = bdf["price"] * 0
                fig.add_trace(go.Scatter(x=bdf["date"], y=y_vals, mode="lines", name=bname, line=dict(dash="dash")))

            fig.update_layout(
                yaxis_title="Worth",
                xaxis_title="Date",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one series to display.")

    # ── Ticker summary table ──────────────────────────────────────────────────
    st.subheader("Holdings & Gains by Ticker")

    summary = _cached_ticker_summary(acct_key, ver)
    if summary.empty:
        st.info("No stock transactions found.")
    else:
        display_df = summary.copy()
        display_df["currently_held"] = (display_df["shares"] > 0) | (display_df["ticker"] == "CASH")

        held = display_df[display_df["currently_held"]].copy()
        not_held = display_df[~display_df["currently_held"]].copy()

        if not held.empty:
            st.markdown("**Currently Held**")
            rows_html = ""
            for _, r in held.iterrows():
                tg = r["total_gain"]
                rg = r["realized_gain"]
                gains = f"{_color_val(tg)} ({_color_val(rg)})"
                shares_str = "—" if r["ticker"] == "CASH" else f"{r['shares']:.4f}"
                rows_html += (
                    f"<tr><td>{r['ticker']}</td><td>{r['name']}</td>"
                    f"<td>{shares_str}</td><td>{fmtd(r['market_value'])}</td>"
                    f"<td>{gains}</td></tr>"
                )
            st.markdown(
                '<table width="100%"><thead><tr>'
                "<th>Ticker</th><th>Name</th><th>Shares</th><th>Market Value</th><th>Gains (Realized)</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )

        if not not_held.empty:
            st.markdown("**Previously Held**")
            rows_html = ""
            for _, r in not_held.iterrows():
                tg = r["total_gain"]
                rg = r["realized_gain"]
                gains = f"{_color_val(tg)} ({_color_val(rg)})"
                rows_html += (
                    f"<tr><td>{r['ticker']}</td><td>{r['name']}</td>"
                    f"<td>{gains}</td></tr>"
                )
            st.markdown(
                '<table width="100%"><thead><tr>'
                "<th>Ticker</th><th>Name</th><th>Gains (Realized)</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )
