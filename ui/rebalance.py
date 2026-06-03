import streamlit as st
import pandas as pd

from data.data_manager import (
    load_accounts,
    load_classifications,
    set_classification,
    get_existing_tickers,
    CATEGORIES,
)
from data.portfolio import compute_holdings, compute_tax_lots
from ui.fmt import fmtd, fmt_acct

# Tax-advantaged account types
TAX_ADVANTAGED = {"roth_ira", "401k_pretax", "401k_roth", "401k_aftertax", "hsa"}

# Group similar account types for consolidation checks
TAX_GROUP = {
    "roth_ira": "Roth",
    "401k_roth": "Roth",
    "401k_pretax": "Pre-Tax",
    "401k_aftertax": "After-Tax",
    "hsa": "Roth",
    "taxable": "Taxable",
}

# Which categories belong where for tax efficiency
CATEGORY_PREFERRED_LOCATION = {
    "very_long_term": "taxable",
    "long_term": "either",
    "short_term": "tax_advantaged",
}

CAT_LABELS = {
    "very_long_term": "Very Long Term",
    "long_term": "Long Term",
    "short_term": "Short Term",
}


def _fmt_cat(val: str) -> str:
    return CAT_LABELS.get(val, val.replace("_", " ").title())


def render():
    st.header("Rebalance")

    holdings = compute_holdings()
    if holdings.empty:
        st.info("No holdings yet. Add buy transactions first.")
        return

    accounts_df = load_accounts()
    if accounts_df.empty:
        st.info("No accounts found.")
        return

    # Build account type lookup
    acct_type_map = dict(zip(accounts_df["name"], accounts_df["taxable_type"]))

    # Exclude CASH from rebalance logic (cash is fungible, not a position)
    holdings = holdings[holdings["ticker"] != "CASH"].copy()

    # Enrich holdings with account type
    holdings["acct_type"] = holdings["account"].map(acct_type_map)
    holdings["is_tax_adv"] = holdings["acct_type"].isin(TAX_ADVANTAGED)

    # Load existing classifications
    class_df = load_classifications()
    class_map = dict(zip(class_df["ticker"], class_df["category"])) if not class_df.empty else {}

    # ── Section 1: Ticker Classification ──────────────────────────────────────
    st.subheader("1. Ticker Classification")
    st.caption("Assign each ticker to a category. This is saved and persists across sessions.")

    # Aggregate holdings across all accounts for the summary
    ticker_agg = holdings.groupby("ticker").agg({
        "market_value": "sum",
        "shares": "sum",
    }).reset_index()
    total_portfolio = ticker_agg["market_value"].sum()
    ticker_agg["pct_portfolio"] = (
        (ticker_agg["market_value"] / total_portfolio * 100) if total_portfolio else 0
    )
    ticker_agg = ticker_agg.sort_values("market_value", ascending=False).reset_index(drop=True)

    # Show classification editor
    changed = False
    cols_header = st.columns([2, 2, 2, 1, 1])
    cols_header[0].markdown("**Ticker**")
    cols_header[1].markdown("**Category**")
    cols_header[2].markdown("**Market Value**")
    cols_header[3].markdown("**% Portfolio**")
    cols_header[4].markdown("**Shares**")

    for i, row in ticker_agg.iterrows():
        ticker = row["ticker"]
        current_cat = class_map.get(ticker, "")
        cols = st.columns([2, 2, 2, 1, 1])
        cols[0].write(ticker)

        cat_options = [""] + CATEGORIES
        current_idx = cat_options.index(current_cat) if current_cat in cat_options else 0
        new_cat = cols[1].selectbox(
            f"cat_{ticker}",
            cat_options,
            index=current_idx,
            format_func=lambda x: "— Select —" if x == "" else _fmt_cat(x),
            key=f"class_{ticker}",
            label_visibility="collapsed",
        )
        if new_cat != current_cat:
            set_classification(ticker, new_cat)
            class_map[ticker] = new_cat
            changed = True

        cols[2].write(fmtd(row['market_value']))
        cols[3].write(f"{row['pct_portfolio']:.1f}%")
        cols[4].write(f"{row['shares']:.4f}")

    if changed:
        st.rerun()

    # ── Section 2: Portfolio Overview by Category ─────────────────────────────
    st.subheader("2. Portfolio Overview by Category")

    # Reload classifications fresh (picks up any changes from section 1)
    class_df = load_classifications()
    class_map = dict(zip(class_df["ticker"], class_df["category"])) if not class_df.empty else {}
    holdings["category"] = holdings["ticker"].map(class_map).fillna("")
    holdings["location"] = holdings["is_tax_adv"].map({True: "Tax-Advantaged", False: "Taxable"})

    # Category summary
    cat_summary = holdings.groupby("category").agg({"market_value": "sum"}).reset_index()
    cat_summary["pct"] = (cat_summary["market_value"] / total_portfolio * 100) if total_portfolio else 0
    cat_summary["category_label"] = cat_summary["category"].apply(
        lambda x: _fmt_cat(x) if x else "Unclassified"
    )

    for _, cr in cat_summary.iterrows():
        st.markdown(f"**{cr['category_label']}**: {fmtd(cr['market_value'])} ({cr['pct']:.1f}%)")

    # Category × Location breakdown
    if not holdings.empty:
        breakdown = holdings.groupby(["category", "location"]).agg({"market_value": "sum"}).reset_index()
        breakdown["category_label"] = breakdown["category"].apply(
            lambda x: _fmt_cat(x) if x else "Unclassified"
        )
        pivot = breakdown.pivot_table(
            index="category_label", columns="location", values="market_value", fill_value=0
        )
        st.dataframe(pivot, use_container_width=True)

    # Flag misplacements
    st.markdown("**Tax Efficiency Flags**")
    flags = []
    for _, h in holdings.iterrows():
        cat = h["category"]
        if not cat:
            continue
        pref = CATEGORY_PREFERRED_LOCATION.get(cat, "either")
        if pref == "taxable" and h["is_tax_adv"]:
            flags.append(
                f"⚠️ **{h['ticker']}** ({_fmt_cat(cat)}) is in tax-advantaged account "
                f"**{fmt_acct(h['account'])}** — consider moving to a taxable account"
            )
        elif pref == "tax_advantaged" and not h["is_tax_adv"]:
            flags.append(
                f"⚠️ **{h['ticker']}** ({_fmt_cat(cat)}) is in taxable account "
                f"**{fmt_acct(h['account'])}** — consider moving to a tax-advantaged account"
            )

    if flags:
        for f in flags:
            st.markdown(f)
    else:
        st.success("All tickers are in tax-efficient locations.")

    # ── Section 3: Account Consolidation ──────────────────────────────────────
    st.subheader("3. Consolidation Recommendations")
    st.caption("Tickers held in multiple accounts of the same or similar type (e.g. Roth IRA and Roth 401k are grouped together).")

    holdings["acct_type_label"] = holdings["acct_type"].apply(
        lambda x: str(x).replace("_", " ").title() if pd.notna(x) and x else "Unknown"
    )
    holdings["tax_group"] = holdings["acct_type"].map(TAX_GROUP).fillna("Other")

    exact_dupes = []
    group_dupes = []
    seen_exact = set()
    seen_group = set()

    for ticker in holdings["ticker"].unique():
        t_holdings = holdings[holdings["ticker"] == ticker]

        # Tier 1: exact same account type
        for atype in t_holdings["acct_type"].unique():
            type_holdings = t_holdings[t_holdings["acct_type"] == atype]
            if len(type_holdings) > 1:
                key = (ticker, atype)
                if key in seen_exact:
                    continue
                seen_exact.add(key)
                shares_info = ", ".join(
                    f"{fmt_acct(r['account'])} ({r['shares']:.4f} shares)"
                    for _, r in type_holdings.iterrows()
                )
                exact_dupes.append(
                    f"🔴 **{ticker}** in multiple **{type_holdings.iloc[0]['acct_type_label']}** "
                    f"accounts: {shares_info}. **Strongly recommend consolidating.**"
                )
                # Mark these accounts so we don't re-flag at group level
                for _, r in type_holdings.iterrows():
                    seen_group.add((ticker, r["account"]))

        # Tier 2: same tax group but different account types
        for group in t_holdings["tax_group"].unique():
            group_holdings = t_holdings[t_holdings["tax_group"] == group]
            if len(group_holdings) <= 1:
                continue
            # Only flag if there are accounts not already flagged in tier 1
            unflagged = group_holdings[
                ~group_holdings.apply(lambda r: (ticker, r["account"]) in seen_group, axis=1)
            ]
            if len(unflagged) < len(group_holdings) and len(unflagged) == 0:
                continue  # all already flagged as exact dupes
            if len(group_holdings["acct_type"].unique()) > 1:
                key = (ticker, group)
                if key not in seen_exact:
                    shares_info = ", ".join(
                        f"{fmt_acct(r['account'])} [{r['acct_type_label']}] ({r['shares']:.4f} shares)"
                        for _, r in group_holdings.iterrows()
                    )
                    group_dupes.append(
                        f"🟡 **{ticker}** in multiple **{group}**-equivalent accounts: "
                        f"{shares_info}. Consider consolidating if possible."
                    )

    if exact_dupes or group_dupes:
        if exact_dupes:
            st.markdown("**Same Account Type** (high priority)")
            for d in exact_dupes:
                st.markdown(d)
        if group_dupes:
            st.markdown("**Same Tax Group** (lower priority — different account types, similar tax treatment)")
            for d in group_dupes:
                st.markdown(d)
    else:
        st.success("No duplicate tickers across same-type accounts.")

    # ── Section 4: Rebalance Suggestions ──────────────────────────────────────
    st.subheader("4. Rebalance Suggestions")
    st.caption(
        "Based on tax-efficiency rules and capital gains impact. "
        "Selling in taxable accounts triggers taxable events — long-term gains (>1yr) are taxed at lower rates."
    )

    # Get tax lot data for capital gains analysis
    tax_lots = compute_tax_lots()
    suggestions = []
    warnings = []

    # Helper: summarize tax impact of selling a position in a given account
    def _tax_impact_note(ticker: str, account: str, is_tax_adv: bool) -> str:
        if is_tax_adv:
            return "(tax-advantaged — no taxable event)"
        lots = tax_lots[
            (tax_lots["ticker"] == ticker) & (tax_lots["account"] == account)
        ] if not tax_lots.empty else pd.DataFrame()
        if lots.empty:
            return ""
        lt = lots[lots["is_long_term"]]
        st_lots = lots[~lots["is_long_term"]]
        lt_gain = lt["unrealized_gain"].sum() if not lt.empty else 0
        st_gain = st_lots["unrealized_gain"].sum() if not st_lots.empty else 0
        lt_shares = lt["shares"].sum() if not lt.empty else 0
        st_shares = st_lots["shares"].sum() if not st_lots.empty else 0
        parts = []
        if lt_shares > 0:
            g = "gain" if lt_gain >= 0 else "loss"
            parts.append(f"LT: {lt_shares:.4f} shares, {fmtd(lt_gain, sign=True)} {g}")
        if st_shares > 0:
            g = "gain" if st_gain >= 0 else "loss"
            parts.append(f"ST: {st_shares:.4f} shares, {fmtd(st_gain, sign=True)} {g}")
        return f"({'; '.join(parts)})" if parts else ""

    # Find misplaced holdings and suggest cross-account swaps.
    # You can't transfer between brokerages — instead, sell in one account
    # and use the proceeds to buy the other ticker (and vice versa).
    # No extra cash needed: each account sells one and buys the other.

    # Collect misplaced holdings into two buckets:
    #   wants_taxable: currently in tax-advantaged, should be in taxable
    #   wants_ta:      currently in taxable, should be in tax-advantaged
    wants_taxable = []  # (ticker, account, market_value, row)
    wants_ta = []       # (ticker, account, market_value, row)

    for _, h in holdings.iterrows():
        cat = h["category"]
        if not cat or h["shares"] <= 0:
            continue
        pref = CATEGORY_PREFERRED_LOCATION.get(cat, "either")
        if pref == "taxable" and h["is_tax_adv"]:
            wants_taxable.append((h["ticker"], h["account"], h["market_value"], h))
        elif pref == "tax_advantaged" and not h["is_tax_adv"]:
            wants_ta.append((h["ticker"], h["account"], h["market_value"], h))

    # Match pairs for swaps — greedy by smaller value
    matched_wt = set()
    matched_ta = set()

    for i, (t_tick, t_acct, t_val, t_row) in enumerate(wants_taxable):
        if i in matched_wt:
            continue
        for j, (ta_tick, ta_acct, ta_val, ta_row) in enumerate(wants_ta):
            if j in matched_ta:
                continue
            if t_tick == ta_tick:
                continue  # same ticker, not a swap
            swap_val = min(t_val, ta_val)
            if swap_val < 50:
                continue  # too small

            note_taxable = _tax_impact_note(ta_tick, ta_acct, False)
            suggestions.append(
                f"**Swap** ~{fmtd(swap_val, decimals=0)} (no extra cash needed):\n"
                f"  - In **{fmt_acct(t_acct)}**: sell {fmtd(swap_val, decimals=0)} of **{t_tick}**, "
                f"buy {fmtd(swap_val, decimals=0)} of **{ta_tick}**\n"
                f"  - In **{fmt_acct(ta_acct)}**: sell {fmtd(swap_val, decimals=0)} of **{ta_tick}**, "
                f"buy {fmtd(swap_val, decimals=0)} of **{t_tick}** {note_taxable}"
            )

            # Short-term gains warning for the taxable side
            if not tax_lots.empty:
                acct_lots = tax_lots[
                    (tax_lots["ticker"] == ta_tick) & (tax_lots["account"] == ta_acct)
                ]
                st_gain_lots = acct_lots[
                    (~acct_lots["is_long_term"]) & (acct_lots["unrealized_gain"] > 0)
                ] if not acct_lots.empty else pd.DataFrame()
                if not st_gain_lots.empty:
                    st_gain_total = st_gain_lots["unrealized_gain"].sum()
                    warnings.append(
                        f"⚠️ Selling **{ta_tick}** in **{fmt_acct(ta_acct)}** would realize "
                        f"**{fmtd(st_gain_total)} in short-term gains**. "
                        f"Consider waiting {365 - st_gain_lots['holding_days'].min()} more days."
                    )

            matched_wt.add(i)
            matched_ta.add(j)
            break

    # Unmatched: no swap partner, but cash is fungible across accounts
    for i, (tick, acct, val, row) in enumerate(wants_taxable):
        if i in matched_wt:
            continue
        taxable_accts = accounts_df[accounts_df["taxable_type"] == "taxable"]["name"].tolist()
        target = taxable_accts[0] if taxable_accts else "a taxable account"
        note = _tax_impact_note(tick, acct, True)
        suggestions.append(
            f"**{tick}** ({fmtd(val, decimals=0)}) prefers taxable:\n"
            f"  - Sell in **{fmt_acct(acct)}**, buy in **{fmt_acct(target)}** {note}"
        )

    for j, (tick, acct, val, row) in enumerate(wants_ta):
        if j in matched_ta:
            continue
        ta_accts = accounts_df[accounts_df["taxable_type"].isin(TAX_ADVANTAGED)]["name"].tolist()
        target = ta_accts[0] if ta_accts else "a tax-advantaged account"
        note = _tax_impact_note(tick, acct, False)
        suggestions.append(
            f"**{tick}** ({fmtd(val, decimals=0)}) prefers tax-advantaged:\n"
            f"  - Sell in **{fmt_acct(acct)}**, buy in **{fmt_acct(target)}** {note}"
        )
        if not tax_lots.empty:
            acct_lots = tax_lots[
                (tax_lots["ticker"] == tick) & (tax_lots["account"] == acct)
            ]
            st_gain_lots = acct_lots[
                (~acct_lots["is_long_term"]) & (acct_lots["unrealized_gain"] > 0)
            ] if not acct_lots.empty else pd.DataFrame()
            if not st_gain_lots.empty:
                st_gain_total = st_gain_lots["unrealized_gain"].sum()
                warnings.append(
                    f"⚠️ Selling **{tick}** in **{fmt_acct(acct)}** would realize "
                    f"**{fmtd(st_gain_total)} in short-term gains**. "
                    f"Consider waiting {365 - st_gain_lots['holding_days'].min()} more days."
                )

    # Consolidation: same ticker in multiple accounts of the same tax group
    # Recommend sell/rebuy instead of impossible direct transfers
    for ticker in holdings["ticker"].unique():
        t_holdings = holdings[holdings["ticker"] == ticker]
        for group in t_holdings["tax_group"].unique():
            group_holdings = t_holdings[t_holdings["tax_group"] == group]
            if len(group_holdings) > 1:
                target_row = group_holdings.loc[group_holdings["shares"].idxmax()]
                for _, h in group_holdings.iterrows():
                    if h["account"] != target_row["account"] and h["shares"] > 0:
                        note = _tax_impact_note(h["ticker"], h["account"], h["is_tax_adv"])
                        suggestions.append(
                            f"**Consolidate {ticker}**: sell {h['shares']:.4f} shares "
                            f"in **{fmt_acct(h['account'])}**, buy in **{fmt_acct(target_row['account'])}** {note}"
                        )

    # Tax-loss harvesting opportunities in taxable accounts
    if not tax_lots.empty:
        taxable_lots = tax_lots.copy()
        taxable_acct_names = accounts_df[accounts_df["taxable_type"] == "taxable"]["name"].tolist()
        taxable_lots = taxable_lots[taxable_lots["account"].isin(taxable_acct_names)]
        loss_lots = taxable_lots[taxable_lots["unrealized_gain"] < -50]  # meaningful losses only
        if not loss_lots.empty:
            st.markdown("**Tax-Loss Harvesting Opportunities**")
            for _, lot in loss_lots.iterrows():
                lt_label = "long-term" if lot["is_long_term"] else "short-term"
                st.markdown(
                    f"💰 **{lot['ticker']}** in **{fmt_acct(lot['account'])}**: "
                    f"{lot['shares']:.4f} shares with {fmtd(lot['unrealized_gain'])} "
                    f"unrealized {lt_label} loss (bought {lot['buy_date']})"
                )

    if warnings:
        st.markdown("**Capital Gains Warnings**")
        for w in warnings:
            st.markdown(w)

    if suggestions:
        st.markdown("**Suggested Moves**")
        for s in suggestions:
            st.markdown(f"→ {s}")
    elif not warnings:
        st.success("Portfolio looks well-balanced! No moves suggested.")
