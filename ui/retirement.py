import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data.data_manager import load_accounts
from data.portfolio import compute_holdings
from ui.fmt import fmtd
from data.retirement_sim import (
    RetirementParams,
    run_simple_projection,
    run_tax_optimized,
    run_simple_monte_carlo,
    run_tax_optimized_monte_carlo,
)

# Account type → tax bucket mapping
BUCKET_MAP = {
    "taxable": "taxable",
    "roth_ira": "roth",
    "401k_roth": "roth",
    "hsa": "roth",
    "401k_pretax": "pretax",
    "401k_aftertax": "aftertax",
}


def _get_account_balances() -> dict[str, float]:
    """Aggregate current holdings into tax buckets."""
    holdings = compute_holdings()
    accounts_df = load_accounts()
    if holdings.empty or accounts_df.empty:
        return {"taxable": 0, "pretax": 0, "roth": 0, "aftertax": 0}

    acct_type_map = dict(zip(accounts_df["name"], accounts_df["taxable_type"]))
    holdings["bucket"] = holdings["account"].map(acct_type_map).map(BUCKET_MAP).fillna("taxable")
    buckets = holdings.groupby("bucket")["market_value"].sum().to_dict()
    return {
        "taxable": buckets.get("taxable", 0),
        "pretax": buckets.get("pretax", 0),
        "roth": buckets.get("roth", 0),
        "aftertax": buckets.get("aftertax", 0),
    }


def _get_roth_contributions() -> float:
    """Compute total direct contributions to Roth accounts from transactions.

    Roth contributions (not conversions) are always withdrawable tax- and
    penalty-free regardless of age.  This sums all 'contribution' type
    transactions in accounts mapped to the 'roth' bucket.
    """
    from data.data_manager import load_transactions
    accounts_df = load_accounts()
    if accounts_df.empty:
        return 0.0
    acct_type_map = dict(zip(accounts_df["name"], accounts_df["taxable_type"]))
    roth_accounts = [name for name, atype in acct_type_map.items()
                     if BUCKET_MAP.get(atype) == "roth"]
    if not roth_accounts:
        return 0.0
    txns = load_transactions()
    if txns.empty:
        return 0.0
    roth_contribs = txns[
        (txns["account"].isin(roth_accounts)) &
        (txns["type"] == "contribution")
    ]
    if roth_contribs.empty:
        return 0.0
    return float(roth_contribs["amount"].astype(float).sum())


def _render_mc_results(mc_results: dict, retirement_age: int, lifespan: int,
                       params: RetirementParams, dollar_label: str, prefix: str):
    """Shared renderer for Monte Carlo results (fan chart, table, depletion stats)."""
    n_sims = mc_results["n_sims"]
    success_pct = mc_results["success_rate"] * 100
    if success_pct >= 90:
        st.success(f"**Success Rate: {success_pct:.1f}%** — Portfolio lasts to age {lifespan} "
                   f"in {success_pct:.0f}% of {n_sims} simulations.")
    elif success_pct >= 70:
        st.warning(f"**Success Rate: {success_pct:.1f}%** — Consider reducing withdrawal or "
                   f"increasing savings.")
    else:
        st.error(f"**Success Rate: {success_pct:.1f}%** — High risk of running out of money.")

    ages = mc_results["ages"]
    vol_label = ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages, y=mc_results["p90"], mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=mc_results["p10"], mode="lines",
        fill="tonexty", fillcolor="rgba(99,110,250,0.1)",
        line=dict(width=0), name="10th–90th percentile",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=mc_results["p75"], mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=mc_results["p25"], mode="lines",
        fill="tonexty", fillcolor="rgba(99,110,250,0.2)",
        line=dict(width=0), name="25th–75th percentile",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=mc_results["p50"], mode="lines",
        line=dict(color="#636EFA", width=2), name="Median",
    ))
    fig.add_vline(x=retirement_age, line_dash="dash", line_color="gray",
                   annotation_text="Retirement")
    fig.update_layout(
        title=f"Monte Carlo ({n_sims} sims)",
        xaxis_title="Age", yaxis_title=f"Portfolio Value ({dollar_label})",
        yaxis_tickformat="$,.0f", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Percentile table at key ages
    st.markdown(f"**Percentile Table** ({n_sims} simulations)")
    key_ages = [retirement_age, 70, 75, 80, 85, lifespan]
    key_ages = sorted(set(a for a in key_ages if params.current_age <= a <= lifespan))
    rows = []
    for a in key_ages:
        idx = a - params.current_age
        if 0 <= idx < len(ages):
            rows.append({
                "Age": a,
                "10th percentile": mc_results["p10"][idx],
                "25th percentile": mc_results["p25"][idx],
                "Median": mc_results["p50"][idx],
                "75th percentile": mc_results["p75"][idx],
                "90th percentile": mc_results["p90"][idx],
            })
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True, hide_index=True,
            key=f"{prefix}_pct_table_{n_sims}",
            column_config={
                "10th percentile": st.column_config.NumberColumn(format="$%,.0f"),
                "25th percentile": st.column_config.NumberColumn(format="$%,.0f"),
                "Median": st.column_config.NumberColumn(format="$%,.0f"),
                "75th percentile": st.column_config.NumberColumn(format="$%,.0f"),
                "90th percentile": st.column_config.NumberColumn(format="$%,.0f"),
            },
        )

    # Depletion stats
    depleted = [a for a in mc_results["depleted_ages"] if a is not None]
    if depleted:
        st.markdown(f"**Depletion Stats** ({len(depleted)} of {n_sims} sims ran out)")
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Earliest Depletion", f"Age {min(depleted)}")
        dc2.metric("Median Depletion", f"Age {int(pd.Series(depleted).median())}")
        dc3.metric("Latest Depletion", f"Age {max(depleted)}")


def render():
    st.header("Retirement Planner")

    # ── Input Parameters ──────────────────────────────────────────────────────
    with st.expander("Settings", expanded=True):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Returns**")
            expected_return = st.number_input(
                "Expected real return (%)", min_value=0.0, max_value=30.0,
                value=5.0, step=0.5, key="ret_return",
                help="Inflation-adjusted. All values in today's dollars."
            ) / 100

            st.markdown("**Age**")
            c1, c2, c3 = st.columns(3)
            current_age = c1.number_input("Current", min_value=18, max_value=100, value=30, key="ret_age")
            retirement_age = c2.number_input("Retire at", min_value=18, max_value=100, value=50, key="ret_retire_age")
            lifespan = c3.number_input("Plan to age", min_value=50, max_value=120, value=90, key="ret_lifespan")

        with col_b:
            st.markdown("**Pre-Retirement Income**")
            annual_income = st.number_input(
                "Annual income ($)", min_value=0.0, value=100000.0, step=5000.0,
                key="ret_annual_income",
                help="W-2/salary income before retirement. Sets the marginal tax bracket "
                     "for Roth conversions during accumulation. Tax on this income itself "
                     "is excluded from the simulation."
            )

            st.markdown("**Withdrawal**")
            wd_mode = st.radio("Withdrawal mode", ["Rate (%)", "Fixed amount ($)"], horizontal=True, key="ret_wd_mode")
            if wd_mode == "Rate (%)":
                withdrawal_rate = st.number_input(
                    "Withdrawal rate (%)", min_value=0.0, max_value=20.0,
                    value=3.5, step=0.5, key="ret_wd_rate"
                ) / 100
                withdrawal_amount = None
            else:
                withdrawal_amount = st.number_input(
                    "Annual withdrawal ($)", min_value=0.0, value=60000.0,
                    step=5000.0, key="ret_wd_amount"
                )
                withdrawal_rate = 0.04  # unused but need default

            st.markdown("**Social Security**")
            sc1, sc2 = st.columns(2)
            ss_annual = sc1.number_input(
                "Annual SS ($)", min_value=0.0, value=0.0, step=1000.0, key="ret_ss"
            )
            ss_start_age = sc2.number_input(
                "SS start age", min_value=62, max_value=70, value=67, key="ret_ss_age"
            )

    # ── Account Balances ──────────────────────────────────────────────────────
    balances = _get_account_balances()
    roth_contribs_from_portfolio = _get_roth_contributions()

    total_bal_placeholder = st.empty()
    with st.expander("Account Balances (auto-populated from portfolio)"):
        bc1, bc2, bc3, bc4 = st.columns(4)
        taxable_bal = bc1.number_input("Taxable", value=float(balances["taxable"]), step=1000.0, key="ret_taxable")
        pretax_bal = bc2.number_input("Pre-Tax", value=float(balances["pretax"]), step=1000.0, key="ret_pretax")
        roth_bal = bc3.number_input("Roth", value=float(balances["roth"]), step=1000.0, key="ret_roth")
        aftertax_bal = bc4.number_input("After-Tax", value=float(balances["aftertax"]), step=1000.0, key="ret_aftertax")

        total_bal = taxable_bal + pretax_bal + roth_bal + aftertax_bal

        cost_basis_pct = st.slider(
            "Taxable cost basis %", min_value=0, max_value=100, value=50, key="ret_cb_pct"
        ) / 100

        roth_contribs = st.number_input(
            "Roth contributions (cost basis)",
            value=float(roth_contribs_from_portfolio),
            step=1000.0, key="ret_roth_contribs",
            help="Total direct Roth contributions (always withdrawable tax-free). "
                 "Auto-populated from portfolio transactions."
        )

    # ── Annual Contributions ──────────────────────────────────────────────────
    total_contrib_placeholder = st.empty()
    with st.expander("Annual Contributions (stop at retirement)"):
        st.caption("Enter annual amounts — applied once per year in the simulation. Contributions stop at retirement.")
        cc1, cc2, cc3, cc4 = st.columns(4)
        contrib_taxable = cc1.number_input("Taxable", min_value=0.0, value=0.0, step=1000.0, key="ret_c_taxable")
        contrib_pretax = cc2.number_input("Pre-Tax", min_value=0.0, value=30500.0, step=1000.0, key="ret_c_pretax",
                                              help="401k max ($24,500) + employer match ($6,000)")
        contrib_roth = cc3.number_input("Roth", min_value=0.0, value=53400.0, step=1000.0, key="ret_c_roth",
                                         help="Mega Backdoor Roth ($41,500) + Backdoor Roth IRA ($7,500) + HSA ($4,400)")
        contrib_aftertax = cc4.number_input("After-Tax", min_value=0.0, value=0.0, step=1000.0, key="ret_c_aftertax")
        total_contrib = contrib_taxable + contrib_pretax + contrib_roth + contrib_aftertax

    total_bal_placeholder.markdown(f"**Total Portfolio Balance: {fmtd(total_bal, decimals=0)}**")
    total_contrib_placeholder.markdown(f"**Total Annual Contributions: {fmtd(total_contrib, decimals=0)}**")

    # Build params
    params = RetirementParams(
        current_age=current_age,
        retirement_age=retirement_age,
        lifespan=lifespan,
        expected_return=expected_return,
        withdrawal_rate=withdrawal_rate,
        withdrawal_amount=withdrawal_amount,
        social_security_annual=ss_annual,
        social_security_start_age=ss_start_age,
        taxable_balance=taxable_bal,
        pretax_balance=pretax_bal,
        roth_balance=roth_bal,
        aftertax_balance=aftertax_bal,
        taxable_cost_basis_pct=cost_basis_pct,
        annual_income=annual_income,
        contribution_taxable=contrib_taxable,
        contribution_pretax=contrib_pretax,
        contribution_roth=contrib_roth,
        contribution_aftertax=contrib_aftertax,
        roth_contributions=roth_contribs,
    )

    # Show cross-reference: rate ↔ amount
    cur_total = params.initial_total
    annual_contrib = contrib_taxable + contrib_pretax + contrib_roth + contrib_aftertax
    years_to_retire = max(retirement_age - current_age, 0)
    if wd_mode == "Rate (%)":
        # Estimate balance at retirement: grow current balance + contributions
        est_retire_bal = cur_total * (1 + expected_return) ** years_to_retire
        if expected_return > 0 and years_to_retire > 0:
            est_retire_bal += annual_contrib * ((1 + expected_return) ** years_to_retire - 1) / expected_return
        else:
            est_retire_bal += annual_contrib * years_to_retire
        equiv_amount = est_retire_bal * withdrawal_rate
        st.caption(
            f"Withdrawal: {withdrawal_rate*100:.1f}% of estimated retirement balance "
            f"{fmtd(est_retire_bal, decimals=0)} = **{fmtd(equiv_amount, decimals=0)}/yr**"
        )
    else:
        st.caption(f"Withdrawal: **{fmtd(withdrawal_amount, decimals=0)}/yr**")

    dollar_label = "real dollars"

    # ── Simulation Tabs ───────────────────────────────────────────────────────
    sim1, sim2, sim3, sim4 = st.tabs([
        "Simple Projection", "Simple Monte Carlo",
        "Tax-Optimized", "Tax-Optimized Monte Carlo",
    ])

    # ── Sim 1: Simple ─────────────────────────────────────────────────────────
    with sim1:
        st.subheader("Simple Flat Projection")
        st.caption(f"Flat {expected_return*100:.1f}% annual return, {dollar_label}.")

        simple_results = run_simple_projection(params)
        df_simple = pd.DataFrame([vars(r) for r in simple_results])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_simple["age"], y=df_simple["total"],
            mode="lines", name="Portfolio Value",
            line=dict(width=2),
        ))
        fig.add_vline(x=retirement_age, line_dash="dash", line_color="gray",
                       annotation_text="Retirement")
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Age", yaxis_title=f"Value ({dollar_label})",
            yaxis_tickformat="$,.0f", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        retire_row = next((r for r in simple_results if r.age == retirement_age), None)
        end_row = simple_results[-1]
        mc1, mc2, mc3 = st.columns(3)
        if retire_row:
            mc1.metric("At Retirement", fmtd(retire_row.total, decimals=0))
        mc2.metric(f"At Age {lifespan}", fmtd(end_row.total, decimals=0))
        if retire_row and retire_row.total > 0:
            annual_wd = retire_row.total * withdrawal_rate if withdrawal_amount is None else withdrawal_amount
            mc3.metric("Annual Withdrawal", fmtd(annual_wd, decimals=0))

        with st.expander("Year-by-Year Table"):
            st.dataframe(
                df_simple[["age", "total", "contributions", "withdrawal", "ss_income", "net_income", "notes"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "total": st.column_config.NumberColumn("Portfolio", format="$%,.0f"),
                    "contributions": st.column_config.NumberColumn("Contributions", format="$%,.0f"),
                    "withdrawal": st.column_config.NumberColumn("Withdrawal", format="$%,.0f"),
                    "ss_income": st.column_config.NumberColumn("Social Security", format="$%,.0f"),
                    "net_income": st.column_config.NumberColumn("Net Income", format="$%,.0f"),
                },
            )

    # ── Sim 2: Simple Monte Carlo ─────────────────────────────────────────────
    with sim2:
        st.subheader("Simple Monte Carlo")
        st.caption("Monte Carlo on the simple projection — randomized annual returns, no tax optimization.")

        smc_col1, smc_col2 = st.columns(2)
        smc_n_str = smc_col1.text_input("Simulations", value="500", key="ret_smc_n")
        try:
            smc_n = max(1, int(smc_n_str))
        except ValueError:
            smc_n = 500
            st.error("Invalid number of simulations, using 500.")
        smc_vol = smc_col2.number_input("Annual volatility (%)", min_value=1.0, max_value=50.0,
                                         value=15.0, step=1.0, key="ret_smc_vol") / 100

        if st.button("Run Simple Monte Carlo", key="ret_smc_run"):
            with st.spinner(f"Running {smc_n} simple simulations..."):
                st.session_state["smc_results"] = run_simple_monte_carlo(params, n_sims=smc_n, volatility=smc_vol)
        if "smc_results" in st.session_state:
            _render_mc_results(st.session_state["smc_results"], retirement_age, lifespan, params, dollar_label, "smc")

    # ── Sim 3: Tax-Optimized ──────────────────────────────────────────────────
    with sim3:
        st.subheader("Tax-Optimized Projection")
        st.caption(
            "Optimizes withdrawal ordering and Roth conversion ladder. "
            "Uses current CA + federal tax brackets."
        )

        tax_results = run_tax_optimized(params)
        df_tax = pd.DataFrame([vars(r) for r in tax_results])

        # Stacked area chart by account type
        fig2 = go.Figure()
        for col, name in [
            ("taxable", "Taxable"), ("pretax", "Pre-Tax"),
            ("roth", "Roth"), ("aftertax", "After-Tax"),
        ]:
            fig2.add_trace(go.Scatter(
                x=df_tax["age"], y=df_tax[col],
                mode="lines", name=name, stackgroup="one",
                line=dict(width=0.5),
            ))
        fig2.add_vline(x=retirement_age, line_dash="dash", line_color="gray",
                        annotation_text="Retirement")
        fig2.update_layout(
            title="Portfolio by Account Type",
            xaxis_title="Age", yaxis_title=f"Value ({dollar_label})",
            yaxis_tickformat="$,.0f", height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Tax & conversion chart
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df_tax["age"], y=df_tax["taxes_paid"],
            name="Taxes Paid", marker_color="#EF553B",
        ))
        fig3.add_trace(go.Bar(
            x=df_tax["age"], y=df_tax["roth_conversion"],
            name="Roth Conversion", marker_color="#00CC96",
        ))
        fig3.update_layout(
            title="Taxes Paid & Roth Conversions",
            xaxis_title="Age", yaxis_title=f"Amount ({dollar_label})",
            yaxis_tickformat="$,.0f", barmode="group", height=350,
        )
        st.plotly_chart(fig3, use_container_width=True)

        retire_row_t = next((r for r in tax_results if r.age == retirement_age), None)
        end_row_t = tax_results[-1]
        total_taxes = sum(r.taxes_paid for r in tax_results)
        total_conversions = sum(r.roth_conversion for r in tax_results)

        tc1, tc2, tc3, tc4 = st.columns(4)
        if retire_row_t:
            tc1.metric("At Retirement", fmtd(retire_row_t.total, decimals=0))
        tc2.metric(f"At Age {lifespan}", fmtd(end_row_t.total, decimals=0))
        tc3.metric("Lifetime Taxes", fmtd(total_taxes, decimals=0))
        tc4.metric("Total Roth Converted", fmtd(total_conversions, decimals=0))

        with st.expander("Decision Walkthrough"):
            st.caption(
                "Year-by-year optimization decisions. Each entry explains what "
                "the optimizer did and why (withdrawal source, tax rate, "
                "Roth conversion rationale)."
            )
            for r in tax_results:
                if not r.notes:
                    continue
                parts = r.notes.split(" | ")
                phase = "🟢 Accumulation" if r.age < retirement_age else "🔴 Withdrawal"
                st.markdown(f"**Age {r.age}** ({phase})")
                for part in parts:
                    st.markdown(f"  - {part}")
                if r.withdrawal > 0 or r.roth_conversion > 0:
                    sources = []
                    for label, val in [("Taxable", r.wd_taxable), ("Pre-Tax", r.wd_pretax),
                                        ("Roth", r.wd_roth), ("After-Tax", r.wd_aftertax),
                                        ("Roth Conv.", r.roth_conversion)]:
                        if val > 0:
                            sources.append(f"{label} {fmtd(val, decimals=0)}")
                    if sources:
                        st.markdown(f"  - **Withdrawal sources:** {' · '.join(sources)}")

        with st.expander("Year-by-Year Table"):
            st.dataframe(
                df_tax[["age", "taxable", "pretax", "roth", "aftertax", "total",
                         "contributions", "withdrawal",
                         "wd_taxable", "wd_pretax", "wd_roth", "wd_aftertax",
                         "roth_conversion", "taxes_paid", "ss_income", "net_income"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "taxable": st.column_config.NumberColumn("Taxable", format="$%,.0f"),
                    "pretax": st.column_config.NumberColumn("Pre-Tax", format="$%,.0f"),
                    "roth": st.column_config.NumberColumn("Roth", format="$%,.0f"),
                    "aftertax": st.column_config.NumberColumn("After-Tax", format="$%,.0f"),
                    "total": st.column_config.NumberColumn("Total", format="$%,.0f"),
                    "contributions": st.column_config.NumberColumn("Contributions", format="$%,.0f"),
                    "withdrawal": st.column_config.NumberColumn("Withdrawal", format="$%,.0f"),
                    "wd_taxable": st.column_config.NumberColumn("WD Taxable", format="$%,.0f"),
                    "wd_pretax": st.column_config.NumberColumn("WD Pre-Tax", format="$%,.0f"),
                    "wd_roth": st.column_config.NumberColumn("WD Roth", format="$%,.0f"),
                    "wd_aftertax": st.column_config.NumberColumn("WD After-Tax", format="$%,.0f"),
                    "roth_conversion": st.column_config.NumberColumn("Roth Conv.", format="$%,.0f"),
                    "taxes_paid": st.column_config.NumberColumn("Taxes", format="$%,.0f"),
                    "ss_income": st.column_config.NumberColumn("SS", format="$%,.0f"),
                    "net_income": st.column_config.NumberColumn("Net Income", format="$%,.0f"),
                },
            )

    # ── Sim 4: Tax-Optimized Monte Carlo ──────────────────────────────────────
    with sim4:
        st.subheader("Tax-Optimized Monte Carlo")
        st.caption(
            "Monte Carlo with DP tax optimization — backward pass computes future tax costs, "
            "forward pass optimizes withdrawal ordering and Roth conversions. Handles depletion."
        )

        tmc_col1, tmc_col2 = st.columns(2)
        tmc_n_str = tmc_col1.text_input("Simulations", value="500", key="ret_tmc_n")
        try:
            tmc_n = max(1, int(tmc_n_str))
        except ValueError:
            tmc_n = 500
            st.error("Invalid number of simulations, using 500.")
        tmc_vol = tmc_col2.number_input("Annual volatility (%)", min_value=1.0, max_value=50.0,
                                         value=15.0, step=1.0, key="ret_tmc_vol") / 100

        if st.button("Run Tax-Optimized Monte Carlo", key="ret_tmc_run"):
            with st.spinner(f"Running {tmc_n} tax-optimized simulations..."):
                st.session_state["tmc_results"] = run_tax_optimized_monte_carlo(params, n_sims=tmc_n, volatility=tmc_vol)
        if "tmc_results" in st.session_state:
            tmc_results = st.session_state["tmc_results"]
            _render_mc_results(tmc_results, retirement_age, lifespan, params, dollar_label, "tmc")

            # Extra: lifetime tax stats
            tax_totals = tmc_results["total_taxes_per_sim"]
            n_tmc = tmc_results["n_sims"]
            st.markdown("**Lifetime Tax Statistics**")
            txc1, txc2, txc3 = st.columns(3)
            txc1.metric("Median Lifetime Taxes", fmtd(np.median(tax_totals), decimals=0))
            txc2.metric("25th percentile", fmtd(np.percentile(tax_totals, 25), decimals=0))
            txc3.metric("75th percentile", fmtd(np.percentile(tax_totals, 75), decimals=0))

            # Shortfall / depletion stats
            shortfall_totals = tmc_results.get("total_shortfall_per_sim", [])
            shortfall_years = tmc_results.get("shortfall_years_per_sim", [])
            sims_with_shortfall = sum(1 for s in shortfall_totals if s > 0)
            if sims_with_shortfall > 0:
                st.markdown("**Withdrawal Shortfall Statistics**")
                st.warning(
                    f"{sims_with_shortfall} of {n_tmc} simulations "
                    f"({sims_with_shortfall/n_tmc*100:.1f}%) had years where "
                    f"the portfolio couldn't fully meet the withdrawal target."
                )
                sc1, sc2, sc3 = st.columns(3)
                shortfall_nonzero = [s for s in shortfall_totals if s > 0]
                sc1.metric("Median Shortfall (of failed)", fmtd(np.median(shortfall_nonzero), decimals=0))
                years_nonzero = [y for y in shortfall_years if y > 0]
                sc2.metric("Median Shortfall Years", f"{np.median(years_nonzero):.0f}")
                sc3.metric("Max Shortfall Years", f"{max(years_nonzero):.0f}")
            else:
                st.success("All simulations met withdrawal targets every year.")
