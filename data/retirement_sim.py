"""Retirement simulation engine.

Supports:
1. Simple flat projection
2. Tax-optimized with Roth conversion ladder (CA + federal)
3. Monte Carlo simulation of #2
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


# ── 2026 Federal Tax Brackets (Single) ────────────────────────────────────────
FED_BRACKETS = [
    (12400, 0.10),
    (50400, 0.12),
    (105700, 0.22),
    (201775, 0.24),
    (256225, 0.32),
    (640600, 0.35),
    (np.inf, 0.37),
]

# ── 2025 CA State Tax Brackets (Single, latest FTB-published) ────────────────
CA_BRACKETS = [
    (11079, 0.01),
    (26264, 0.02),
    (41452, 0.04),
    (57542, 0.06),
    (72724, 0.08),
    (371479, 0.093),
    (445771, 0.103),
    (742953, 0.113),
    (np.inf, 0.123),
]

FED_STANDARD_DEDUCTION = 16100
CA_STANDARD_DEDUCTION = 5706

# Long-term capital gains brackets (single, 2026)
FED_LTCG_BRACKETS = [
    (49450, 0.0),
    (545500, 0.15),
    (np.inf, 0.20),
]


def _compute_bracket_tax(taxable_income: float, brackets: list[tuple]) -> float:
    """Compute tax from a progressive bracket schedule."""
    tax = 0.0
    prev = 0.0
    for top, rate in brackets:
        if taxable_income <= prev:
            break
        taxed = min(taxable_income, top) - prev
        tax += taxed * rate
        prev = top
    return tax


def compute_federal_tax(ordinary_income: float, ltcg: float = 0) -> float:
    """Federal tax on ordinary income + LTCG."""
    agi = max(ordinary_income - FED_STANDARD_DEDUCTION, 0)
    fed_ordinary = _compute_bracket_tax(agi, FED_BRACKETS)
    fed_cg = _compute_bracket_tax(agi + ltcg, FED_LTCG_BRACKETS) - _compute_bracket_tax(agi, FED_LTCG_BRACKETS)
    return fed_ordinary + fed_cg


def compute_ca_tax(ordinary_income: float) -> float:
    """CA state tax on ordinary income (CA taxes LTCG as ordinary)."""
    agi = max(ordinary_income - CA_STANDARD_DEDUCTION, 0)
    return _compute_bracket_tax(agi, CA_BRACKETS)


def compute_total_tax(ordinary_income: float, ltcg: float = 0) -> float:
    """Federal + CA tax. CA taxes cap gains as ordinary income."""
    return compute_federal_tax(ordinary_income, ltcg) + compute_ca_tax(ordinary_income + ltcg)


@dataclass
class RetirementParams:
    current_age: int = 30
    retirement_age: int = 65
    lifespan: int = 90
    expected_return: float = 0.07  # real (inflation-adjusted)
    # Withdrawal: specify rate OR amount, not both
    withdrawal_rate: float = 0.035
    withdrawal_amount: float | None = None
    social_security_annual: float = 0
    social_security_start_age: int = 67
    # Account balances by tax bucket
    taxable_balance: float = 0
    pretax_balance: float = 0  # traditional IRA / 401k pretax
    roth_balance: float = 0  # roth IRA / 401k roth / HSA
    aftertax_balance: float = 0  # 401k after-tax
    # Cost basis for taxable (fraction that is basis, not gains)
    taxable_cost_basis_pct: float = 0.5
    # Pre-retirement annual income (W-2 etc.) — used to set the correct
    # marginal tax bracket during accumulation.  The tax ON this income is
    # ignored (non-marginal); only its bracket-positioning effect matters.
    annual_income: float = 0
    # Annual contributions (added yearly, stop at retirement)
    contribution_taxable: float = 0
    contribution_pretax: float = 0
    contribution_roth: float = 0
    contribution_aftertax: float = 0
    # Roth IRA withdrawal ordering:
    #   1. Direct contributions — always accessible, tax/penalty-free
    #   2. Conversions — tax-free after 5 years per conversion
    #   3. Earnings — tax/penalty-free only after age 59½ AND first
    #      Roth contribution was ≥5 years ago
    # roth_contributions: total direct contributions (always accessible)
    roth_contributions: float = 0
    # roth_conversions_by_year: list of (year, amount) for each past
    #   conversion, used to compute the 5-year maturity clock.
    #   Year = calendar year the conversion occurred.
    roth_conversions_by_year: list = field(default_factory=list)

    @property
    def initial_total(self) -> float:
        return self.taxable_balance + self.pretax_balance + self.roth_balance + self.aftertax_balance

    def fixed_withdrawal_for(self, balance_at_retirement: float) -> float:
        """Compute fixed annual withdrawal in real dollars.
        Rate mode: rate × balance at retirement (fixed once).
        Amount mode: use the specified amount directly."""
        if self.withdrawal_amount is not None:
            return self.withdrawal_amount
        return balance_at_retirement * self.withdrawal_rate


@dataclass
class YearResult:
    age: int
    year: int
    taxable: float = 0
    pretax: float = 0
    roth: float = 0
    aftertax: float = 0
    total: float = 0
    gross_withdrawal: float = 0  # 3.5% × portfolio at retirement (year 1 pull)
    spending: float = 0          # standard of living = year1 pull - year1 taxes (flat forever)
    withdrawal: float = 0        # actual pull from portfolio this year (drops when SS starts)
    wd_taxable: float = 0
    wd_pretax: float = 0
    wd_roth: float = 0
    wd_aftertax: float = 0
    roth_conversion: float = 0
    taxes_paid: float = 0        # total taxes (spending + conversion)
    spending_taxes: float = 0    # taxes from withdrawal income only
    conversion_taxes: float = 0  # taxes from Roth conversions (deducted from portfolio)
    ss_income: float = 0
    net_income: float = 0        # spending + SS (what you actually live on)
    contributions: float = 0
    notes: str = ""


# ── Simulation 1: Simple Flat Projection ─────────────────────────────────────

def run_simple_projection(p: RetirementParams) -> list[YearResult]:
    """Simple projection: annual compounding, contributions pre-retirement."""
    annual_return = p.expected_return
    total = p.taxable_balance + p.pretax_balance + p.roth_balance + p.aftertax_balance
    annual_contrib = (p.contribution_taxable + p.contribution_pretax +
                      p.contribution_roth + p.contribution_aftertax)

    fixed_wd = None  # computed once at retirement
    results = []
    for year_offset in range(p.lifespan - p.current_age + 1):
        age = p.current_age + year_offset
        is_retired = age >= p.retirement_age

        ss = p.social_security_annual if age >= p.social_security_start_age else 0
        contrib = annual_contrib if not is_retired else 0

        if is_retired:
            if fixed_wd is None:
                fixed_wd = p.fixed_withdrawal_for(total)
            withdrawal = max(fixed_wd - ss, 0)
        else:
            withdrawal = 0

        # Grow, contribute, withdraw — all annual
        total = total * (1 + annual_return) + contrib - withdrawal
        total = max(total, 0)

        results.append(YearResult(
            age=age, year=year_offset,
            total=round(total, 2),
            withdrawal=round(withdrawal, 2),
            contributions=round(contrib, 2),
            ss_income=round(ss, 2),
            net_income=round(withdrawal + ss, 2),
            notes="accumulation" if not is_retired else "withdrawal",
        ))

    return results


# ── DP Tax Optimizer ─────────────────────────────────────────────────────────
#
# Two-pass approach:
#   Pass 1 (backward): compute the marginal tax rate at which each bucket
#          will be drawn down in each future year ("shadow price").
#   Pass 2 (forward):  use those shadow prices to decide:
#          (a) optimal withdrawal ordering — cheapest effective rate first
#          (b) Roth conversion — convert when cost_now < shadow_price_pretax
#
# Objective: #1 meet withdrawal, #2 minimize lifetime taxes paid.

def _accessible_roth(
    roth_balance: float,
    roth_contributions: float,
    conversions_by_year: list[tuple[int, float]],
    current_year: int,
    age: int,
) -> float:
    """Compute how much of the Roth balance can be withdrawn tax/penalty-free.

    Roth IRA ordering:
      1. Direct contributions — always accessible
      2. Conversions — accessible (tax-free) after 5 years per conversion
      3. Earnings — accessible only if age >= 59.5 AND first Roth
         contribution/conversion was >= 5 years ago

    Returns the tax-free accessible amount (capped at roth_balance)."""
    if roth_balance <= 0:
        return 0.0

    accessible = roth_contributions  # always available

    # Add matured conversions (5-year rule)
    for conv_year, conv_amount in conversions_by_year:
        if current_year - conv_year >= 5:
            accessible += conv_amount

    # Earnings are accessible if age >= 59.5 (we use >= 60 as integer approx)
    if age >= 60:
        accessible = roth_balance  # everything accessible

    return min(accessible, roth_balance)


def _marginal_rate_at(ordinary_income: float, extra: float = 1000) -> float:
    """Marginal combined (fed+CA) tax rate at a given ordinary income level."""
    if extra <= 0:
        return 0.0
    return (compute_total_tax(ordinary_income + extra) -
            compute_total_tax(ordinary_income)) / extra


def _backward_pass(p: RetirementParams, annual_return: float,
                    pretax_trajectory: list[float] | None = None) -> list[dict]:
    """Compute shadow prices: the future marginal tax rate on pretax dollars.

    At each year, computes the marginal rate if the remaining pretax balance
    were spread evenly over remaining years + 10 (SECURE Act inherited IRA).

    If *pretax_trajectory* is provided (from a prior forward pass), it is used
    as the pretax balance at each year.  Otherwise, a no-conversion projection
    is used as the initial estimate.

    Returns list indexed by year_offset."""
    years = p.lifespan - p.current_age + 1
    cbp = p.taxable_cost_basis_pct
    retire_offset = max(p.retirement_age - p.current_age, 0)

    # ── Pretax balance trajectory ─────────────────────────────────────────
    if pretax_trajectory is not None:
        pretax_by_year = pretax_trajectory
    else:
        # No-conversion, no-withdrawal projection (worst-case pile)
        pretax = p.pretax_balance
        pretax_by_year = []
        for y in range(years):
            age = p.current_age + y
            if age < p.retirement_age:
                pretax += p.contribution_pretax
            pretax_by_year.append(pretax)
            pretax = pretax * (1 + annual_return)

    # ── Compute shadow price at each year ─────────────────────────────────
    # The shadow at year y is the weighted-average marginal rate across all
    # future years (y..lifespan + 10yr inherited IRA).  Each future year
    # has ordinary income = annual_pretax + SS(if applicable).  Years before
    # SS starts have lower ordinary income and thus lower marginal rates,
    # creating a conversion window.
    shadow = []
    for y in range(years):
        age = p.current_age + y

        if age < p.retirement_age:
            shadow.append(None)  # filled below
        else:
            years_left = p.lifespan - age
            effective_years = years_left + 10  # SECURE Act inherited IRA
            remaining_pretax = pretax_by_year[y] if y < len(pretax_by_year) else 0

            if remaining_pretax > 0 and effective_years > 0:
                r = annual_return
                if r > 0.001:
                    annual_pretax = remaining_pretax * r / (1 - (1 + r) ** -effective_years)
                else:
                    annual_pretax = remaining_pretax / effective_years

                # Weighted average rate across future years, accounting for
                # SS starting partway through retirement.
                sum_rate = 0.0
                sum_rate_at = 0.0
                sum_rate_tx = 0.0
                for fy in range(effective_years):
                    future_age = age + fy
                    ss_fy = p.social_security_annual if future_age >= p.social_security_start_age else 0
                    ord_fy = ss_fy + annual_pretax
                    sum_rate += _marginal_rate_at(ord_fy)
                    sum_rate_at += _marginal_rate_at(ord_fy) * 0.5
                    if remaining_pretax > 0:
                        sum_rate_tx += (compute_total_tax(ord_fy, 1000 * (1 - cbp)) -
                                        compute_total_tax(ord_fy, 0)) / 1000
                rate = sum_rate / effective_years
                rate_at = sum_rate_at / effective_years
                rate_tx = sum_rate_tx / effective_years
            else:
                rate = 0.0
                rate_at = 0.0
                rate_tx = 0.0

            shadow.append({
                "future_pretax_rate": rate,
                "future_aftertax_rate": rate_at,
                "future_taxable_rate": rate_tx,
                "future_roth_rate": 0.0,
            })

    # Fill accumulation years with the first retirement year's shadow
    retire_shadow = shadow[retire_offset] if retire_offset < len(shadow) and shadow[retire_offset] else {
        "future_pretax_rate": 0.15, "future_aftertax_rate": 0.075,
        "future_taxable_rate": 0.05, "future_roth_rate": 0.0,
    }
    for y in range(retire_offset):
        shadow[y] = retire_shadow

    return shadow


def _withdrawal_cost_per_dollar(
    source: str, amount: float, ss: float, ordinary_so_far: float,
    cost_basis_pct: float,
) -> float:
    """Effective tax rate per dollar withdrawn from a source this year."""
    if amount <= 0:
        return 0.0
    if source == "roth":
        return 0.0
    elif source == "taxable":
        gain = amount * (1 - cost_basis_pct)
        tax = compute_total_tax(ss, gain) - compute_total_tax(ss, 0)
        return tax / amount
    elif source == "aftertax":
        gain = amount * 0.5
        tax = (compute_total_tax(ordinary_so_far + gain) -
               compute_total_tax(ordinary_so_far))
        return tax / amount
    elif source == "pretax":
        tax = (compute_total_tax(ordinary_so_far + amount) -
               compute_total_tax(ordinary_so_far))
        return tax / amount
    return 0.0


def _deduct_tax_from_portfolio(
    tax: float, taxable: float, aftertax: float, roth: float,
) -> tuple[float, float, float, list[str]]:
    """Deduct a tax bill from portfolio balances.
    Priority: taxable → after-tax → roth (cheapest liquid sources first).
    Returns updated balances and explanatory notes."""
    notes = []
    remaining = tax
    if remaining > 0 and taxable > 0:
        take = min(remaining, taxable)
        taxable -= take
        remaining -= take
        notes.append(f"tax ${take:,.0f} from taxable")
    if remaining > 0 and aftertax > 0:
        take = min(remaining, aftertax)
        aftertax -= take
        remaining -= take
        notes.append(f"tax ${take:,.0f} from after-tax")
    if remaining > 0 and roth > 0:
        take = min(remaining, roth)
        roth -= take
        remaining -= take
        notes.append(f"tax ${take:,.0f} from roth")
    return taxable, aftertax, roth, notes


# ── Simulation 2: DP Tax-Optimized ───────────────────────────────────────────

def run_tax_optimized(p: RetirementParams) -> list[YearResult]:
    """DP tax-optimized simulation.

    Pass 1: backward pass computes future marginal tax rates per bucket.
    Pass 2: forward pass makes optimal decisions each year:
      - Withdrawal: sources sorted by (current_cost - future_cost), cheapest first.
        This way we draw from buckets whose current tax is low AND whose future
        tax savings from keeping the money are also low.
      - Roth conversion: convert pre-tax → Roth in $1k steps as long as
        marginal_cost_now < future_pretax_rate (the DP shadow price).
    """
    annual_return = p.expected_return

    # Iterative shadow-price convergence:
    # 1. Compute shadow from no-conversion pretax trajectory (worst case)
    # 2. Run forward pass → get actual pretax trajectory after conversions
    # 3. Recompute shadow from actual trajectory
    # 4. Blend old and new shadow prices (dampen oscillation)
    # 5. Repeat until shadow converges
    shadow = _backward_pass(p, annual_return)
    for _iteration in range(10):
        results = _run_forward_pass(p, shadow)
        pretax_traj = [r.pretax for r in results]
        raw_shadow = _backward_pass(p, annual_return, pretax_trajectory=pretax_traj)
        # Blend: average old and new shadow to dampen oscillation
        max_delta = 0.0
        for y in range(len(shadow)):
            for key in ("future_pretax_rate", "future_aftertax_rate",
                        "future_taxable_rate"):
                old_val = shadow[y][key]
                new_val = raw_shadow[y][key]
                blended = 0.5 * old_val + 0.5 * new_val
                delta = abs(blended - old_val)
                if delta > max_delta:
                    max_delta = delta
                shadow[y][key] = blended
        if max_delta < 0.003:  # converged within 0.3%
            break
    # Final forward pass with converged shadow
    return _run_forward_pass(p, shadow)


def _run_forward_pass(p: RetirementParams, shadow: list[dict]) -> list[YearResult]:
    """Single forward pass using pre-computed shadow prices."""
    annual_return = p.expected_return
    taxable = p.taxable_balance
    pretax = p.pretax_balance
    roth = p.roth_balance
    aftertax = p.aftertax_balance
    cost_basis_pct = p.taxable_cost_basis_pct
    fixed_wd = None
    spending_level = None  # set on first retirement year: pull - taxes = standard of living
    results = []

    # Roth accessibility tracking
    current_year = datetime.now().year
    roth_contribs = p.roth_contributions  # always-accessible basis
    roth_conv_list = list(p.roth_conversions_by_year)  # [(year, amount), ...]

    for y in range(p.lifespan - p.current_age + 1):
        age = p.current_age + y
        sim_year = current_year + y
        is_retired = age >= p.retirement_age
        notes = []
        sp = shadow[y]

        ss = p.social_security_annual if age >= p.social_security_start_age else 0
        # W-2 income sets the bracket floor during accumulation
        base_income = p.annual_income if not is_retired else 0
        annual_contrib = 0.0

        # ── Contributions ──────────────────────────────────────────────
        if not is_retired:
            taxable += p.contribution_taxable
            pretax += p.contribution_pretax
            roth += p.contribution_roth
            roth_contribs += p.contribution_roth  # track Roth contributions
            aftertax += p.contribution_aftertax
            annual_contrib = (p.contribution_taxable + p.contribution_pretax +
                              p.contribution_roth + p.contribution_aftertax)
            if annual_contrib > 0:
                notes.append(f"Contributed ${annual_contrib:,.0f}")

        # Compute accessible Roth for this year
        roth_accessible = _accessible_roth(roth, roth_contribs, roth_conv_list, sim_year, age)

        # ── Withdrawal need ────────────────────────────────────────────
        # Year 1: pull = 3.5% × portfolio.  spending = pull - taxes.
        #         This "spending" is the standard of living, flat forever.
        # Year 2+: SS offsets spending need from portfolio.
        #          Gross-up: find pull s.t. pull - taxes = max(spending - SS, 0).
        total = taxable + pretax + roth + aftertax
        gross_wd = 0.0
        if is_retired:
            if fixed_wd is None:
                fixed_wd = p.fixed_withdrawal_for(total)
            gross_wd = fixed_wd
            if total > 0:
                if spending_level is None:
                    # Year 1: pull the full 3.5%, taxes come out of it
                    gross_need = fixed_wd
                else:
                    # Subsequent years: portfolio provides spending - SS
                    gross_need = max(spending_level - ss, 0)
            else:
                gross_need = 0
                target_spend = spending_level if spending_level else fixed_wd
                if target_spend > ss:
                    notes.append(f"DEPLETED — need ${target_spend - ss:,.0f} but portfolio is $0")
        else:
            gross_need = 0

        withdrawn = 0.0
        taxes = 0.0
        conv_tax_paid = 0.0
        roth_conversion = 0.0
        ordinary_income = 0.0
        wd_breakdown = {"taxable": 0.0, "pretax": 0.0, "roth": 0.0, "aftertax": 0.0}

        # ── Unified greedy optimizer: withdrawal + conversion ─────────
        # At each $1k increment, pick the cheapest action among:
        #   spend_taxable  — LTCG on gains, no ordinary income
        #   spend_roth     — 0 tax (contributions/matured conversions/age≥60)
        #   spend_aftertax — half is ordinary income
        #   spend_pretax   — fully ordinary income (also frees bracket
        #                    space that would have gone to conversion)
        #   convert_pretax — ordinary income now, saves future_pretax_rate
        # Spending actions reduce the remaining spending need.
        # Conversion has no spending value but reduces lifetime taxes
        # when marginal_now < future_pretax_rate.
        #
        # After the greedy pass we know: total withdrawn, total converted,
        # total tax.  We then gross-up (pull extra to cover the tax bill)
        # and re-run with the adjusted target.

        base_ord = base_income + ss
        future_pretax_rate = sp["future_pretax_rate"]

        def _run_greedy(spending_target):
            """Greedy incremental optimizer.  Returns (wd_by_source, convert,
            taxes, conv_tax, ordinary_income) for the given spending target."""
            step = 1000.0
            bal = {"taxable": taxable, "pretax": pretax,
                   "roth": roth, "aftertax": aftertax}
            spend_left = spending_target
            wd = {"taxable": 0.0, "pretax": 0.0, "roth": 0.0, "aftertax": 0.0}
            conv = 0.0
            g_tax = 0.0
            g_conv_tax = 0.0  # tax attributable to conversions only
            g_ord = 0.0  # ordinary income accumulated
            g_ltcg = 0.0  # cumulative LTCG realized this year
            roth_used = 0.0  # how much accessible Roth used so far

            # Tax budget for conversion: taxable/aftertax/accessible-roth can
            # pay directly. Pretax can self-fund by converting extra.
            def _conv_tax_budget():
                accessible_roth_remaining = max(roth_accessible - roth_used, 0)
                liquid = bal["taxable"] + bal["aftertax"] + min(accessible_roth_remaining, bal["roth"])
                pretax_headroom = bal["pretax"] * 0.3
                return liquid + pretax_headroom

            max_iters = int((sum(bal.values()) + pretax) / step) + 10
            for _ in range(max_iters):
                # Build candidates for the next $1k
                cands = []  # (action, source, marginal_cost)

                # Spending candidates (only if still need to spend)
                if spend_left > 0:
                    chunk = min(step, spend_left)
                    if bal["taxable"] >= chunk:
                        gain = chunk * (1 - cost_basis_pct)
                        # Stack LTCG: cost of adding this gain on top of existing LTCG
                        cost = (compute_total_tax(base_ord + g_ord, g_ltcg + gain) -
                                compute_total_tax(base_ord + g_ord, g_ltcg)) / chunk
                        cands.append(("spend", "taxable", cost))
                    # Roth: accessible portion (contributions + matured conversions)
                    accessible_remaining = max(roth_accessible - roth_used, 0)
                    if bal["roth"] >= chunk and accessible_remaining >= chunk:
                        cands.append(("spend", "roth", 0.0))
                    if bal["aftertax"] >= chunk:
                        gain = chunk * 0.5
                        cost = (compute_total_tax(base_ord + g_ord + gain, g_ltcg) -
                                compute_total_tax(base_ord + g_ord, g_ltcg)) / chunk
                        cands.append(("spend", "aftertax", cost))
                    if bal["pretax"] >= chunk:
                        cost = (compute_total_tax(base_ord + g_ord + chunk, g_ltcg) -
                                compute_total_tax(base_ord + g_ord, g_ltcg)) / chunk
                        cands.append(("spend", "pretax", cost))
                    # Roth earnings (inaccessible portion): last resort, same as tax-free
                    # but only if age >= 60 and all accessible used up
                    if bal["roth"] >= chunk and accessible_remaining < chunk and age >= 60:
                        cands.append(("spend", "roth", 0.0))

                # Conversion candidate: net cost = marginal_now - future savings
                if bal["pretax"] >= step and _conv_tax_budget() > 0:
                    marginal_now = (compute_total_tax(base_ord + g_ord + step, g_ltcg) -
                                    compute_total_tax(base_ord + g_ord, g_ltcg)) / step
                    net_conv_cost = marginal_now - future_pretax_rate
                    if net_conv_cost < 0:
                        inc_tax = (compute_total_tax(base_ord + g_ord + step, g_ltcg) -
                                   compute_total_tax(base_ord + g_ord, g_ltcg))
                        if inc_tax <= _conv_tax_budget():
                            cands.append(("convert", "pretax", net_conv_cost))

                if not cands:
                    break

                # Pick cheapest action
                cands.sort(key=lambda x: x[2])
                action, src, cost = cands[0]

                if action == "convert" and cost >= 0:
                    break

                if action == "spend":
                    chunk = min(step, spend_left)
                    bal[src] -= chunk
                    wd[src] += chunk
                    spend_left -= chunk
                    if src == "pretax":
                        inc_tax = (compute_total_tax(base_ord + g_ord + chunk, g_ltcg) -
                                   compute_total_tax(base_ord + g_ord, g_ltcg))
                        g_tax += inc_tax
                        g_ord += chunk
                    elif src == "aftertax":
                        gain = chunk * 0.5
                        inc_tax = (compute_total_tax(base_ord + g_ord + gain, g_ltcg) -
                                   compute_total_tax(base_ord + g_ord, g_ltcg))
                        g_tax += inc_tax
                        g_ord += gain
                    elif src == "taxable":
                        gain = chunk * (1 - cost_basis_pct)
                        inc_tax = (compute_total_tax(base_ord + g_ord, g_ltcg + gain) -
                                   compute_total_tax(base_ord + g_ord, g_ltcg))
                        g_tax += inc_tax
                        g_ltcg += gain
                    elif src == "roth":
                        roth_used += chunk
                elif action == "convert":
                    inc_tax = (compute_total_tax(base_ord + g_ord + step, g_ltcg) -
                               compute_total_tax(base_ord + g_ord, g_ltcg))
                    bal["pretax"] -= step
                    bal["roth"] += step
                    conv += step
                    g_tax += inc_tax
                    g_conv_tax += inc_tax
                    g_ord += step
                    # Pay conversion tax from portfolio
                    remaining_tax = inc_tax
                    for pay_src in ("taxable", "aftertax", "roth"):
                        if remaining_tax <= 0:
                            break
                        pay = min(remaining_tax, bal[pay_src])
                        bal[pay_src] -= pay
                        remaining_tax -= pay
                    # If still unpaid, self-fund from pretax (convert extra)
                    if remaining_tax > 0 and bal["pretax"] > 0:
                        extra = min(remaining_tax, bal["pretax"])
                        bal["pretax"] -= extra
                        bal["roth"] += extra
                        conv += extra
                        remaining_tax -= extra

                # If spending done and no beneficial conversion, stop
                if spend_left <= 0 and action != "convert":
                    if bal["pretax"] < step or _conv_tax_budget() <= 0:
                        break
                    next_marginal = (compute_total_tax(base_ord + g_ord + step, g_ltcg) -
                                     compute_total_tax(base_ord + g_ord, g_ltcg)) / step
                    if next_marginal >= future_pretax_rate:
                        break

            return wd, conv, g_tax, g_conv_tax, g_ord, bal

        # Run greedy optimizer
        if gross_need > 0:
            total_avail = taxable + pretax + roth + aftertax
            inaccessible_roth = roth - roth_accessible
            if inaccessible_roth > 1:
                notes.append(f"Roth ${inaccessible_roth:,.0f} inaccessible (earnings/immature conversions)")

            if spending_level is None:
                # Year 1: pull fixed_wd, taxes come out of it
                target = min(gross_need, total_avail)
                wd_plan, conv_plan, tax_plan, ctax_plan, ord_plan, bal_plan = _run_greedy(target)
                spend_tax_yr1 = tax_plan - ctax_plan
                spending_level = target - spend_tax_yr1  # lock standard of living
                notes.append(f"Set spending ${spending_level:,.0f}/yr (WD ${target:,.0f} - tax ${spend_tax_yr1:,.0f})")
            else:
                # Subsequent years: gross-up to net spending_level - SS
                net_need = max(spending_level - ss, 0)
                target = net_need
                for _ in range(5):
                    wd_plan, conv_plan, tax_plan, ctax_plan, ord_plan, bal_plan = _run_greedy(target)
                    spend_tax_est = tax_plan - ctax_plan
                    new_target = min(net_need + spend_tax_est, total_avail)
                    if abs(new_target - target) < 1:
                        break
                    target = new_target
                notes.append(f"Spend ${spending_level:,.0f} (SS ${ss:,.0f}, pull ${target:,.0f})")

            # Apply the final plan — balances from greedy are local copies,
            # so we apply withdrawals and conversions to the outer variables.
            # Conversion tax was already paid inside _run_greedy (from bal),
            # so we use bal_plan as the authoritative post-greedy balances.
            taxable = bal_plan["taxable"]
            pretax = bal_plan["pretax"]
            roth = bal_plan["roth"]
            aftertax = bal_plan["aftertax"]

            withdrawn = sum(wd_plan.values())
            wd_breakdown = wd_plan
            taxes = tax_plan
            conv_tax_paid = ctax_plan
            ordinary_income = ord_plan
            roth_conversion = conv_plan

            if conv_plan > 0:
                roth_conv_list.append((sim_year, conv_plan))
                eff_rate = ctax_plan / conv_plan if conv_plan > 0 else 0
                notes.append(
                    f"Roth convert ${conv_plan:,.0f} "
                    f"(eff {eff_rate:.1%} vs future {future_pretax_rate:.1%})")

            # Log withdrawal breakdown
            for src in ("taxable", "roth", "aftertax", "pretax"):
                if wd_plan[src] > 0:
                    if src == "roth":
                        notes.append(f"WD ${wd_plan[src]:,.0f} Roth (tax-free)")
                    else:
                        notes.append(f"WD ${wd_plan[src]:,.0f} {src}")

            spend_tax_actual = taxes - conv_tax_paid
            if withdrawn < target:
                shortfall = target - withdrawn
                notes.append(f"SHORTFALL ${shortfall:,.0f}")
            spending_net = withdrawn - spend_tax_actual
            notes.append(f"Pull ${withdrawn:,.0f} → tax ${spend_tax_actual:,.0f} → net spend ${spending_net:,.0f} | conv ${conv_plan:,.0f} conv-tax ${conv_tax_paid:,.0f}")

        else:
            # No spending need — still do Roth conversions if beneficial
            wd_plan, conv_plan, tax_plan, ctax_plan, ord_plan, bal_plan = _run_greedy(0)
            if conv_plan > 0:
                # Apply greedy result balances directly
                taxable = bal_plan["taxable"]
                pretax = bal_plan["pretax"]
                roth = bal_plan["roth"]
                aftertax = bal_plan["aftertax"]
                conv_tax_paid = ctax_plan
                taxes = tax_plan
                ordinary_income = ord_plan
                roth_conversion = conv_plan
                roth_conv_list.append((sim_year, conv_plan))
                eff_rate = ctax_plan / conv_plan if conv_plan > 0 else 0
                notes.append(
                    f"Roth convert ${conv_plan:,.0f} "
                    f"(eff {eff_rate:.1%} vs future {future_pretax_rate:.1%})")
            elif pretax > 0:
                marginal_now = _marginal_rate_at(base_ord, 1000)
                notes.append(
                    f"No Roth conversion "
                    f"(marginal {marginal_now:.1%} >= future pretax {future_pretax_rate:.1%})")

        # ── Growth ─────────────────────────────────────────────────────
        taxable = max(taxable * (1 + annual_return), 0)
        pretax = max(pretax * (1 + annual_return), 0)
        roth = max(roth * (1 + annual_return), 0)
        aftertax = max(aftertax * (1 + annual_return), 0)
        total = taxable + pretax + roth + aftertax

        spend_tax = taxes - conv_tax_paid
        spending_out = spending_level if spending_level else 0.0  # flat standard of living

        results.append(YearResult(
            age=age, year=y,
            taxable=round(taxable, 2), pretax=round(pretax, 2),
            roth=round(roth, 2), aftertax=round(aftertax, 2),
            total=round(total, 2),
            gross_withdrawal=round(gross_wd, 2),
            spending=round(spending_out, 2),
            withdrawal=round(withdrawn, 2),
            wd_taxable=round(wd_breakdown["taxable"], 2),
            wd_pretax=round(wd_breakdown["pretax"], 2),
            wd_roth=round(wd_breakdown["roth"], 2),
            wd_aftertax=round(wd_breakdown["aftertax"], 2),
            roth_conversion=round(roth_conversion, 2),
            taxes_paid=round(taxes, 2),
            spending_taxes=round(spend_tax, 2),
            conversion_taxes=round(conv_tax_paid, 2),
            ss_income=round(ss, 2),
            net_income=round(spending_out, 2),
            contributions=round(annual_contrib, 2),
            notes=" | ".join(notes) if notes else ("accumulation" if not is_retired else ""),
        ))

    return results


# ── Monte Carlo helpers ───────────────────────────────────────────────────────

def _mc_percentiles(all_totals: np.ndarray, ages: list, depleted_ages: list, n_sims: int) -> dict:
    """Compute percentiles and success rate from MC results."""
    p10 = np.percentile(all_totals, 10, axis=0)
    p25 = np.percentile(all_totals, 25, axis=0)
    p50 = np.percentile(all_totals, 50, axis=0)
    p75 = np.percentile(all_totals, 75, axis=0)
    p90 = np.percentile(all_totals, 90, axis=0)
    success_count = sum(1 for d in depleted_ages if d is None)
    return {
        "ages": ages,
        "p10": p10.tolist(),
        "p25": p25.tolist(),
        "p50": p50.tolist(),
        "p75": p75.tolist(),
        "p90": p90.tolist(),
        "success_rate": success_count / n_sims,
        "n_sims": n_sims,
        "depleted_ages": depleted_ages,
    }


# ── Simulation 3: Simple Monte Carlo ─────────────────────────────────────────

def run_simple_monte_carlo(
    p: RetirementParams,
    n_sims: int = 500,
    volatility: float = 0.15,
) -> dict:
    """Monte Carlo of the simple projection with annual steps."""
    base_return = p.expected_return
    years = p.lifespan - p.current_age + 1
    annual_contrib = (p.contribution_taxable + p.contribution_pretax +
                      p.contribution_roth + p.contribution_aftertax)

    rng = np.random.default_rng(42)
    all_totals = np.zeros((n_sims, years))
    depleted_ages = []

    total_start = p.taxable_balance + p.pretax_balance + p.roth_balance + p.aftertax_balance

    for sim in range(n_sims):
        annual_returns = rng.normal(base_return, volatility, years)
        total = total_start
        depleted_age = None
        fixed_wd = None

        for y in range(years):
            age = p.current_age + y
            is_retired = age >= p.retirement_age
            ret = annual_returns[y]

            ss = p.social_security_annual if age >= p.social_security_start_age else 0
            contrib = annual_contrib if not is_retired else 0

            if is_retired:
                if fixed_wd is None:
                    fixed_wd = p.fixed_withdrawal_for(total)
                withdrawal = max(fixed_wd - ss, 0)
            else:
                withdrawal = 0

            total = total * (1 + ret) + contrib - withdrawal
            total = max(total, 0)
            all_totals[sim, y] = total

            if total < 100 and is_retired and depleted_age is None:
                depleted_age = age

        depleted_ages.append(depleted_age)

    ages = list(range(p.current_age, p.lifespan + 1))
    return _mc_percentiles(all_totals, ages, depleted_ages, n_sims)


# ── Simulation 4: Tax-Optimized Monte Carlo ──────────────────────────────────

def _run_tax_optimized_single_year(
    age: int, is_retired: bool,
    taxable: float, pretax: float, roth: float, aftertax: float,
    cost_basis_pct: float, ss: float,
    gross_need: float, annual_return: float,
    c_taxable: float = 0, c_pretax: float = 0,
    c_roth: float = 0, c_aftertax: float = 0,
    shadow_prices: dict | None = None,
    annual_income: float = 0,
    roth_accessible: float = 0,
) -> tuple[float, float, float, float, float, float, float]:
    """Run one year of the DP tax-optimized strategy (MC version).

    Uses shadow_prices (from backward pass) if provided, otherwise falls
    back to a reasonable heuristic.  Handles depletion: if balances can't
    meet gross_need, withdraws everything available.
    """
    withdrawn = 0.0
    taxes = 0.0
    roth_conversion = 0.0
    # W-2 income sets bracket floor during accumulation
    base_income = annual_income if not is_retired else 0

    # Contributions (pre-retirement)
    if not is_retired:
        taxable += c_taxable
        pretax += c_pretax
        roth += c_roth
        aftertax += c_aftertax

    # ── Unified greedy optimizer (mirrors forward pass) ────────────────
    base_ord = base_income + ss
    sp = shadow_prices or {
        "future_pretax_rate": 0.15, "future_taxable_rate": 0.05,
        "future_aftertax_rate": 0.10, "future_roth_rate": 0.0,
    }
    future_pretax_rate = sp["future_pretax_rate"]
    ordinary_income = 0.0

    def _mc_greedy(spending_target):
        stp = 1000.0
        bal = {"taxable": taxable, "pretax": pretax,
               "roth": roth, "aftertax": aftertax}
        spend_left = spending_target
        wd = {"taxable": 0.0, "pretax": 0.0, "roth": 0.0, "aftertax": 0.0}
        conv = 0.0
        g_tax = 0.0
        g_conv_tax = 0.0
        g_ord = 0.0
        g_ltcg = 0.0
        mc_roth_used = 0.0

        def _ctb():
            acc_rem = max(roth_accessible - mc_roth_used, 0)
            liquid = bal["taxable"] + bal["aftertax"] + min(acc_rem, bal["roth"])
            return liquid + bal["pretax"] * 0.3

        max_iters = int((sum(bal.values()) + pretax) / stp) + 10
        for _ in range(max_iters):
            cands = []
            if spend_left > 0:
                chunk = min(stp, spend_left)
                if bal["taxable"] >= chunk:
                    gain = chunk * (1 - cost_basis_pct)
                    c = (compute_total_tax(base_ord + g_ord, g_ltcg + gain) -
                         compute_total_tax(base_ord + g_ord, g_ltcg)) / chunk
                    cands.append(("spend", "taxable", c))
                acc_rem = max(roth_accessible - mc_roth_used, 0)
                if bal["roth"] >= chunk and acc_rem >= chunk:
                    cands.append(("spend", "roth", 0.0))
                if bal["aftertax"] >= chunk:
                    gain = chunk * 0.5
                    c = (compute_total_tax(base_ord + g_ord + gain, g_ltcg) -
                         compute_total_tax(base_ord + g_ord, g_ltcg)) / chunk
                    cands.append(("spend", "aftertax", c))
                if bal["pretax"] >= chunk:
                    c = (compute_total_tax(base_ord + g_ord + chunk, g_ltcg) -
                         compute_total_tax(base_ord + g_ord, g_ltcg)) / chunk
                    cands.append(("spend", "pretax", c))
                if bal["roth"] >= chunk and acc_rem < chunk and age >= 60:
                    cands.append(("spend", "roth", 0.0))

            if bal["pretax"] >= stp and _ctb() > 0:
                mn = (compute_total_tax(base_ord + g_ord + stp, g_ltcg) -
                      compute_total_tax(base_ord + g_ord, g_ltcg)) / stp
                nc = mn - future_pretax_rate
                if nc < 0:
                    itx = (compute_total_tax(base_ord + g_ord + stp, g_ltcg) -
                           compute_total_tax(base_ord + g_ord, g_ltcg))
                    if itx <= _ctb():
                        cands.append(("convert", "pretax", nc))

            if not cands:
                break
            cands.sort(key=lambda x: x[2])
            action, src, cost = cands[0]
            if action == "convert" and cost >= 0:
                break

            if action == "spend":
                chunk = min(stp, spend_left)
                bal[src] -= chunk
                wd[src] += chunk
                spend_left -= chunk
                if src == "pretax":
                    itx = (compute_total_tax(base_ord + g_ord + chunk, g_ltcg) -
                           compute_total_tax(base_ord + g_ord, g_ltcg))
                    g_tax += itx; g_ord += chunk
                elif src == "aftertax":
                    gain = chunk * 0.5
                    itx = (compute_total_tax(base_ord + g_ord + gain, g_ltcg) -
                           compute_total_tax(base_ord + g_ord, g_ltcg))
                    g_tax += itx; g_ord += gain
                elif src == "taxable":
                    gain = chunk * (1 - cost_basis_pct)
                    itx = (compute_total_tax(base_ord + g_ord, g_ltcg + gain) -
                           compute_total_tax(base_ord + g_ord, g_ltcg))
                    g_tax += itx; g_ltcg += gain
                elif src == "roth":
                    mc_roth_used += chunk
            elif action == "convert":
                itx = (compute_total_tax(base_ord + g_ord + stp, g_ltcg) -
                       compute_total_tax(base_ord + g_ord, g_ltcg))
                bal["pretax"] -= stp
                bal["roth"] += stp
                conv += stp; g_tax += itx; g_conv_tax += itx; g_ord += stp
                rem_tax = itx
                for ps in ("taxable", "aftertax", "roth"):
                    if rem_tax <= 0:
                        break
                    pay = min(rem_tax, bal[ps])
                    bal[ps] -= pay; rem_tax -= pay
                if rem_tax > 0 and bal["pretax"] > 0:
                    extra = min(rem_tax, bal["pretax"])
                    bal["pretax"] -= extra
                    bal["roth"] += extra
                    conv += extra
                    rem_tax -= extra

            if spend_left <= 0 and action != "convert":
                if bal["pretax"] < stp or _ctb() <= 0:
                    break
                nm = (compute_total_tax(base_ord + g_ord + stp, g_ltcg) -
                      compute_total_tax(base_ord + g_ord, g_ltcg)) / stp
                if nm >= future_pretax_rate:
                    break

        return wd, conv, g_tax, g_conv_tax, g_ord, bal

    if gross_need > 0:
        total_avail = taxable + pretax + roth + aftertax
        spending_need = min(gross_need, total_avail)
        target = spending_need
        for _ in range(5):
            wd_plan, conv_plan, tax_plan, ctax_plan, ord_plan, bal_plan = _mc_greedy(target)
            new_target = min(spending_need + tax_plan, total_avail)
            if abs(new_target - target) < 1:
                break
            target = new_target

        # Use bal_plan directly — conversion tax already paid inside greedy
        taxable = bal_plan["taxable"]
        pretax = bal_plan["pretax"]
        roth = bal_plan["roth"]
        aftertax = bal_plan["aftertax"]
        withdrawn = sum(wd_plan.values())
        taxes = tax_plan
        roth_conversion = conv_plan
    else:
        wd_plan, conv_plan, tax_plan, ctax_plan, ord_plan, bal_plan = _mc_greedy(0)
        if conv_plan > 0:
            taxable = bal_plan["taxable"]
            pretax = bal_plan["pretax"]
            roth = bal_plan["roth"]
            aftertax = bal_plan["aftertax"]
            taxes = tax_plan
            roth_conversion = conv_plan

    # ── Growth ─────────────────────────────────────────────────────────
    taxable = max(taxable * (1 + annual_return), 0)
    pretax = max(pretax * (1 + annual_return), 0)
    roth = max(roth * (1 + annual_return), 0)
    aftertax = max(aftertax * (1 + annual_return), 0)

    return taxable, pretax, roth, aftertax, withdrawn, taxes, roth_conversion


def run_tax_optimized_monte_carlo(
    p: RetirementParams,
    n_sims: int = 500,
    volatility: float = 0.15,
) -> dict:
    """Monte Carlo with DP tax-optimized strategy.

    Computes backward-pass shadow prices once (using expected return),
    then runs each simulation forward using those shadow prices for decisions.
    Handles depletion: if portfolio hits zero, records shortfall.
    """
    base_return = p.expected_return
    years = p.lifespan - p.current_age + 1

    # Compute shadow prices once using expected return
    shadow = _backward_pass(p, base_return)

    rng = np.random.default_rng(42)
    all_totals = np.zeros((n_sims, years))
    all_taxes = np.zeros((n_sims, years))
    all_shortfalls = np.zeros((n_sims, years))
    depleted_ages = []

    mc_current_year = datetime.now().year

    for sim in range(n_sims):
        annual_returns = rng.normal(base_return, volatility, years)

        taxable = p.taxable_balance
        pretax = p.pretax_balance
        roth = p.roth_balance
        aftertax = p.aftertax_balance
        depleted_age = None
        fixed_wd = None
        mc_roth_contribs = p.roth_contributions
        mc_roth_conv_list = list(p.roth_conversions_by_year)

        for y in range(years):
            age = p.current_age + y
            sim_year = mc_current_year + y
            is_retired = age >= p.retirement_age
            ret = annual_returns[y]

            ss = p.social_security_annual if age >= p.social_security_start_age else 0
            total = taxable + pretax + roth + aftertax

            if is_retired:
                if fixed_wd is None:
                    fixed_wd = p.fixed_withdrawal_for(total)
                gross_need = max(fixed_wd - ss, 0) if total > 0 else 0
            else:
                gross_need = 0

            c_t = p.contribution_taxable if not is_retired else 0
            c_p = p.contribution_pretax if not is_retired else 0
            c_r = p.contribution_roth if not is_retired else 0
            c_a = p.contribution_aftertax if not is_retired else 0
            if not is_retired:
                mc_roth_contribs += c_r

            mc_roth_acc = _accessible_roth(roth + c_r, mc_roth_contribs, mc_roth_conv_list, sim_year, age)

            taxable, pretax, roth, aftertax, withdrawn, taxes, rc = _run_tax_optimized_single_year(
                age, is_retired, taxable, pretax, roth, aftertax,
                p.taxable_cost_basis_pct, ss, gross_need, ret,
                c_t, c_p, c_r, c_a,
                shadow[y],
                annual_income=p.annual_income,
                roth_accessible=mc_roth_acc,
            )

            if rc > 0:
                mc_roth_conv_list.append((sim_year, rc))

            total = taxable + pretax + roth + aftertax
            all_totals[sim, y] = total
            all_taxes[sim, y] = taxes

            # Track shortfall
            shortfall = max(gross_need - withdrawn, 0)
            all_shortfalls[sim, y] = shortfall

            if is_retired and depleted_age is None:
                if total < 100 or shortfall > gross_need * 0.5:
                    depleted_age = age

        depleted_ages.append(depleted_age)

    ages = list(range(p.current_age, p.lifespan + 1))
    result = _mc_percentiles(all_totals, ages, depleted_ages, n_sims)
    result["taxes_p50"] = np.percentile(all_taxes, 50, axis=0).tolist()
    result["taxes_p25"] = np.percentile(all_taxes, 25, axis=0).tolist()
    result["taxes_p75"] = np.percentile(all_taxes, 75, axis=0).tolist()
    result["total_taxes_per_sim"] = all_taxes.sum(axis=1).tolist()
    result["total_shortfall_per_sim"] = all_shortfalls.sum(axis=1).tolist()
    result["shortfall_years_per_sim"] = (all_shortfalls > 0).sum(axis=1).tolist()
    return result
