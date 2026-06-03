"""Sanity checks for data/retirement_sim.py.

Run:  python -m pytest tests/test_retirement_sim.py -v
"""
import pytest
from data.retirement_sim import (
    compute_total_tax, compute_federal_tax, compute_ca_tax,
    RetirementParams, run_simple_projection, run_tax_optimized,
    run_simple_monte_carlo, run_tax_optimized_monte_carlo,
)


# ── Tax math ─────────────────────────────────────────────────────────────────

class TestTaxMath:
    def test_zero_income(self):
        assert compute_total_tax(0) == 0
        assert compute_total_tax(0, 0) == 0

    def test_federal_standard_deduction(self):
        assert compute_federal_tax(16100) == 0
        assert compute_federal_tax(16099) == 0

    def test_ca_standard_deduction(self):
        assert compute_ca_tax(5706) == 0
        assert compute_ca_tax(5705) == 0

    def test_50k_ordinary_manual_bracket(self):
        """Manually compute 50k ordinary income through brackets."""
        fed_agi = 50000 - 16100  # 33900
        fed = 12400 * 0.10 + (33900 - 12400) * 0.12
        ca_agi = 50000 - 5706    # 44294
        ca = (11079 * 0.01
              + (26264 - 11079) * 0.02
              + (41452 - 26264) * 0.04
              + (44294 - 41452) * 0.06)
        assert abs(compute_total_tax(50000) - (fed + ca)) < 0.01

    def test_ltcg_zero_bracket(self):
        """40k LTCG with 0 ordinary → fed LTCG at 0%, only CA taxes."""
        t = compute_total_tax(0, 40000)
        ca_only = compute_ca_tax(40000)
        assert abs(t - ca_only) < 0.01

    def test_ltcg_stacking(self):
        """50k ordinary + 10k LTCG: LTCG still in 0% fed bracket."""
        t = compute_total_tax(50000, 10000)
        fed_ord = compute_federal_tax(50000)
        ca_60 = compute_ca_tax(60000)
        assert abs(t - (fed_ord + ca_60)) < 0.01

    def test_ltcg_15pct_bracket(self):
        """100k LTCG with 0 ordinary: 49450 at 0%, rest at 15%."""
        t = compute_total_tax(0, 100000)
        fed_cg = (100000 - 49450) * 0.15
        ca = compute_ca_tax(100000)
        assert abs(t - (fed_cg + ca)) < 0.01

    def test_monotonic(self):
        """Tax should be monotonically increasing with income."""
        prev = 0
        for income in range(0, 200001, 10000):
            t = compute_total_tax(income)
            assert t >= prev, f"Tax decreased at {income}"
            prev = t

    def test_marginal_rate_below_100pct(self):
        """Marginal rate should never exceed 100%."""
        for income in range(0, 500001, 25000):
            t1 = compute_total_tax(income)
            t2 = compute_total_tax(income + 1000)
            marginal = (t2 - t1) / 1000
            assert marginal < 1.0, f"Marginal rate {marginal} at {income}"


# ── Simple projection ────────────────────────────────────────────────────────

class TestSimpleProjection:
    def test_zero_return_withdrawal(self):
        """100k, 0% return, 4% rate → fixed $4k/yr withdrawal (rate × balance at retirement)."""
        p = RetirementParams(
            current_age=65, retirement_age=65, lifespan=67,
            expected_return=0.0, withdrawal_rate=0.04, taxable_balance=100000,
        )
        r = run_simple_projection(p)
        assert r[0].withdrawal == 4000.0
        assert r[0].total == 96000.0
        # yr1: still $4k (fixed), not 96k * 0.04
        assert r[1].withdrawal == 4000.0
        assert r[1].total == 92000.0

    def test_contributions_accumulate(self):
        """10k/yr contrib, 0% return, retire at 32. Rate applied to balance at retirement."""
        p = RetirementParams(
            current_age=30, retirement_age=32, lifespan=33,
            expected_return=0.0, withdrawal_rate=0.04, contribution_pretax=10000,
        )
        r = run_simple_projection(p)
        assert r[0].total == 10000.0  # age 30
        assert r[1].total == 20000.0  # age 31
        # age 32: balance at retirement = 20000, 4% = 800
        assert r[2].withdrawal == 800.0

    def test_growth(self):
        """10% return, no withdrawal."""
        p = RetirementParams(
            current_age=30, retirement_age=90, lifespan=31,
            expected_return=0.10, taxable_balance=100000,
        )
        r = run_simple_projection(p)
        assert abs(r[0].total - 110000) < 0.01  # 100k * 1.10
        assert abs(r[1].total - 121000) < 0.01  # 110k * 1.10

    def test_ss_reduces_withdrawal(self):
        """SS should reduce portfolio withdrawal."""
        p = RetirementParams(
            current_age=67, retirement_age=67, lifespan=68,
            expected_return=0.0, withdrawal_amount=50000,
            social_security_annual=20000, social_security_start_age=67,
            taxable_balance=500000,
        )
        r = run_simple_projection(p)
        assert r[0].withdrawal == 30000.0  # 50k - 20k SS


# ── Tax-optimized ────────────────────────────────────────────────────────────

class TestTaxOptimized:
    def test_roth_only_zero_taxes(self):
        """Pure Roth portfolio should never pay taxes."""
        p = RetirementParams(
            current_age=65, retirement_age=65, lifespan=70,
            expected_return=0.0, withdrawal_rate=0.04, roth_balance=500000,
        )
        r = run_tax_optimized(p)
        total_tax = sum(row.taxes_paid for row in r)
        assert total_tax == 0

    def test_pretax_withdrawals_incur_tax(self):
        """Pre-tax withdrawals must generate taxes."""
        p = RetirementParams(
            current_age=65, retirement_age=65, lifespan=67,
            expected_return=0.0, withdrawal_rate=0.04, pretax_balance=500000,
        )
        r = run_tax_optimized(p)
        for row in r:
            if row.withdrawal > 0:
                assert row.taxes_paid > 0, f"Age {row.age}: pretax wd but no tax"

    def test_roth_conversion_happens(self):
        """Pre-retirement with pretax, high future rate → should convert.

        Large pretax balance with high withdrawal rate creates high future
        marginal rates.  During accumulation (with W-2 setting bracket
        floor low), conversion in low brackets should happen."""
        p = RetirementParams(
            current_age=60, retirement_age=65, lifespan=90,
            expected_return=0.0, pretax_balance=2_000_000,
            annual_income=0,
        )
        r = run_tax_optimized(p)
        total_conv = sum(row.roth_conversion for row in r)
        assert total_conv > 0, "Should convert some pretax to Roth"

    def test_balance_conservation_zero_return(self):
        """With 0% return: initial = final + withdrawn + conversion_taxes.
        Withdrawn includes gross-up (spending + wd_tax) pulled from accounts.
        Conversion taxes are separately deducted from portfolio balances."""
        p = RetirementParams(
            current_age=60, retirement_age=65, lifespan=70,
            expected_return=0.0, pretax_balance=500000,
            withdrawal_rate=0.04,
        )
        r = run_tax_optimized(p)
        total_withdrawn = sum(row.withdrawal for row in r)
        total_conv_tax = sum(row.conversion_taxes for row in r)
        final = r[-1].total
        assert abs(final + total_withdrawn + total_conv_tax - 500000) < 10, (
            f"Conservation: {final:.0f} + {total_withdrawn:.0f} + {total_conv_tax:.0f} "
            f"= {final + total_withdrawn + total_conv_tax:.0f} != 500000"
        )

    def test_depleted_note(self):
        """Depleted portfolio should note shortfall."""
        p = RetirementParams(
            current_age=65, retirement_age=65, lifespan=90,
            expected_return=0.0, withdrawal_amount=100000, pretax_balance=50000,
        )
        r = run_tax_optimized(p)
        depleted_notes = [row.notes for row in r if "DEPLETED" in row.notes or "SHORTFALL" in row.notes]
        assert len(depleted_notes) > 0, "Should note depletion"

    def test_no_withdrawal_before_retirement(self):
        """No withdrawals should happen before retirement age."""
        p = RetirementParams(
            current_age=30, retirement_age=65, lifespan=70,
            expected_return=0.07, pretax_balance=100000,
            contribution_pretax=20000, withdrawal_rate=0.04,
        )
        r = run_tax_optimized(p)
        for row in r:
            if row.age < 65:
                assert row.withdrawal == 0, f"Age {row.age}: unexpected withdrawal"


# ── Monte Carlo ──────────────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_zero_vol_matches_deterministic(self):
        """MC with 0 volatility should match simple projection."""
        p = RetirementParams(
            current_age=65, retirement_age=65, lifespan=67,
            expected_return=0.0, withdrawal_rate=0.04, taxable_balance=100000,
        )
        mc = run_simple_monte_carlo(p, n_sims=5, volatility=0.0)
        assert abs(mc["p50"][0] - 96000) < 1

    def test_shortfall_tracking(self):
        """Impossible scenario should produce shortfalls in every sim."""
        p = RetirementParams(
            current_age=65, retirement_age=65, lifespan=90,
            expected_return=-0.05, withdrawal_amount=50000, pretax_balance=100000,
        )
        mc = run_tax_optimized_monte_carlo(p, n_sims=5, volatility=0.0)
        sf = mc["total_shortfall_per_sim"]
        assert all(s > 0 for s in sf), "All sims should have shortfalls"

    def test_depletion_rate(self):
        """Impossible scenario → 0% success rate."""
        p = RetirementParams(
            current_age=65, retirement_age=65, lifespan=90,
            expected_return=-0.05, withdrawal_amount=50000, pretax_balance=100000,
        )
        mc = run_tax_optimized_monte_carlo(p, n_sims=5, volatility=0.0)
        assert mc["success_rate"] == 0.0

    def test_easy_scenario_success(self):
        """Large portfolio with small withdrawal → 100% success."""
        p = RetirementParams(
            current_age=65, retirement_age=65, lifespan=90,
            expected_return=0.07, withdrawal_rate=0.02, roth_balance=2000000,
        )
        mc = run_tax_optimized_monte_carlo(p, n_sims=20, volatility=0.10)
        assert mc["success_rate"] >= 0.9


# ── Unused import check ─────────────────────────────────────────────────────

class TestCodeQuality:
    def test_no_stale_params(self):
        """RetirementParams should not have real_terms or inflation."""
        p = RetirementParams()
        assert not hasattr(p, "real_terms")
        assert not hasattr(p, "inflation")
