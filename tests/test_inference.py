 
import numpy as np
from ablab.simulate import simulate_binomial, simulate_continuous, make_covariate
from ablab.tests import ztest_proportions, ttest_independent, welch_ttest, mannwhitney_u_test
from ablab.metrics import cuped, ci_proportion_wilson, ci_diff_proportions_wald
from ablab.guardrails import srm_chisq, aa_sanity_check

def test_proportions_ztest_detects_lift():
    n, p0, lift = 50_000, 0.10, 0.05
    a, b = simulate_binomial(n, p_control=p0, lift=lift, seed=42)
    res = ztest_proportions(a.sum(), len(a), b.sum(), len(b))
    assert res["pvalue"] < 0.05
    assert res["estimate"][1] > res["estimate"][0]

def test_ttests_and_mannwhitney_detect_difference(): 
    a, b = simulate_continuous(4_000, mu_control=5.0, sigma=1.5, lift=0.10, seed=99)
    welch = welch_ttest(a, b)
    pooled = ttest_independent(a, b, equal_var=True)
    mw = mannwhitney_u_test(a, b)
    assert welch["pvalue"] < 0.05
    assert pooled["pvalue"] < 0.05
    assert mw["pvalue"] < 0.05

def test_cuped_reduces_variance():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=20_000)
    x = make_covariate(y, rho=0.7, seed=1)
    y_adj, theta = cuped(y, x)
    var_reduction = (np.var(y) - np.var(y_adj)) / np.var(y)
    assert theta != 0.0
    assert var_reduction > 0.25 

def test_confidence_intervals_behave():
    n, p0, lift = 30_000, 0.08, 0.04
    a, b = simulate_binomial(n, p_control=p0, lift=lift, seed=5)
    x1, x2 = a.sum(), b.sum()
    p1, p2 = x1/n, x2/n

    lo_a, hi_a = ci_proportion_wilson(x1, n)
    assert lo_a <= p1 <= hi_a

    lo_diff, hi_diff = ci_diff_proportions_wald(x1, n, x2, n)
    assert hi_diff > 0

def test_guardrails_srm_and_aa():
    srm = srm_chisq(10_000, 15_000, expected_ratio=1.0)
    assert srm["pvalue"] < 1e-6

    n = 50_000
    a = np.random.binomial(1, 0.1, size=n)
    b = np.random.binomial(1, 0.1, size=n)
    res = aa_sanity_check(a.sum(), n, b.sum(), n, alpha=0.05)
    assert res["pvalue"] >= 0.001 
