 
from __future__ import annotations
import numpy as np
from scipy import stats
from .tests import ztest_proportions

def srm_chisq(n_a: int, n_b: int, expected_ratio: float = 1.0) -> dict:
    """
    Sample Ratio Mismatch test via chi-square goodness-of-fit.
    expected_ratio = n_b / n_a that you planned (default 1:1).
    """
    total = n_a + n_b
    exp_a = total / (1 + expected_ratio)
    exp_b = total - exp_a
    obs = np.array([n_a, n_b], dtype=float)
    exp = np.array([exp_a, exp_b], dtype=float)
    chi2 = ((obs - exp) ** 2 / np.maximum(exp, 1e-12)).sum()
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return {"stat": float(chi2), "pvalue": float(p), "expected": (exp_a, exp_b)}

def aa_sanity_check(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """
    A/A sanity check using two-proportion test; returns whether H0 is rejected at alpha.
    """
    res = ztest_proportions(x_a, n_a, x_b, n_b)
    return {"pvalue": res["pvalue"], "reject_h0": bool(res["pvalue"] < alpha)}
