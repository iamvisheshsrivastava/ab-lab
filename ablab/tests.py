 
from __future__ import annotations
import math
import numpy as np
from scipy import stats

def ztest_proportions(
    x1: int, n1: int, x2: int, n2: int, two_sided: bool = True, continuity: bool = False
) -> dict:
    """
    Two-proportion z-test (H0: p1 == p2).
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        z = 0.0
    else:
        z = (p2 - p1) / se
        if continuity:
            # simple CC: subtract 0.5/n term in numerator direction
            cc = 0.5 * (1/n1 + 1/n2)
            z = (abs(p2 - p1) - cc) / se * np.sign(p2 - p1)
    if two_sided:
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p = 1 - stats.norm.cdf(z)  # tests p2 > p1
    return {"stat": z, "pvalue": p, "estimate": (p1, p2)}

def ttest_independent(a: np.ndarray, b: np.ndarray, equal_var: bool = True) -> dict:
    """
    Student's t-test with pooled variance (equal_var=True).
    """
    t, p = stats.ttest_ind(a, b, equal_var=equal_var)
    return {"stat": float(t), "pvalue": float(p), "estimate": (np.mean(a), np.mean(b))}

def welch_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Welch's t-test (unequal variances).
    """
    return ttest_independent(a, b, equal_var=False)

def mannwhitney_u_test(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> dict:
    """
    Mann–Whitney U test for continuous/ordinal data with non-normality/outliers.
    """
    u, p = stats.mannwhitneyu(a, b, alternative=alternative)
    return {"stat": float(u), "pvalue": float(p), "estimate": (np.median(a), np.median(b))}
