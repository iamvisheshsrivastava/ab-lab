 
from __future__ import annotations
import math
import numpy as np
from typing import Tuple
from scipy import stats

def cuped(y: np.ndarray, covariate: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    CUPED adjustment: y' = y - theta * (x - mean(x)), where theta = cov(y, x) / var(x).
    """
    x = covariate
    x_centered = x - np.mean(x)
    var_x = np.var(x_centered, ddof=1)
    if var_x <= 1e-12:
        return y.copy(), 0.0
    theta = np.cov(y, x, ddof=1)[0, 1] / var_x
    y_adj = y - theta * x_centered
    return y_adj, float(theta)

def lift_absolute(p_a: float, p_b: float) -> float:
    return float(p_b - p_a)

def lift_relative(p_a: float, p_b: float) -> float:
    return float((p_b - p_a) / max(p_a, 1e-12))

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d using pooled variance. Raises ValueError for degenerate inputs
    (fewer than 2 observations per group, or zero pooled variance with a
    non-zero mean difference, which makes the effect size undefined) instead
    of silently returning NaN (see issue #12).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError(
            "cohens_d requires at least 2 observations per group to estimate variance"
        )
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    df = n1 + n2 - 2
    sp = ((n1 - 1)*s1 + (n2 - 1)*s2) / df
    if sp < 0 or not np.isfinite(sp):
        raise ValueError("Pooled variance is invalid (negative or non-finite)")
    if sp == 0:
        mean_a, mean_b = np.mean(a), np.mean(b)
        if mean_a == mean_b:
            return 0.0
        raise ValueError(
            "Pooled variance is zero but group means differ; Cohen's d is undefined (infinite)"
        )
    return float((np.mean(b) - np.mean(a)) / math.sqrt(sp))

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """
    Small-sample-corrected effect size. Delegates validation to cohens_d, so
    the same degenerate-input cases raise ValueError (see issue #12).
    """
    d = cohens_d(a, b)
    n1, n2 = len(a), len(b)
    df = n1 + n2 - 2
    J = 1 - (3 / (4*df - 1))
    return float(J * d)

def ci_proportion_wilson(x: int | float, n: int, conf_level: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for a single proportion.
    """
    if n < 0 or x < 0 or x > n:
        raise ValueError("Require n >= 0 and 0 <= x <= n")
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - (1 - conf_level)/2)
    p = x / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    halfwidth = z * math.sqrt((p*(1 - p) + z**2/(4*n)) / n) / denom
    return float(center - halfwidth), float(center + halfwidth)

def ci_diff_proportions_wald(x1: int, n1: int, x2: int, n2: int, conf_level: float = 0.95) -> Tuple[float, float]:
    """
    Wald CI for difference (p2 - p1). For quick visualizations; for production prefer
    score-based (Newcombe) methods.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Group sizes must be positive")
    if not (0 <= x1 <= n1) or not (0 <= x2 <= n2):
        raise ValueError("Successes must satisfy 0 <= x <= n")
    p1, p2 = x1/n1, x2/n2
    se = math.sqrt(p1*(1 - p1)/n1 + p2*(1 - p2)/n2)
    z = stats.norm.ppf(1 - (1 - conf_level)/2)
    diff = p2 - p1
    return float(diff - z*se), float(diff + z*se)
