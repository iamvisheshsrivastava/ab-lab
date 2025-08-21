 
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
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = ((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2)
    if sp <= 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / math.sqrt(sp))

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    d = cohens_d(a, b)
    n1, n2 = len(a), len(b)
    df = n1 + n2 - 2
    if df <= 0:
        return d
    J = 1 - (3 / (4*df - 1))
    return float(J * d)

def ci_proportion_wilson(x: int | float, n: int, conf_level: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for a single proportion.
    """
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
    p1, p2 = x1/n1, x2/n2
    se = math.sqrt(p1*(1 - p1)/n1 + p2*(1 - p2)/n2)
    z = stats.norm.ppf(1 - (1 - conf_level)/2)
    diff = p2 - p1
    return float(diff - z*se), float(diff + z*se)
