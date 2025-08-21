 
from __future__ import annotations
import math
import numpy as np
from scipy import stats
from .simulate import simulate_binomial

def _z_alpha(alpha: float, two_sided: bool) -> float:
    return stats.norm.ppf(1 - alpha/2) if two_sided else stats.norm.ppf(1 - alpha)

def sample_size_proportions(
    p1: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    relative: bool = True,
) -> int:
    """
    Per-group sample size for two-proportion z-test.
    mde: relative (e.g., 0.02 => +2% over p1) if relative=True, else absolute delta.
    """
    if relative:
        p2 = p1 * (1 + mde)
    else:
        p2 = p1 + mde
    p2 = min(max(p2, 1e-12), 1 - 1e-12)
    z1 = _z_alpha(alpha, two_sided)
    z2 = stats.norm.ppf(power)
    q1, q2 = 1 - p1, 1 - p2
    se = math.sqrt(p1*q1 + p2*q2)
    n = ((z1 + z2) * se / (p2 - p1))**2
    return int(math.ceil(n))

def mde_proportions(
    p1: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    relative: bool = True,
) -> float:
    """
    Approximate MDE given baseline p1 and per-group n.
    """
    z1 = _z_alpha(alpha, two_sided)
    z2 = stats.norm.ppf(power)
    # solve for |p2 - p1| ≈ (z1 + z2) * sqrt((p1q1 + p2q2)/n)
    # iterate once by guessing p2 ~ p1
    q1 = 1 - p1
    se = math.sqrt(2 * p1*q1 / n_per_group)
    delta = (z1 + z2) * se
    if relative:
        return float(delta / max(p1, 1e-12))
    return float(delta)

def sample_size_means(
    sigma: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int:
    """
    Per-group n for t-test on means (using normal approx).
    """
    z1 = _z_alpha(alpha, two_sided)
    z2 = stats.norm.ppf(power)
    n = 2 * (sigma * (z1 + z2) / mde) ** 2
    return int(math.ceil(n))

def power_simulation_binomial(
    p_control: float,
    lift: float,
    n_per_group: int,
    *,
    nsims: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> float:
    """
    Monte-Carlo power for two-proportion z-test.
    """
    rng = np.random.default_rng(seed)
    rejs = 0
    for i in range(nsims):
        a, b = simulate_binomial(
            n_per_group, p_control, lift, relative=True, seed=rng.integers(1e9)
        )
        x1, x2 = a.sum(), b.sum()
        from .tests import ztest_proportions
        res = ztest_proportions(x1, len(a), x2, len(b))
        if res["pvalue"] < alpha:
            rejs += 1
    return rejs / nsims
