
from __future__ import annotations
import math
import numpy as np
from typing import Tuple

__all__ = [
    "mixture_sprt_proportions",
    "mixture_sprt_means",
    "always_valid_ci_proportions",
    "always_valid_ci_means",
]

def _check_counts(x1: int, n1: int, x2: int, n2: int) -> None:
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Group sizes must be positive")
    if not (0 <= x1 <= n1) or not (0 <= x2 <= n2):
        raise ValueError("Successes must satisfy 0 <= x <= n")

def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

def _mixture_e_value(stat: float, var: float, tau2: float) -> float:
    """
    Nonnegative e-value (test martingale) for a sequential test of a
    Gaussian-location null (true difference == 0) against a N(0, tau2)
    mixture alternative, given an observed statistic `stat` whose sampling
    distribution is approximately N(0, var) under the null (the usual
    large-sample normal approximation used by the fixed-horizon tests in
    `ablab/tests.py`).

    This is Robbins' mixture SPRT likelihood ratio. Because it is a
    nonnegative martingale with E[Lambda] == 1 under the null, Ville's
    inequality gives P(exists n: Lambda_n >= 1/alpha) <= alpha -- i.e. the
    resulting p-value / rejection rule stays valid under continuous
    monitoring (any number of peeks), unlike a fixed-horizon p-value.
    """
    if var <= 0:
        raise ValueError("var must be positive")
    if tau2 <= 0:
        raise ValueError("tau2 (mixture variance) must be positive")
    return math.sqrt(var / (var + tau2)) * math.exp(
        (stat ** 2) * tau2 / (2 * var * (var + tau2))
    )

def _decision(e_value: float, alpha: float) -> str:
    """
    "reject": evidence against H0 strong enough to stop and declare a winner
      (anytime-valid at level alpha, by Ville's inequality).
    "accept": e-value has fallen below alpha, i.e. the data are much more
      consistent with H0 than with the mixture alternative -- a common
      (if less formally symmetric) heuristic for "stop, no effect found".
    "continue": keep collecting data.
    """
    _check_alpha(alpha)
    if e_value >= 1 / alpha:
        return "reject"
    if e_value <= alpha:
        return "accept"
    return "continue"

def _always_valid_width(var: float, tau2: float, alpha: float) -> float:
    """
    Half-width of the confidence sequence obtained by inverting
    `_mixture_e_value` at every candidate null value delta0 (i.e. the set of
    delta0 for which the mixture-SPRT test of H0: true diff == delta0 does
    not reject). See module docstring / derivation in the sequential-testing
    issue for the algebra.
    """
    return math.sqrt(
        (2 * var * (var + tau2) / tau2)
        * math.log(math.sqrt((var + tau2) / var) / alpha)
    )

def mixture_sprt_proportions(
    x_a: int, n_a: int, x_b: int, n_b: int, *, tau: float = 0.1, alpha: float = 0.05
) -> dict:
    """
    Mixture-SPRT / always-valid test for H0: p_a == p_b, using a normal
    mixing prior (std dev `tau`) on the plausible difference p_b - p_a.

    Unlike `ablab.tests.ztest_proportions`, the returned e-value / p-value /
    decision stay statistically valid if you recompute them after every new
    observation (continuous monitoring), at the cost of some power relative
    to a fixed-horizon test planned for a single look.

    Returns
    -------
    dict with keys:
        e_value : the test martingale value (>= 1/alpha triggers "reject").
        pvalue  : an anytime-valid p-value, min(1, 1/e_value).
        stat    : observed p_b - p_a.
        variance: pooled-variance estimate of Var(stat) at this sample size.
        decision: "continue" | "reject" | "accept".
    """
    _check_counts(x_a, n_a, x_b, n_b)
    _check_alpha(alpha)
    if tau <= 0:
        raise ValueError("tau (mixture prior std dev) must be positive")
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    var = p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)
    stat = p_b - p_a
    e_value = 1.0 if var <= 0 else _mixture_e_value(stat, var, tau ** 2)
    pvalue = min(1.0, 1.0 / e_value) if e_value > 0 else 1.0
    return {
        "e_value": float(e_value),
        "pvalue": float(pvalue),
        "stat": float(stat),
        "variance": float(var),
        "decision": _decision(e_value, alpha),
    }

def mixture_sprt_means(
    a: np.ndarray, b: np.ndarray, *, tau: float | None = None, alpha: float = 0.05
) -> dict:
    """
    Mixture-SPRT / always-valid test for H0: mean(a) == mean(b).

    tau : prior std dev on the plausible mean difference. If omitted,
        defaults to the (pooled) sample standard deviation -- a
        weakly-informative, scale-matched choice.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    _check_alpha(alpha)
    if n_a < 2 or n_b < 2:
        raise ValueError("mixture_sprt_means requires at least 2 observations per group")
    if tau is not None and tau <= 0:
        raise ValueError("tau (mixture prior std dev) must be positive")
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    var = var_a / n_a + var_b / n_b
    stat = float(np.mean(b) - np.mean(a))
    if tau is None:
        pooled_sd = math.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 1.0
        tau = max(pooled_sd, 1e-6)
    e_value = 1.0 if var <= 0 else _mixture_e_value(stat, var, tau ** 2)
    pvalue = min(1.0, 1.0 / e_value) if e_value > 0 else 1.0
    return {
        "e_value": float(e_value),
        "pvalue": float(pvalue),
        "stat": stat,
        "variance": float(var),
        "decision": _decision(e_value, alpha),
    }

def always_valid_ci_proportions(
    x_a: int, n_a: int, x_b: int, n_b: int, *, tau: float = 0.1, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Continuously-monitorable (anytime-valid) confidence sequence for
    p_b - p_a, obtained by inverting `mixture_sprt_proportions` at every
    candidate null value. Valid to inspect after every new observation,
    unlike a fixed-n Wald/Wilson interval.
    """
    _check_counts(x_a, n_a, x_b, n_b)
    _check_alpha(alpha)
    if tau <= 0:
        raise ValueError("tau (mixture prior std dev) must be positive")
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    var = p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)
    stat = p_b - p_a
    if var <= 0:
        return (float(stat), float(stat))
    width = _always_valid_width(var, tau ** 2, alpha)
    return (float(stat - width), float(stat + width))

def always_valid_ci_means(
    a: np.ndarray, b: np.ndarray, *, tau: float | None = None, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Continuously-monitorable confidence sequence for mean(b) - mean(a).
    See `always_valid_ci_proportions` / `mixture_sprt_means` for details.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    _check_alpha(alpha)
    if n_a < 2 or n_b < 2:
        raise ValueError("always_valid_ci_means requires at least 2 observations per group")
    if tau is not None and tau <= 0:
        raise ValueError("tau (mixture prior std dev) must be positive")
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    var = var_a / n_a + var_b / n_b
    stat = float(np.mean(b) - np.mean(a))
    if tau is None:
        pooled_sd = math.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 1.0
        tau = max(pooled_sd, 1e-6)
    if var <= 0:
        return (stat, stat)
    width = _always_valid_width(var, tau ** 2, alpha)
    return (stat - width, stat + width)
