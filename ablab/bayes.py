 
from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Tuple

def beta_posteriors(
    x1: int, n1: int, x2: int, n2: int, alpha_prior: float = 1.0, beta_prior: float = 1.0
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Conjugate Beta-Binomial posteriors for p_A and p_B.
    Returns (alpha_a, beta_a), (alpha_b, beta_b).
    """
    a_a, b_a = alpha_prior + x1, beta_prior + (n1 - x1)
    a_b, b_b = alpha_prior + x2, beta_prior + (n2 - x2)
    return (float(a_a), float(b_a)), (float(a_b), float(b_b))

def posterior_samples(
    post_a: Tuple[float, float],
    post_b: Tuple[float, float],
    nsamples: int = 100_000,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pa = rng.beta(post_a[0], post_a[1], size=nsamples)
    pb = rng.beta(post_b[0], post_b[1], size=nsamples)
    return pa, pb

def prob_b_beats_a(
    post_a: Tuple[float, float],
    post_b: Tuple[float, float],
    nsamples: int = 100_000,
    seed: int | None = None,
) -> float:
    pa, pb = posterior_samples(post_a, post_b, nsamples, seed)
    return float((pb > pa).mean())

def credible_interval_beta(
    post_params: Tuple[float, float], conf_level: float = 0.95
) -> Tuple[float, float]:
    a, b = post_params
    lo = stats.beta.ppf((1 - conf_level)/2, a, b)
    hi = stats.beta.ppf(1 - (1 - conf_level)/2, a, b)
    return float(lo), float(hi)

def prob_relative_lift_gt_zero(
    post_a: Tuple[float, float],
    post_b: Tuple[float, float],
    nsamples: int = 100_000,
    seed: int | None = None,
) -> float:
    pa, pb = posterior_samples(post_a, post_b, nsamples, seed)
    rel = (pb - pa) / np.maximum(pa, 1e-12)
    return float((rel > 0).mean())
