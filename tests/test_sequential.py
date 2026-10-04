
import numpy as np
import pytest

from ablab.sequential import (
    mixture_sprt_proportions,
    mixture_sprt_means,
    always_valid_ci_proportions,
    always_valid_ci_means,
)


def test_mixture_sprt_proportions_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        mixture_sprt_proportions(1, 0, 1, 10)
    with pytest.raises(ValueError):
        mixture_sprt_proportions(11, 10, 1, 10)
    with pytest.raises(ValueError):
        mixture_sprt_proportions(1, 10, 1, 10, alpha=1.5)
    with pytest.raises(ValueError):
        mixture_sprt_proportions(1, 10, 1, 10, tau=0)


def test_mixture_sprt_proportions_detects_large_effect():
    # Large, obvious effect with plenty of data should reject.
    res = mixture_sprt_proportions(1000, 10_000, 2000, 10_000, alpha=0.05)
    assert res["decision"] == "reject"
    assert res["pvalue"] < 0.05
    assert res["e_value"] > 1 / 0.05


def test_mixture_sprt_proportions_continue_or_accept_for_no_effect_small_n():
    # Tiny sample, identical rates: not enough evidence either way to reject.
    res = mixture_sprt_proportions(5, 50, 5, 50, alpha=0.05)
    assert res["decision"] in ("continue", "accept")
    assert res["pvalue"] > 0.05


def test_mixture_sprt_type_i_error_is_controlled_under_repeated_peeking():
    """
    Simulate many A/A experiments (no true effect) and peek at the sequential
    test every 50 observations. The fraction of experiments that ever
    reject at any peek should stay close to (and not wildly exceed) alpha,
    unlike a fixed-horizon p-value recomputed at every peek.
    """
    rng = np.random.default_rng(0)
    alpha = 0.05
    n_reps = 200
    n_peeks = 10
    peek_every = 50
    p_true = 0.10

    false_rejections = 0
    for rep in range(n_reps):
        x_a = x_b = 0
        n_a = n_b = 0
        rejected = False
        for _ in range(n_peeks):
            x_a += int(rng.binomial(peek_every, p_true))
            x_b += int(rng.binomial(peek_every, p_true))
            n_a += peek_every
            n_b += peek_every
            res = mixture_sprt_proportions(x_a, n_a, x_b, n_b, tau=0.1, alpha=alpha)
            if res["decision"] == "reject":
                rejected = True
                break
        if rejected:
            false_rejections += 1

    false_rejection_rate = false_rejections / n_reps
    # Generous slack for simulation noise with n_reps=200; the key property
    # is that it does not blow up toward 1.0 the way naive repeated
    # fixed-horizon peeking would.
    assert false_rejection_rate <= alpha + 0.15


def test_always_valid_ci_proportions_contains_zero_for_aa():
    rng = np.random.default_rng(1)
    p_true = 0.2
    n = 5000
    x_a = int(rng.binomial(n, p_true))
    x_b = int(rng.binomial(n, p_true))
    lo, hi = always_valid_ci_proportions(x_a, n, x_b, n, alpha=0.05)
    assert lo < 0 < hi


def test_always_valid_ci_proportions_excludes_zero_for_large_effect():
    lo, hi = always_valid_ci_proportions(1000, 10_000, 2000, 10_000, alpha=0.05)
    assert lo > 0


def test_mixture_sprt_means_detects_large_effect():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, size=2000)
    b = rng.normal(0.5, 1.0, size=2000)
    res = mixture_sprt_means(a, b, alpha=0.05)
    assert res["decision"] == "reject"


def test_mixture_sprt_means_requires_min_observations():
    with pytest.raises(ValueError):
        mixture_sprt_means(np.array([1.0]), np.array([1.0, 2.0]))


def test_always_valid_ci_means_contains_true_diff():
    rng = np.random.default_rng(3)
    a = rng.normal(10.0, 2.0, size=3000)
    b = rng.normal(10.5, 2.0, size=3000)
    lo, hi = always_valid_ci_means(a, b, alpha=0.05)
    assert lo < 0.5 < hi
