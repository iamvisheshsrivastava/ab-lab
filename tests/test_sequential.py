
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


def test_mixture_sprt_means_type_i_error_is_controlled_under_repeated_peeking():
    """
    Mirrors test_mixture_sprt_type_i_error_is_controlled_under_repeated_peeking
    for the means path (#13): run many A/A experiments (no true effect) and
    peek at the growing sample every `peek_every` observations, relying on
    the *default* tau (i.e. not passing `tau` explicitly).

    Before the fix, the default tau was recomputed from the (growing)
    sample's own pooled standard deviation at every peek -- a different
    mixture distribution at each look -- which breaks the Ville's-inequality
    guarantee and can inflate the false-rejection rate well past alpha. With
    a fixed, non-data-dependent default (`DEFAULT_TAU_MEANS`), the false
    rejection rate across repeated peeks should stay close to alpha.
    """
    rng = np.random.default_rng(7)
    alpha = 0.05
    n_reps = 200
    n_peeks = 10
    peek_every = 50

    false_rejections = 0
    for rep in range(n_reps):
        a_vals = []
        b_vals = []
        rejected = False
        for _ in range(n_peeks):
            a_vals.extend(rng.normal(0.0, 1.0, size=peek_every).tolist())
            b_vals.extend(rng.normal(0.0, 1.0, size=peek_every).tolist())
            res = mixture_sprt_means(
                np.array(a_vals), np.array(b_vals), alpha=alpha
            )
            if res["decision"] == "reject":
                rejected = True
                break
        if rejected:
            false_rejections += 1

    false_rejection_rate = false_rejections / n_reps
    # Same generous slack as the proportions version: the key property is
    # that the false-rejection rate does not blow up toward 1.0 the way
    # recomputing tau from the current sample at every peek would.
    assert false_rejection_rate <= alpha + 0.15


def test_mixture_sprt_means_default_tau_is_fixed_not_data_dependent():
    """
    The default tau must be a constant, independent of the sample passed in
    -- recomputing it from the current sample's variance at each peek (the
    pre-fix behavior) is exactly the bug #13 describes. We prove this by
    checking that omitting `tau` always gives the *same* result as passing
    `tau=DEFAULT_TAU_MEANS` explicitly, even across samples with very
    different scales/variances (which would change a data-dependent tau).
    """
    from ablab.sequential import DEFAULT_TAU_MEANS

    assert DEFAULT_TAU_MEANS == 1.0

    rng = np.random.default_rng(11)
    for sigma in (0.01, 1.0, 50.0):
        a = rng.normal(0.0, sigma, size=200)
        b = rng.normal(0.0, sigma, size=200) + 0.01
        res_default = mixture_sprt_means(a, b, alpha=0.05)
        res_explicit = mixture_sprt_means(a, b, tau=DEFAULT_TAU_MEANS, alpha=0.05)
        assert res_default == res_explicit

        lo_d, hi_d = always_valid_ci_means(a, b, alpha=0.05)
        lo_e, hi_e = always_valid_ci_means(a, b, tau=DEFAULT_TAU_MEANS, alpha=0.05)
        assert (lo_d, hi_d) == (lo_e, hi_e)


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
