
import numpy as np
import pytest

from ablab.bayes import (
    beta_posteriors,
    posterior_samples,
    prob_b_beats_a,
    credible_interval_beta,
    prob_relative_lift_gt_zero,
)


def test_beta_posteriors_returns_valid_params():
    post_a, post_b = beta_posteriors(50, 500, 60, 500)
    for (a, b) in (post_a, post_b):
        assert a > 0 and b > 0

    # Default Jeffreys-ish uniform prior (1, 1): posterior = (1 + x, 1 + n - x)
    assert post_a == (51.0, 451.0)
    assert post_b == (61.0, 441.0)


def test_beta_posteriors_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        beta_posteriors(-1, 10, 1, 10)
    with pytest.raises(ValueError):
        beta_posteriors(11, 10, 1, 10)
    with pytest.raises(ValueError):
        beta_posteriors(1, 10, 1, 10, alpha_prior=0)
    with pytest.raises(ValueError):
        beta_posteriors(1, 10, 1, 10, beta_prior=-1)


def test_posterior_samples_shapes_and_range():
    post_a, post_b = beta_posteriors(50, 500, 60, 500)
    pa, pb = posterior_samples(post_a, post_b, nsamples=1_000, seed=0)

    assert pa.shape == (1_000,)
    assert pb.shape == (1_000,)
    assert np.all((pa >= 0) & (pa <= 1))
    assert np.all((pb >= 0) & (pb <= 1))


def test_posterior_samples_is_deterministic_with_seed():
    post_a, post_b = beta_posteriors(50, 500, 60, 500)
    pa1, pb1 = posterior_samples(post_a, post_b, nsamples=500, seed=42)
    pa2, pb2 = posterior_samples(post_a, post_b, nsamples=500, seed=42)

    assert np.array_equal(pa1, pa2)
    assert np.array_equal(pb1, pb2)


def test_prob_b_beats_a_is_near_half_for_identical_arms():
    post_a, post_b = beta_posteriors(500, 1_000, 500, 1_000)
    prob = prob_b_beats_a(post_a, post_b, nsamples=200_000, seed=0)
    assert 0.45 <= prob <= 0.55


def test_prob_b_beats_a_trends_toward_one_as_b_improves():
    post_a, _ = beta_posteriors(100, 1_000, 100, 1_000)
    probs = []
    for b_successes in (100, 200, 400, 700):
        _, post_b = beta_posteriors(100, 1_000, b_successes, 1_000)
        probs.append(prob_b_beats_a(post_a, post_b, nsamples=50_000, seed=1))

    # Monotonically increasing as B's observed rate improves.
    assert all(earlier <= later for earlier, later in zip(probs, probs[1:]))
    assert probs[-1] > 0.99


def test_credible_interval_beta_contains_posterior_mean():
    post = (51.0, 451.0)
    mean = post[0] / (post[0] + post[1])
    lo, hi = credible_interval_beta(post, conf_level=0.95)

    assert lo < mean < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_credible_interval_widens_with_fewer_observations():
    # Same observed rate (~10%), but one posterior backed by far fewer trials
    # -> should have a wider (less certain) credible interval.
    few_obs_post, _ = beta_posteriors(10, 100, 0, 1)
    many_obs_post, _ = beta_posteriors(1_000, 10_000, 0, 1)

    lo_few, hi_few = credible_interval_beta(few_obs_post)
    lo_many, hi_many = credible_interval_beta(many_obs_post)

    assert (hi_few - lo_few) > (hi_many - lo_many)


def test_prob_relative_lift_gt_zero_no_lift_case():
    post_a, post_b = beta_posteriors(500, 1_000, 500, 1_000)
    prob = prob_relative_lift_gt_zero(post_a, post_b, nsamples=200_000, seed=0)
    assert 0.45 <= prob <= 0.55


def test_prob_relative_lift_gt_zero_known_lift_case():
    post_a, post_b = beta_posteriors(100, 1_000, 200, 1_000)
    prob = prob_relative_lift_gt_zero(post_a, post_b, nsamples=200_000, seed=0)
    assert prob > 0.95
