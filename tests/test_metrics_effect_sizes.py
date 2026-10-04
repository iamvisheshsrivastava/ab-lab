
import math
import numpy as np
import pytest

from ablab.metrics import cohens_d, hedges_g


def test_cohens_d_matches_known_value():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    d = cohens_d(a, b)
    # Equal variances, mean shift of 1, pooled sd = sample sd of either group.
    sd = np.std(a, ddof=1)
    assert math.isclose(d, 1.0 / sd, rel_tol=1e-9)


def test_hedges_g_is_close_to_cohens_d_for_large_samples():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=5000)
    b = rng.normal(0.3, 1, size=5000)
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    assert abs(d - g) < 0.01


def test_cohens_d_rejects_single_observation_groups():
    # Issue #12: group size 1 used to silently return NaN instead of raising.
    with pytest.raises(ValueError):
        cohens_d(np.array([1.0]), np.array([2.0]))


def test_cohens_d_rejects_empty_group():
    with pytest.raises(ValueError):
        cohens_d(np.array([]), np.array([1.0, 2.0]))


def test_hedges_g_rejects_single_observation_groups():
    with pytest.raises(ValueError):
        hedges_g(np.array([1.0]), np.array([2.0]))


def test_cohens_d_zero_variance_identical_means_returns_zero():
    # Both groups constant and equal -> well-defined zero effect, not an error.
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([5.0, 5.0, 5.0])
    assert cohens_d(a, b) == 0.0


def test_cohens_d_zero_variance_different_means_raises():
    # Both groups constant but different -> effect size is formally infinite;
    # this must raise rather than silently returning 0.0 or NaN.
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([6.0, 6.0, 6.0])
    with pytest.raises(ValueError):
        cohens_d(a, b)


def test_cohens_d_never_returns_nan():
    cases = [
        (np.array([1.0]), np.array([2.0])),
        (np.array([5.0, 5.0]), np.array([5.0, 5.0])),
    ]
    for a, b in cases:
        try:
            result = cohens_d(a, b)
        except ValueError:
            continue
        assert not math.isnan(result)
