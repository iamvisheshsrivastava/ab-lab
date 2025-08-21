 
import numpy as np
from ablab.simulate import simulate_binomial, simulate_continuous, make_covariate

def test_binomial_shapes_and_rates():
    n = 20_000
    p0 = 0.10
    lift = 0.05  # +5% relative
    a, b = simulate_binomial(n, p_control=p0, lift=lift, seed=123)
    assert len(a) == n and len(b) == n

    p_a = a.mean()
    p_b = b.mean()
    # sanity: control near baseline, variant higher
    assert abs(p_a - p0) < 0.02
    assert p_b > p_a

def test_continuous_shapes_and_effect():
    n = 5_000
    mu0, sigma, lift = 10.0, 2.0, 0.10
    a, b = simulate_continuous(n, mu_control=mu0, sigma=sigma, lift=lift, seed=7)
    assert len(a) == n and len(b) == n
    assert b.mean() > a.mean()

def test_make_covariate_has_correlation():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=10_000)
    x = make_covariate(y, rho=0.6, seed=1)
    # sample correlation should be close to requested rho
    corr = np.corrcoef(y, x)[0, 1]
    assert 0.5 <= corr <= 0.7
