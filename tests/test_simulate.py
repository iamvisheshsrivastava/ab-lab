
import numpy as np
import pytest

from ablab.simulate import simulate_binomial, simulate_continuous, make_covariate

def test_binomial_shapes_and_rates():
    n = 20_000
    p0 = 0.10
    lift = 0.05  # +5% relative
    a, b = simulate_binomial(n, p_control=p0, lift=lift, seed=123)
    assert len(a) == n and len(b) == n

    p_a = a.mean()
    p_b = b.mean()
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
    corr = np.corrcoef(y, x)[0, 1]
    assert 0.5 <= corr <= 0.7


# --- Issue #16: input validation ---

@pytest.mark.parametrize("n", [0, -5, 2.5, "10"])
def test_simulate_binomial_rejects_invalid_n(n):
    # Used to return empty arrays or raise an opaque numpy error.
    with pytest.raises(ValueError, match="n must be a positive integer"):
        simulate_binomial(n, p_control=0.5)

@pytest.mark.parametrize("p_control", [-0.1, 1.1])
def test_simulate_binomial_rejects_out_of_range_p_control(p_control):
    # Used to silently clip the probability instead of raising.
    with pytest.raises(ValueError, match="p_control must be in \\[0, 1\\]"):
        simulate_binomial(100, p_control=p_control)

def test_simulate_binomial_rejects_negative_noise_std():
    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        simulate_binomial(100, p_control=0.5, noise_std=-1.0)

@pytest.mark.parametrize("bot_rate", [-0.1, 1.1])
def test_simulate_binomial_rejects_out_of_range_bot_rate(bot_rate):
    with pytest.raises(ValueError, match="bot_rate must be in \\[0, 1\\]"):
        simulate_binomial(100, p_control=0.5, bot_rate=bot_rate)

@pytest.mark.parametrize("bot_success_p", [-0.1, 1.1])
def test_simulate_binomial_rejects_out_of_range_bot_success_p(bot_success_p):
    with pytest.raises(ValueError, match="bot_success_p must be in \\[0, 1\\]"):
        simulate_binomial(100, p_control=0.5, bot_rate=0.1, bot_success_p=bot_success_p)

def test_simulate_binomial_rejects_out_of_range_seasonality_amplitude():
    with pytest.raises(ValueError, match="seasonality_amplitude must be in \\[0, 1\\)"):
        simulate_binomial(100, p_control=0.5, seasonality_amplitude=1.0)

@pytest.mark.parametrize("n", [0, -5, 2.5, "10"])
def test_simulate_continuous_rejects_invalid_n(n):
    with pytest.raises(ValueError, match="n must be a positive integer"):
        simulate_continuous(n, mu_control=10.0, sigma=2.0)

@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_simulate_continuous_rejects_non_positive_sigma(sigma):
    with pytest.raises(ValueError, match="sigma must be positive"):
        simulate_continuous(100, mu_control=10.0, sigma=sigma)

def test_simulate_continuous_rejects_negative_noise_std():
    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        simulate_continuous(100, mu_control=10.0, sigma=2.0, noise_std=-1.0)

@pytest.mark.parametrize("bot_rate", [-0.1, 1.1])
def test_simulate_continuous_rejects_out_of_range_bot_rate(bot_rate):
    with pytest.raises(ValueError, match="bot_rate must be in \\[0, 1\\]"):
        simulate_continuous(100, mu_control=10.0, sigma=2.0, bot_rate=bot_rate)

def test_simulate_continuous_rejects_negative_bot_sigma():
    with pytest.raises(ValueError, match="bot_sigma must be non-negative"):
        simulate_continuous(100, mu_control=10.0, sigma=2.0, bot_sigma=-1.0)

def test_make_covariate_rejects_empty_y():
    with pytest.raises(ValueError, match="y must not be empty"):
        make_covariate(np.array([]))

@pytest.mark.parametrize("rho", [-1.5, 1.5])
def test_make_covariate_rejects_out_of_range_rho(rho):
    with pytest.raises(ValueError, match="rho must be in \\[-1, 1\\]"):
        make_covariate(np.array([1.0, 2.0, 3.0]), rho=rho)
