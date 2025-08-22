 
from ablab.power import (
    sample_size_proportions,
    mde_proportions,
    sample_size_means,
    power_simulation_binomial,
)
from ablab.simulate import simulate_continuous

def test_sample_size_monotonicity_proportions():
    p0 = 0.10
    n_small_lift = sample_size_proportions(p0, mde=0.02, power=0.8, alpha=0.05, relative=True)
    n_big_lift = sample_size_proportions(p0, mde=0.05, power=0.8, alpha=0.05, relative=True)
    assert n_big_lift < n_small_lift  

    n_higher_power = sample_size_proportions(p0, mde=0.03, power=0.9, alpha=0.05, relative=True)
    n_lower_power = sample_size_proportions(p0, mde=0.03, power=0.7, alpha=0.05, relative=True)
    assert n_higher_power > n_lower_power  

def test_mde_matches_sample_size_inversion():
    p0, lift = 0.12, 0.04
    n = sample_size_proportions(p0, mde=lift, power=0.8, alpha=0.05, relative=True)
    mde_est = mde_proportions(p0, n, power=0.8, alpha=0.05, relative=True)
    assert abs(mde_est - lift) < 0.02

def test_power_simulation_close_to_target():
    p0, lift = 0.10, 0.05
    n = sample_size_proportions(p0, mde=lift, power=0.8, alpha=0.05, relative=True)
    power = power_simulation_binomial(p0, lift, n, nsims=400, alpha=0.05, seed=123)
    assert power >= 0.70

def test_sample_size_means_behaves():
    n1 = sample_size_means(sigma=3.0, mde=0.5, power=0.8)
    n2 = sample_size_means(sigma=3.0, mde=1.0, power=0.8)
    assert n2 < n1

    assert n1 > 0 and n2 > 0
