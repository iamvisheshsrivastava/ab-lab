
import numpy as np
import pytest

from ablab.bandits import (
    thompson_sampling_step,
    epsilon_greedy_step,
    ucb1_step,
    simulate_bandit_run,
)


def test_thompson_sampling_step_valid_output():
    arm = thompson_sampling_step([5, 50], [10, 100], seed=0)
    assert arm in (0, 1)


def test_thompson_sampling_step_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        thompson_sampling_step([1, 2], [1])
    with pytest.raises(ValueError):
        thompson_sampling_step([], [])
    with pytest.raises(ValueError):
        thompson_sampling_step([5], [3])  # successes > trials
    with pytest.raises(ValueError):
        thompson_sampling_step([1, 1], [10, 10], alpha_prior=0)


def test_thompson_sampling_favors_better_arm_on_average():
    # Arm 1 has a much higher observed rate; over many draws it should be
    # picked more often than arm 0.
    picks = [
        thompson_sampling_step([5, 90], [100, 100], seed=i) for i in range(500)
    ]
    assert np.mean(picks) > 0.5


def test_epsilon_greedy_step_exploits_when_epsilon_zero():
    arm = epsilon_greedy_step([5, 90], [100, 100], epsilon=0.0, seed=0)
    assert arm == 1


def test_epsilon_greedy_step_explores_unseen_arms_first():
    arm = epsilon_greedy_step([5, 0], [100, 0], epsilon=0.0, seed=0)
    assert arm == 1


def test_epsilon_greedy_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        epsilon_greedy_step([1, 1], [10, 10], epsilon=1.5)


def test_ucb1_step_plays_unseen_arms_first():
    assert ucb1_step([5, 0, 0], [50, 0, 10]) == 1


def test_ucb1_step_rejects_negative_c():
    with pytest.raises(ValueError):
        ucb1_step([1, 1], [10, 10], c=-1.0)


@pytest.mark.parametrize("strategy", ["thompson", "epsilon_greedy", "ucb1"])
def test_simulate_bandit_run_shapes(strategy):
    run = simulate_bandit_run([0.1, 0.3], 500, strategy=strategy, seed=0)
    assert run["allocations"].shape == (500,)
    assert run["rewards"].shape == (500,)
    assert run["cumulative_regret"].shape == (500,)
    assert len(run["arm_counts"]) == 2
    assert sum(run["arm_counts"]) == 500
    assert np.all(np.diff(run["cumulative_regret"]) >= -1e-9)  # non-decreasing


def test_simulate_bandit_run_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        simulate_bandit_run([0.1], 100)  # need >= 2 arms
    with pytest.raises(ValueError):
        simulate_bandit_run([0.1, 1.5], 100)  # not a probability
    with pytest.raises(ValueError):
        simulate_bandit_run([0.1, 0.2], 0)
    with pytest.raises(ValueError):
        simulate_bandit_run([0.1, 0.2], 100, strategy="bogus")


def test_thompson_sampling_converges_to_best_arm_allocation_share():
    # Over a long run, Thompson Sampling should allocate a large majority of
    # traffic to the best arm once it has learned the rates.
    run = simulate_bandit_run([0.05, 0.05, 0.30], 5000, strategy="thompson", seed=42)
    last_quarter = run["allocations"][-1250:]
    best_arm_share = np.mean(last_quarter == 2)
    assert best_arm_share > 0.6


def test_regret_grows_slower_than_linear_for_thompson():
    run = simulate_bandit_run([0.05, 0.30], 4000, strategy="thompson", seed=1)
    regret = run["cumulative_regret"]
    # Average regret per step in the second half should be much lower than
    # in the first half, since the bandit has had time to learn.
    first_half_rate = regret[len(regret)//2 - 1] / (len(regret)//2)
    second_half_rate = (regret[-1] - regret[len(regret)//2 - 1]) / (len(regret)//2)
    assert second_half_rate < first_half_rate
