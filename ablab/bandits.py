
from __future__ import annotations
import math
import numpy as np
from typing import Sequence

__all__ = [
    "thompson_sampling_step",
    "epsilon_greedy_step",
    "ucb1_step",
    "simulate_bandit_run",
]

def _check_arms(arm_successes: Sequence[int], arm_trials: Sequence[int]) -> int:
    k = len(arm_trials)
    if len(arm_successes) != k:
        raise ValueError("arm_successes and arm_trials must have the same length")
    if k == 0:
        raise ValueError("Need at least one arm")
    for s, t in zip(arm_successes, arm_trials):
        if t < 0 or s < 0 or s > t:
            raise ValueError("Each arm needs 0 <= successes <= trials")
    return k

def thompson_sampling_step(
    arm_successes: Sequence[int],
    arm_trials: Sequence[int],
    *,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    seed: int | None = None,
) -> int:
    """
    Thompson Sampling allocation for Beta-Bernoulli arms: draw one sample
    from each arm's Beta posterior (same conjugate update used by
    `ablab.bayes.beta_posteriors` -- alpha_prior + successes,
    beta_prior + failures) and allocate to the arm with the largest draw.
    """
    _check_arms(arm_successes, arm_trials)
    if alpha_prior <= 0 or beta_prior <= 0:
        raise ValueError("Prior parameters must be positive")
    rng = np.random.default_rng(seed)
    samples = [
        rng.beta(alpha_prior + s, beta_prior + (t - s))
        for s, t in zip(arm_successes, arm_trials)
    ]
    return int(np.argmax(samples))

def epsilon_greedy_step(
    arm_successes: Sequence[int],
    arm_trials: Sequence[int],
    *,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> int:
    """
    Epsilon-greedy allocation: with probability `epsilon` explore a uniformly
    random arm, otherwise exploit the arm with the highest observed success
    rate (unseen arms, rate undefined, are treated as +inf so they are tried
    at least once before pure exploitation kicks in).
    """
    k = _check_arms(arm_successes, arm_trials)
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    rng = np.random.default_rng(seed)
    if rng.random() < epsilon:
        return int(rng.integers(k))
    rates = [
        (s / t) if t > 0 else math.inf for s, t in zip(arm_successes, arm_trials)
    ]
    return int(np.argmax(rates))

def ucb1_step(
    arm_successes: Sequence[int],
    arm_trials: Sequence[int],
    *,
    c: float = 2.0,
) -> int:
    """
    UCB1 allocation: play any never-tried arm first, then the arm maximizing
    the observed rate plus an exploration bonus c * sqrt(log(total) / trials).
    """
    k = _check_arms(arm_successes, arm_trials)
    if c < 0:
        raise ValueError("c (exploration coefficient) must be non-negative")
    for i, t in enumerate(arm_trials):
        if t == 0:
            return i
    total = sum(arm_trials)
    scores = [
        s / t + c * math.sqrt(math.log(total) / t)
        for s, t in zip(arm_successes, arm_trials)
    ]
    return int(np.argmax(scores))

def simulate_bandit_run(
    true_rates: Sequence[float],
    n_steps: int,
    *,
    strategy: str = "thompson",
    seed: int | None = None,
    epsilon: float = 0.1,
    c: float = 2.0,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> dict:
    """
    Simulate `n_steps` of adaptive allocation across Bernoulli arms with the
    given true conversion rates, analogous to `simulate_binomial` for a
    fixed two-arm split.

    strategy : "thompson" | "epsilon_greedy" | "ucb1"

    Returns a dict with keys:
        allocations      : np.ndarray[int], the arm index chosen at each step.
        rewards           : np.ndarray[int], the 0/1 reward observed at each step.
        cumulative_regret : np.ndarray[float], running regret vs. always
                             playing the best true arm.
        arm_counts        : list[int], total trials per arm at the end.
        arm_successes     : list[int], total successes per arm at the end.
    """
    k = len(true_rates)
    if k < 2:
        raise ValueError("Need at least 2 arms")
    if not all(0 <= p <= 1 for p in true_rates):
        raise ValueError("true_rates must be probabilities in [0, 1]")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if strategy not in ("thompson", "epsilon_greedy", "ucb1"):
        raise ValueError(f"Unknown strategy: {strategy!r}")

    rng = np.random.default_rng(seed)
    successes = [0] * k
    trials = [0] * k
    best_rate = max(true_rates)

    allocations = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=int)
    cumulative_regret = np.zeros(n_steps, dtype=float)
    running_regret = 0.0

    for t in range(n_steps):
        if strategy == "thompson":
            arm = thompson_sampling_step(
                successes, trials,
                alpha_prior=alpha_prior, beta_prior=beta_prior,
                seed=int(rng.integers(1_000_000_000)),
            )
        elif strategy == "epsilon_greedy":
            arm = epsilon_greedy_step(
                successes, trials, epsilon=epsilon,
                seed=int(rng.integers(1_000_000_000)),
            )
        else:  # ucb1
            arm = ucb1_step(successes, trials, c=c)

        reward = int(rng.random() < true_rates[arm])
        successes[arm] += reward
        trials[arm] += 1
        running_regret += best_rate - true_rates[arm]

        allocations[t] = arm
        rewards[t] = reward
        cumulative_regret[t] = running_regret

    return {
        "allocations": allocations,
        "rewards": rewards,
        "cumulative_regret": cumulative_regret,
        "arm_counts": trials,
        "arm_successes": successes,
    }
