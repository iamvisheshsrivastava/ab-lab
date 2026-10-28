from __future__ import annotations
import numpy as np

def _seasonality_curve(n: int, amplitude: float = 0.0, periods: int = 7, phase: float = 0.0) -> np.ndarray:
    """
    Simple sinusoidal seasonality in [1-amplitude, 1+amplitude].
    """
    if amplitude <= 0:
        return np.ones(n)
    t = np.arange(n)
    curve = 1.0 + amplitude * np.sin(2 * np.pi * t / periods + phase)
    # keep strictly positive scaling
    return np.clip(curve, 1 - amplitude, 1 + amplitude)

def simulate_binomial(
    n: int,
    p_control: float,
    lift: float = 0.0,
    *,
    relative: bool = True,
    seed: int | None = None,
    seasonality_amplitude: float = 0.0,
    seasonality_periods: int = 7,
    noise_std: float = 0.0,
    bot_rate: float = 0.0,
    bot_success_p: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate binary outcomes (e.g., conversion) for A and B.

    Parameters
    ----------
    n : int
        Samples per group.
    p_control : float
        Baseline probability for control.
    lift : float
        Effect size (relative if relative=True, else absolute delta).
    relative : bool
        Interpret lift as relative percentage (e.g., 0.02 => +2% relative).
    seed : int | None
        Random seed.
    seasonality_amplitude : float
        Sinusoidal amplitude in [0, 1). 0 means disabled.
    seasonality_periods : int
        Period length for the seasonal sine wave.
    noise_std : float
        Gaussian noise added to probabilities (clipped to [0,1]).
    bot_rate : float
        Fraction of traffic from "bots" with probability bot_success_p.
    bot_success_p : float
        Success prob for bots.

    Returns
    -------
    (a, b) : arrays of 0/1 with length n each.

    Raises
    ------
    ValueError
        If n is not a positive integer, p_control / bot_rate / bot_success_p
        are outside [0, 1], noise_std is negative, or seasonality_amplitude
        is outside [0, 1) (see issue #16).
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not 0 <= p_control <= 1:
        raise ValueError("p_control must be in [0, 1]")
    if not 0 <= seasonality_amplitude < 1:
        raise ValueError("seasonality_amplitude must be in [0, 1)")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not 0 <= bot_rate <= 1:
        raise ValueError("bot_rate must be in [0, 1]")
    if not 0 <= bot_success_p <= 1:
        raise ValueError("bot_success_p must be in [0, 1]")

    rng = np.random.default_rng(seed)

    season = _seasonality_curve(n, seasonality_amplitude, seasonality_periods, phase=rng.uniform(0, 2*np.pi))
    noise_a = rng.normal(0, noise_std, size=n)
    noise_b = rng.normal(0, noise_std, size=n)

    p_a = np.clip(p_control * season + noise_a, 0.0, 1.0)

    if relative:
        p_b_base = p_control * (1.0 + lift)
    else:
        p_b_base = p_control + lift
    p_b = np.clip(p_b_base * season + noise_b, 0.0, 1.0)

    # mix in bots
    if bot_rate > 0:
        mask_a = rng.random(n) < bot_rate
        mask_b = rng.random(n) < bot_rate
        p_a = np.where(mask_a, bot_success_p, p_a)
        p_b = np.where(mask_b, bot_success_p, p_b)

    a = rng.binomial(1, p_a)
    b = rng.binomial(1, p_b)
    return a.astype(int), b.astype(int)

def simulate_continuous(
    n: int,
    mu_control: float,
    sigma: float,
    lift: float = 0.0,
    *,
    relative: bool = True,
    seed: int | None = None,
    seasonality_amplitude: float = 0.0,
    seasonality_periods: int = 7,
    noise_std: float = 0.0,
    bot_rate: float = 0.0,
    bot_mean: float = 0.0,
    bot_sigma: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate continuous outcomes (e.g., revenue) for A and B.

    Raises
    ------
    ValueError
        If n is not a positive integer, sigma is non-positive, noise_std or
        bot_sigma is negative, bot_rate is outside [0, 1], or
        seasonality_amplitude is outside [0, 1) (see issue #16).
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 <= seasonality_amplitude < 1:
        raise ValueError("seasonality_amplitude must be in [0, 1)")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not 0 <= bot_rate <= 1:
        raise ValueError("bot_rate must be in [0, 1]")
    if bot_sigma < 0:
        raise ValueError("bot_sigma must be non-negative")

    rng = np.random.default_rng(seed)

    season = _seasonality_curve(n, seasonality_amplitude, seasonality_periods, phase=rng.uniform(0, 2*np.pi))
    mu_a = mu_control * season + rng.normal(0, noise_std, size=n)
    if relative:
        mu_b_base = mu_control * (1.0 + lift)
    else:
        mu_b_base = mu_control + lift
    mu_b = mu_b_base * season + rng.normal(0, noise_std, size=n)

    a = rng.normal(mu_a, sigma, size=n)
    b = rng.normal(mu_b, sigma, size=n)

    if bot_rate > 0:
        mask_a = rng.random(n) < bot_rate
        mask_b = rng.random(n) < bot_rate
        a = np.where(mask_a, rng.normal(bot_mean, bot_sigma, size=n), a)
        b = np.where(mask_b, rng.normal(bot_mean, bot_sigma, size=n), b)

    return a, b

def make_covariate(y: np.ndarray, rho: float = 0.5, seed: int | None = None) -> np.ndarray:
    """
    Create a covariate correlated with target y (for CUPED demos).

    Raises
    ------
    ValueError
        If y is empty or rho is outside [-1, 1] (see issue #16).
    """
    if len(y) == 0:
        raise ValueError("y must not be empty")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be in [-1, 1]")
    rng = np.random.default_rng(seed)
    y_std = (y - y.mean()) / (y.std() + 1e-12)
    noise = rng.normal(0, 1, size=len(y))
    x = rho * y_std + np.sqrt(max(1 - rho**2, 0.0)) * noise
    # un-standardize to ~N(0,1)
    return x
