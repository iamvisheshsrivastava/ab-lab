 
"""
AB Lab — End-to-End A/B Testing Toolkit (core package)

Public API re-exports for convenience.
"""
from .simulate import simulate_binomial, simulate_continuous, make_covariate
from .tests import (
    ztest_proportions,
    ttest_independent,
    welch_ttest,
    mannwhitney_u_test,
)
from .metrics import (
    cuped,
    lift_absolute,
    lift_relative,
    cohens_d,
    hedges_g,
    ci_proportion_wilson,
    ci_diff_proportions_wald,
)
from .power import (
    sample_size_proportions,
    sample_size_means,
    mde_proportions,
    power_simulation_binomial,
)
from .guardrails import srm_chisq, aa_sanity_check
from .bayes import (
    beta_posteriors,
    prob_b_beats_a,
    posterior_samples,
    credible_interval_beta,
    prob_relative_lift_gt_zero,
)
from .sequential import (
    mixture_sprt_proportions,
    mixture_sprt_means,
    always_valid_ci_proportions,
    always_valid_ci_means,
)
from .bandits import (
    thompson_sampling_step,
    epsilon_greedy_step,
    ucb1_step,
    simulate_bandit_run,
)
from .report import build_report
__all__ = [
    # simulate
    "simulate_binomial",
    "simulate_continuous",
    "make_covariate",
    # tests
    "ztest_proportions",
    "ttest_independent",
    "welch_ttest",
    "mannwhitney_u_test",
    # metrics
    "cuped",
    "lift_absolute",
    "lift_relative",
    "cohens_d",
    "hedges_g",
    "ci_proportion_wilson",
    "ci_diff_proportions_wald",
    # power
    "sample_size_proportions",
    "sample_size_means",
    "mde_proportions",
    "power_simulation_binomial",
    # guardrails
    "srm_chisq",
    "aa_sanity_check",
    # bayes
    "beta_posteriors",
    "prob_b_beats_a",
    "posterior_samples",
    "credible_interval_beta",
    "prob_relative_lift_gt_zero",
    # sequential
    "mixture_sprt_proportions",
    "mixture_sprt_means",
    "always_valid_ci_proportions",
    "always_valid_ci_means",
    # bandits
    "thompson_sampling_step",
    "epsilon_greedy_step",
    "ucb1_step",
    "simulate_bandit_run",
    # report
    "build_report",
]
