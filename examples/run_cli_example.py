"""
Minimal end-to-end CLI example for ab-lab.

Simulates a binary-metric A/B test (e.g. conversion rate), sizes it up front,
runs the frequentist test, and applies CUPED to see how much variance a
pre-experiment covariate can soak up. No Streamlit required -- just the
`ablab` library.

Run it with:

    python examples/run_cli_example.py
"""
from __future__ import annotations

from ablab.simulate import simulate_binomial, make_covariate
from ablab.tests import ztest_proportions
from ablab.metrics import cuped, lift_relative
from ablab.power import sample_size_proportions
from ablab.guardrails import aa_sanity_check


def main() -> None:
    p_control = 0.10
    relative_lift = 0.08  # +8% relative lift on conversion
    alpha = 0.05
    power = 0.8

    # 1. How many users per arm do we need to detect this lift?
    n_required = sample_size_proportions(
        p_control, relative_lift, alpha=alpha, power=power
    )
    print(f"Required sample size per arm: {n_required:,}")

    # 2. Simulate the experiment (use the required n so the test is well-powered).
    a, b = simulate_binomial(n_required, p_control, relative_lift, seed=42)

    # 3. Run the two-proportion z-test.
    result = ztest_proportions(a.sum(), len(a), b.sum(), len(b))
    print(
        f"z = {result['stat']:.3f}, p = {result['pvalue']:.4f}, "
        f"p_a = {result['estimate'][0]:.4f}, p_b = {result['estimate'][1]:.4f}"
    )
    print(f"Observed relative lift: {lift_relative(*result['estimate']):+.2%}")

    # 4. A/A sanity check on a same-arm split -- should NOT reject at alpha.
    aa = aa_sanity_check(a[: n_required // 2].sum(), n_required // 2,
                          a[n_required // 2:].sum(), n_required - n_required // 2)
    print(f"A/A sanity check p-value: {aa['pvalue']:.4f} (reject_h0={aa['reject_h0']})")

    # 5. CUPED variance reduction using a correlated pre-experiment covariate.
    covariate = make_covariate(a.astype(float), rho=0.5, seed=7)
    adjusted, theta = cuped(a.astype(float), covariate)
    print(f"CUPED theta: {theta:.4f} (raw var={a.astype(float).var():.5f}, "
          f"adjusted var={adjusted.var():.5f})")


if __name__ == "__main__":
    main()
