 
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from ablab.simulate import simulate_binomial, simulate_continuous, make_covariate
from ablab.tests import ztest_proportions, ttest_independent, welch_ttest, mannwhitney_u_test
from ablab.metrics import (
    cuped,
    ci_proportion_wilson,
    ci_diff_proportions_wald,
    lift_absolute,
    lift_relative,
    cohens_d,
    hedges_g,
)
from ablab.power import sample_size_proportions, sample_size_means, power_simulation_binomial, mde_proportions
from ablab.guardrails import srm_chisq, aa_sanity_check
from ablab.bayes import beta_posteriors, credible_interval_beta, prob_b_beats_a, prob_relative_lift_gt_zero

st.set_page_config(page_title="AB Lab", layout="wide")

st.title("🧪 AB Lab — A/B Testing Toolkit")

with st.sidebar:
    st.header("Simulation Settings")
    metric_type = st.radio("Metric type", ["Binary (conversion)", "Continuous (value)"])
    n = st.number_input("Samples per group", 100, 1_000_000, 5000, step=100)
    season_amp = st.slider("Seasonality amplitude", 0.0, 0.9, 0.0, 0.05)
    noise_std = st.slider("Noise (std)", 0.0, 0.5, 0.0, 0.01)
    bot_rate = st.slider("Bot traffic rate", 0.0, 0.9, 0.0, 0.01)
    use_cuped = st.checkbox("Apply CUPED (variance reduction)", value=False)
    show_bayes = st.checkbox("Show Bayesian view", value=False)
    st.divider()
    st.header("Power / Sizing")
    alpha = st.number_input("Significance α", 0.0001, 0.2, 0.05, format="%.4f")
    target_power = st.number_input("Target power", 0.5, 0.9999, 0.8, format="%.3f")

colL, colR = st.columns([1, 1])

if metric_type.startswith("Binary"):
    with st.expander("Binary parameters", expanded=True):
        p0 = st.number_input("Baseline conversion (control)", 0.00001, 0.99999, 0.10, format="%.5f")
        rel_lift = st.number_input("Relative lift for Variant B (e.g. 0.02 = +2%)", -0.9, 5.0, 0.02, step=0.01, format="%.4f")
        bot_p = st.number_input("Bot success probability", 0.0, 1.0, 0.0, step=0.01)
    a, b = simulate_binomial(
        n=n,
        p_control=p0,
        lift=rel_lift,
        relative=True,
        seasonality_amplitude=season_amp,
        noise_std=noise_std,
        bot_rate=bot_rate,
        bot_success_p=bot_p,
    )
    if use_cuped:
        cov_a = make_covariate(a, rho=0.5, seed=42)
        cov_b = make_covariate(b, rho=0.5, seed=43)
        a_adj, theta_a = cuped(a.astype(float), cov_a)
        b_adj, theta_b = cuped(b.astype(float), cov_b)
        # Re-threshold adjusted to 0/1 by probability rounding for demo
        a = (a_adj > np.median(a_adj)).astype(int)
        b = (b_adj > np.median(b_adj)).astype(int)
    x1, x2 = a.sum(), b.sum()
    p1, p2 = x1/len(a), x2/len(b)

    with colL:
        st.subheader("Frequentist Inference")
        res = ztest_proportions(x1, len(a), x2, len(b))
        ci = ci_diff_proportions_wald(x1, len(a), x2, len(b))
        st.metric("Control rate", f"{p1:.4%}")
        st.metric("Variant rate", f"{p2:.4%}")
        st.write(f"**Relative lift**: {lift_relative(p1, p2):.2%}  |  **p-value**: {res['pvalue']:.4g}")
        st.write(f"**95% CI (p_B − p_A)**: [{ci[0]:.4%}, {ci[1]:.4%}]")

        st.subheader("Guardrails")
        srm = srm_chisq(len(a), len(b))
        st.write(f"SRM χ² p-value: **{srm['pvalue']:.4g}**")
        aa = aa_sanity_check(x1, len(a), x2, len(b))
        st.write(f"A/A sanity rejects H₀ at α={alpha:.3f}? **{aa['reject_h0']}** (p={aa['pvalue']:.4g})")

    with colR:
        st.subheader("Visualization")
        df = pd.DataFrame({
            "group": ["A"]*len(a) + ["B"]*len(b),
            "converted": np.r_[a, b],
        })
        fig = px.histogram(df, x="converted", color="group", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Power & Sizing")
        mde_est = mde_proportions(p1, n, alpha=alpha, power=target_power)
        needed_n = sample_size_proportions(p1, mde=abs(rel_lift), alpha=alpha, power=target_power, relative=True)
        st.write(f"Approx. **MDE** at n={n} per group: **{mde_est:.2%}**")
        st.write(f"Needed n/group for lift={rel_lift:.2%}: **{needed_n:,}**")
        if st.button("Monte-Carlo power (quick sim)"):
            pow_est = power_simulation_binomial(p1, rel_lift, n, nsims=1000, alpha=alpha, seed=7)
            st.write(f"Estimated empirical power: **{pow_est:.3f}**")

    if show_bayes:
        st.subheader("Bayesian View (Beta-Binomial)")
        post_a, post_b = beta_posteriors(x1, len(a), x2, len(b))
        ci_a = credible_interval_beta(post_a)
        ci_b = credible_interval_beta(post_b)
        prob_b = prob_b_beats_a(post_a, post_b, nsamples=50_000, seed=1)
        prob_rlift = prob_relative_lift_gt_zero(post_a, post_b, nsamples=50_000, seed=2)
        st.write(f"Posterior A 95% CI: [{ci_a[0]:.4%}, {ci_a[1]:.4%}]")
        st.write(f"Posterior B 95% CI: [{ci_b[0]:.4%}, {ci_b[1]:.4%}]")
        st.write(f"P(B>A): **{prob_b:.3f}**  |  P(relative lift > 0): **{prob_rlift:.3f}**")

with st.expander("Continuous parameters", expanded=True):
    mu0 = st.number_input(
        "Baseline mean (control)",
        min_value=-1_000_000.0,
        max_value=1_000_000.0,
        value=10.0,
        step=0.1,
    )
    sigma = st.number_input(
        "Std dev (both groups)",
        min_value=0.0001,
        max_value=1_000_000.0,
        value=3.0,
        step=0.1,
    )
    rel_lift = st.number_input(
        "Relative lift for Variant B (e.g. 0.05 = +5%)",
        min_value=-0.9,
        max_value=5.0,
        value=0.05,
        step=0.01,
        format="%.2f",
    )
    bot_mean = st.number_input(
        "Bot mean",
        min_value=-1_000_000.0,
        max_value=1_000_000.0,
        value=0.0,
        step=0.1,
    )
    bot_sigma = st.number_input(
        "Bot std",
        min_value=0.0001,
        max_value=1_000_000.0,
        value=0.1,
        step=0.1,
    )


    a, b = simulate_continuous(
        n=n,
        mu_control=mu0,
        sigma=sigma,
        lift=rel_lift,
        relative=True,
        seasonality_amplitude=season_amp,
        noise_std=noise_std,
        bot_rate=bot_rate,
        bot_mean=bot_mean,
        bot_sigma=bot_sigma,
    )
    if use_cuped:
        cov_a = make_covariate(a, rho=0.6, seed=11)
        cov_b = make_covariate(b, rho=0.6, seed=12)
        a, theta_a = cuped(a, cov_a)
        b, theta_b = cuped(b, cov_b)
        st.caption(f"CUPED applied. Theta_A={theta_a:.3f}, Theta_B={theta_b:.3f}")

    with colL:
        st.subheader("Frequentist Inference")
        res_welch = welch_ttest(a, b)
        res_pooled = ttest_independent(a, b, equal_var=True)
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        st.write(f"Welch t-test p-value: **{res_welch['pvalue']:.4g}**")
        st.write(f"Pooled t-test p-value: **{res_pooled['pvalue']:.4g}**")
        st.write(f"Cohen's d: **{d:.3f}**, Hedges' g: **{g:.3f}**")

        st.subheader("Guardrails")
        # for continuous A/A, quick check via Mann-Whitney
        mw = mannwhitney_u_test(a, b)
        st.write(f"Mann–Whitney U p-value: **{mw['pvalue']:.4g}**")

    with colR:
        st.subheader("Visualization")
        df = pd.DataFrame({"value": np.r_[a, b], "group": ["A"]*len(a) + ["B"]*len(b)})
        fig = px.histogram(df, x="value", color="group", nbins=40, barmode="overlay", opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Power & Sizing (means)")
        mde_demo = 0.05 * mu0 if mu0 != 0 else 0.05  # illustrative
        needed_n = sample_size_means(sigma, mde=mde_demo, alpha=alpha, power=target_power)
        st.write(f"For MDE≈{mde_demo:.3g}, σ={sigma:.3g}: **n/group ≈ {needed_n:,}**")
