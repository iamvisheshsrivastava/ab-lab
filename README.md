# AB Lab - End-to-End A/B Testing Toolkit

**Live demo → https://ab-lab.streamlit.app/** (hosted on Streamlit Community Cloud; free tier apps sleep after inactivity — the first load may take a few seconds to wake up)

AB Lab is a small Python library plus a Streamlit app for the parts of A/B testing I kept re-writing by hand: sizing an experiment, simulating fake data to sanity-check a design, running the actual test, and then arguing with myself about whether the result is real.

## Features

- Simulate binary metrics (conversion) and continuous metrics (revenue, time-on-page), with optional seasonality, noise, and bot traffic mixed in so the "clean" case isn't the only one you can test against
- Two-proportion z-test, pooled and Welch's t-test, and Mann-Whitney U for when a metric is too skewed for a t-test to be trustworthy
- Sample-size and MDE calculators (closed-form) plus a Monte Carlo power simulation for when the closed-form normal approximation feels too optimistic
- CUPED variance reduction using a pre-experiment covariate — helps most when that covariate is actually correlated with the metric, does nothing otherwise
- Guardrails: sample-ratio-mismatch (chi-square) and an A/A sanity check
- A Beta-Binomial Bayesian view for conversion experiments, if you want P(B beats A) instead of a p-value
- Streamlit UI wrapping all of the above so you don't have to open a notebook every time

A couple of honest caveats: the Wald CI for a difference in proportions is the quick-and-dirty version, not the more robust Newcombe score interval — fine for a dashboard, not for a paper. `mde_proportions` is a one-shot approximation (it assumes p2 ≈ p1 when estimating the standard error) rather than an iterative solve, so treat it as a ballpark, not a guarantee.

## Quickstart

Clone the repo:

```bash
git clone https://github.com/iamvisheshsrivastava/ab-lab.git
cd ab-lab
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app/app.py
```

## Project Structure

```text
ab-lab/
|-- app/                # Streamlit web app
|   `-- app.py
|-- ablab/              # Core library modules
|   |-- bayes.py
|   |-- guardrails.py
|   |-- metrics.py
|   |-- power.py
|   |-- simulate.py
|   `-- tests.py
|-- notebooks/          # Jupyter examples
|-- examples/           # Script and CLI demos
|-- tests/              # Unit tests
|-- README.md
|-- requirements.txt
`-- pyproject.toml
```

## Example Usage

### Simulate an A/B test

```python
from ablab.simulate import simulate_binomial
from ablab.tests import ztest_proportions

# Generate synthetic data
a, b = simulate_binomial(n=5000, p_control=0.10, lift=0.02)

# Run z-test
result = ztest_proportions(a.sum(), len(a), b.sum(), len(b))
print(result)
```

### CUPED adjustment

```python
from ablab.metrics import cuped

adjusted, theta = cuped(y_metric, covariate)
```

There's also a self-contained script, `examples/run_cli_example.py`, that sizes an
experiment, simulates it, runs the z-test, an A/A check, and CUPED end to end —
useful as a starting point if you'd rather not launch the Streamlit app.

## Testing

Run the test suite with:

```bash
pytest
```

## Tech Stack

- Python
- NumPy, Pandas, SciPy, Statsmodels, scikit-learn
- Streamlit
- Matplotlib and Plotly
- GitHub Actions for CI

## Roadmap

- Add Bayesian posterior plots
- Extend sequential testing demos
- Publish as `pip install ablab`
- Add example datasets

## Contributing

Contributions are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Author

Vishesh Srivastava

- [Portfolio](https://visheshsrivastava.com)
- [LinkedIn](https://linkedin.com/in/iamvisheshsrivastava)
- [GitHub](https://github.com/iamvisheshsrivastava)
