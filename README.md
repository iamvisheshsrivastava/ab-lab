# AB Lab - End-to-End A/B Testing Toolkit

AB Lab is a Python toolkit and Streamlit web app for designing, simulating, and analyzing A/B experiments. It covers the full workflow from experiment design and power calculation to inference and decision support.

## Features

- Data simulation for binary metrics such as conversion rate and continuous metrics such as revenue
- Frequentist inference including t-tests, Welch's test, z-tests for proportions, and Mann-Whitney U
- Power and sample-size estimation with both closed-form calculators and Monte Carlo simulation
- CUPED variance reduction
- Guardrails such as A/A sanity checks and sample-ratio-mismatch detection
- Optional Bayesian views for conversion-rate experiments
- Streamlit UI for interactive experiment exploration

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
