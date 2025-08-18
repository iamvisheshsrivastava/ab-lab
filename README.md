# 🧪 AB Lab — End-to-End A/B Testing Toolkit

AB Lab is a Python toolkit and Streamlit web app for designing, simulating, and analyzing A/B experiments.
It helps you explore **experiment design → power calculation → inference → decision making**, all in one place.

---

## ✨ Features

* **Data Simulation**

  * Binary (conversion rate) & continuous (revenue/metric) data
  * Add noise, seasonality, or bots for realism
* **Frequentist Inference**

  * t-test, Welch’s test, z-test for proportions, Mann-Whitney U
  * Confidence intervals & effect size estimates
* **Power & Sample Size**

  * Closed-form calculators for MDE & sample size
  * Monte-Carlo simulation for power analysis
* **Bias Reduction**

  * CUPED adjustment for variance reduction
* **Guardrails**

  * A/A sanity checks
  * SRM (sample ratio mismatch) detection
* **Bayesian View (optional)**

  * Beta-Binomial for conversion rate posteriors
* **Interactive UI**

  * Streamlit app with sliders, plots, and results
  * Visual decision report: lift, CI, p-value, decision

---

## 🚀 Quickstart

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

---

## 📂 Project Structure

```
ab-lab/
  app/                # Streamlit web app
    app.py
  ablab/              # Core library (simulation, tests, metrics, etc.)
    simulate.py
    tests.py
    power.py
    bayes.py
    guardrails.py
    metrics.py
  notebooks/          # Jupyter examples
    01_quickstart.ipynb
    02_power_simulation.ipynb
  examples/           # CLI & script demos
  tests/              # Unit tests
  README.md
  requirements.txt
  pyproject.toml
```

---

## 📊 Example Usage

### Simulate an A/B test (binary metric)

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

---

## 📸 Screenshots (TBA)



---

## 🧰 Tech Stack

* **Python** (NumPy, Pandas, SciPy, Statsmodels, scikit-learn)
* **Streamlit** (interactive UI)
* **Matplotlib / Plotly** (visualizations)
* **GitHub Actions** (CI tests & linting)

---

## ✅ Roadmap

* [ ] Add Bayesian posterior plots
* [ ] Extend sequential testing demos
* [ ] Publish as `pip install ablab`
* [ ] Add example datasets

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

[MIT](LICENSE)

---

## 👤 Author

**Vishesh Srivastava**

* 🌐 [Portfolio](https://visheshsrivastava.com)
* 💼 [LinkedIn](https://linkedin.com/in/iamvisheshsrivastava)
* 🐙 [GitHub](https://github.com/iamvisheshsrivastava)

---
