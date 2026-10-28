# Contributing to AB Lab

Thanks for considering a contribution. AB Lab is a small Python stats library
(`ablab/`) plus a Streamlit app (`app/app.py`) on top of it, so most changes
fall into one of two buckets: fixing/extending the library, or the UI that
wraps it. Both are welcome.

## Getting set up

1. Fork the repo and clone your fork:

   ```bash
   git clone https://github.com/<you>/ab-lab.git
   cd ab-lab
   ```

2. Create a virtual environment and install the dev dependencies (this
   includes `pytest` plus an editable install of `ablab` itself, so both the
   app and the test suite can `import ablab`):

   ```bash
   python -m venv venv
   source venv/bin/activate   # venv\Scripts\activate on Windows
   pip install -r requirements-dev.txt
   ```

   If you only want the runtime dependencies (no tests, no editable
   install), `pip install -r requirements.txt` is enough.

3. Run the app locally to sanity-check UI changes:

   ```bash
   streamlit run app/app.py
   ```

## Running tests

The test suite lives in `tests/` and covers each `ablab` module
(`test_bandits.py`, `test_bayes.py`, `test_inference.py`,
`test_metrics_effect_sizes.py`, `test_power.py`, `test_report.py`,
`test_sequential.py`, `test_simulate.py`).

```bash
pytest
```

Run a single file or test while iterating:

```bash
pytest tests/test_metrics_effect_sizes.py -v
pytest tests/test_metrics_effect_sizes.py::test_cohens_d_raises_on_degenerate_input -v
```

CI (`.github/workflows/tests.yml`) runs `pytest -v` against Python 3.10 and
3.12 on every push and pull request to `master`. A PR won't merge if it
doesn't pass.

If you add or change behavior in `ablab/`, add or update a test in the
matching `tests/test_*.py` file. Bug fixes should include a regression test
that fails without the fix.

## Code style

There's no enforced linter/formatter in CI yet, but please match the
existing conventions in `ablab/`:

- `from __future__ import annotations` at the top of modules, with type
  hints on public function signatures.
- Validate inputs explicitly and raise `ValueError` with a clear message for
  bad arguments, rather than letting NumPy/SciPy fail with an opaque error
  or silently returning `NaN`/a nonsense value (see `ablab/metrics.py` and
  `ablab/power.py` for the pattern).
- Docstrings on public functions: a one-line summary, and a short note on
  any non-obvious parameter or caveat (several modules already link back to
  the GitHub issue that motivated a particular check — keep doing that,
  it's helpful context for the next reader).
- Keep `ablab/` free of Streamlit imports; UI-only logic belongs in
  `app/app.py`.
- Prefer small, focused functions over adding parameters to an existing one
  when the behavior is genuinely different (see `bandits.py`'s separate
  `thompson_sampling_step` / `epsilon_greedy_step` / `ucb1_step`).

## Pull request process

1. Open an issue first for anything beyond a small fix, so we can agree on
   the approach before you put time into it. For small, obvious fixes
   (typos, a missing validation check, a missing test) a PR on its own is
   fine.
2. Create a branch off `master` named for what it does, e.g.
   `fix/lift-relative-zero-baseline` or `feat/newcombe-ci`.
3. Keep PRs focused — one issue or one logical change per PR. Unrelated
   cleanup makes review harder; open a separate PR for it.
4. Make sure `pytest` passes locally before pushing.
5. Reference the issue your PR addresses (e.g. `Fixes #15`) in the PR
   description, and briefly describe what changed and why.
6. A maintainer will review and may ask for changes. Once CI is green and
   the review is resolved, it'll be merged.

## Reporting bugs / proposing features

Open a GitHub issue. For bugs, include a minimal code snippet or steps to
reproduce, what you expected, and what actually happened. For features,
a short description of the use case is more useful than implementation
details — we can figure those out together.

Issues labeled `good first issue` are scoped to be small and
self-contained — a good place to start if you're new to the codebase.
