
from ablab.report import build_report


def test_build_report_contains_key_fields():
    results = {
        "pvalue": 0.0321,
        "effect_size": 0.45,
        "ci_95_diff": (0.01, 0.05),
    }
    guardrails = {
        "aa_sanity_check": {"pvalue": 0.6, "reject_h0": False},
        "srm_chisq": {"stat": 0.1, "pvalue": 0.75},
    }
    metadata = {"title": "My Experiment", "samples_per_group": 1000, "alpha": 0.05}

    html = build_report(results, guardrails, metadata)

    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    assert "My Experiment" in html
    assert "0.0321" in html
    assert "0.45" in html
    assert "aa_sanity_check" in html
    assert "srm_chisq" in html
    assert "PASS" in html  # aa_sanity_check reject_h0=False -> PASS badge
    assert "samples_per_group" in html


def test_build_report_marks_failed_guardrail():
    guardrails = {"aa_sanity_check": {"pvalue": 0.0001, "reject_h0": True}}
    html = build_report({"pvalue": 0.01}, guardrails, {})
    assert "FAIL" in html


def test_build_report_handles_missing_optional_sections():
    html = build_report({"pvalue": 0.5})
    assert "<!DOCTYPE html>" in html
    assert "AB Lab Experiment Report" in html  # default title


def test_build_report_escapes_html_in_values():
    html = build_report({"note": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
