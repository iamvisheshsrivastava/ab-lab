
from __future__ import annotations
import html
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

__all__ = ["build_report"]

_CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #fff; }
h1 { border-bottom: 2px solid #4b7bec; padding-bottom: 0.5rem; }
h2 { color: #4b7bec; margin-top: 2rem; }
h3 { margin-top: 1.25rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem; }
th, td { border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }
th { background: #f4f6fb; width: 35%; }
.badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 0.75rem; font-size: 0.8rem; font-weight: 600; margin-left: 0.5rem; }
.badge-pass { background: #e3fcef; color: #087f5b; }
.badge-fail { background: #fff0f0; color: #c92a2a; }
.meta { color: #666; font-size: 0.9rem; }
"""

def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return html.escape(f"{value:.6g}")
    if isinstance(value, (tuple, list)):
        return html.escape("[" + ", ".join(_fmt_raw(v) for v in value) + "]")
    return html.escape(str(value))

def _fmt_raw(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

def _table_rows(data: Mapping[str, Any]) -> str:
    return "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v)}</td></tr>" for k, v in data.items()
    )

def _section(title: str, data: Mapping[str, Any]) -> str:
    if not data:
        return ""
    return f"<h2>{html.escape(title)}</h2>\n<table>\n{_table_rows(data)}\n</table>"

def build_report(
    results: Mapping[str, Any],
    guardrails: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Render a self-contained HTML report summarizing an experiment run, for
    sharing with stakeholders without a screenshot (issue #9).

    Parameters
    ----------
    results : mapping of result name -> value. No schema is enforced, so
        this works directly with the dicts already produced by
        `ablab.tests.ztest_proportions`/`ttest_independent`, effect sizes
        from `ablab.metrics.cohens_d`/`hedges_g`, CIs, and/or a Bayesian
        summary from `ablab.bayes` -- flatten whichever of those a caller
        has computed into one dict (e.g. {"pvalue": ..., "effect_size": ...,
        "ci_95": (lo, hi)}).
    guardrails : mapping of guardrail name -> its result dict (e.g. output
        of `ablab.guardrails.srm_chisq` / `aa_sanity_check`). If an entry
        contains a "reject_h0" key, a PASS/FAIL badge is rendered next to
        its heading (reject_h0=True -> FAIL, i.e. the guardrail tripped).
    metadata : run configuration / context (metric type, sample sizes,
        baseline rate, etc.). An optional "title" key is used as the report
        heading.

    Returns
    -------
    A standalone HTML document (str) with inline CSS -- no external assets,
    so it can be saved, emailed, or downloaded as-is. To get a PDF, open the
    HTML in a browser and use "Print -> Save as PDF"; this keeps the
    module's dependencies limited to the Python standard library.
    """
    results = dict(results or {})
    guardrails = dict(guardrails or {})
    metadata = dict(metadata or {})

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = html.escape(str(metadata.get("title", "AB Lab Experiment Report")))
    meta_rest = {k: v for k, v in metadata.items() if k != "title"}

    guardrail_parts = []
    for name, info in guardrails.items():
        label = html.escape(str(name))
        badge = ""
        body = ""
        if isinstance(info, Mapping):
            if "reject_h0" in info:
                failed = bool(info["reject_h0"])
                cls = "badge-fail" if failed else "badge-pass"
                text = "FAIL" if failed else "PASS"
                badge = f'<span class="badge {cls}">{text}</span>'
            body = f"<table>\n{_table_rows(info)}\n</table>"
        else:
            body = f"<p>{_fmt(info)}</p>"
        guardrail_parts.append(f"<h3>{label}{badge}</h3>\n{body}")

    guardrails_html = ""
    if guardrail_parts:
        guardrails_html = "<h2>Guardrails</h2>\n" + "\n".join(guardrail_parts)

    body_html = "\n".join(
        part for part in (
            _section("Run metadata", meta_rest),
            _section("Results", results),
            guardrails_html,
        ) if part
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated {generated_at}</p>
{body_html}
</body>
</html>
"""
