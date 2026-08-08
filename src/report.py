"""
Phase 4: HTML diff report generator.

Builds a single self-contained HTML file (no external deps, no build
step) showing: run metadata, a scorecard vs baseline, every regressed
case with old vs new output side by side, and a trend line over the
last N runs. This is what gets linked from the Slack alert.
"""

from __future__ import annotations

import json
from pathlib import Path

from .diff import RunDiff

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

_SEVERITY_COLORS = {"ok": "#1a7f37", "warning": "#9a6700", "critical": "#cf222e"}


def _load_recent_runs(n: int = 10) -> list[dict]:
    RUNS_DIR.mkdir(exist_ok=True)
    run_files = sorted(RUNS_DIR.glob("*.json"))[-n:]
    runs = []
    for f in run_files:
        with f.open("r", encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def _trend_svg(runs: list[dict], width: int = 600, height: int = 120) -> str:
    """Minimal hand-rolled SVG line chart — no charting lib dependency."""
    if len(runs) < 2:
        return "<p><em>Need at least 2 runs to show a trend.</em></p>"

    pass_rates = [r["pass_rate"] for r in runs]
    n = len(pass_rates)
    pad = 20
    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    def x_at(i: int) -> float:
        return pad + (i / (n - 1)) * plot_w if n > 1 else pad

    def y_at(rate: float) -> float:
        return pad + (1 - rate) * plot_h

    points = " ".join(f"{x_at(i):.1f},{y_at(r):.1f}" for i, r in enumerate(pass_rates))
    dots = "".join(
        f'<circle cx="{x_at(i):.1f}" cy="{y_at(r):.1f}" r="3" fill="#2563eb" />'
        for i, r in enumerate(pass_rates)
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#ccc" />
      <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#ccc" />
      <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2" />
      {dots}
    </svg>
    """


def _case_table(cases: list[dict], columns: list[tuple[str, str]]) -> str:
    if not cases:
        return "<p><em>None.</em></p>"
    header = "".join(f"<th>{label}</th>" for _, label in columns)
    rows = ""
    for c in cases:
        cells = "".join(f"<td>{c.get(key, '')}</td>" for key, _ in columns)
        rows += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"


def generate_html_report(run_record: dict, diff: RunDiff) -> str:
    recent_runs = _load_recent_runs()
    severity_color = _SEVERITY_COLORS.get(diff.severity, "#666")

    regression_table = _case_table(
        diff.regressions,
        [
            ("case_id", "Case"),
            ("expected_category", "Expected"),
            ("baseline_actual_category", "Was"),
            ("current_actual_category", "Now"),
            ("baseline_summary", "Old Summary"),
            ("current_summary", "New Summary"),
        ],
    )
    improvement_table = _case_table(
        diff.improvements, [("case_id", "Case"), ("expected_category", "Expected"), ("current_actual_category", "Now")]
    )

    per_cat_rows = "".join(
        f"<tr><td>{cat}</td><td>{delta * 100:+.1f}%</td></tr>"
        for cat, delta in sorted(diff.per_category_deltas.items())
    )

    baseline_line = (
        f"vs baseline <code>{diff.baseline_run_id}</code>"
        if diff.baseline_run_id
        else "no prior baseline — this run is the new baseline"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Eval Report — {run_record['run_id']}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1f2328; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 16px; margin-top: 32px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 6px; color: white; font-weight: 600; background: {severity_color}; }}
  .scorecard {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 16px 0; }}
  .metric {{ background: #f6f8fa; border-radius: 8px; padding: 12px 20px; min-width: 140px; }}
  .metric .label {{ font-size: 12px; color: #57606a; }}
  .metric .value {{ font-size: 22px; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }}
  th {{ background: #f6f8fa; }}
  code {{ background: #f6f8fa; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>Eval Report — {run_record['prompt_version']} <span class="status-badge">{diff.severity.upper()}</span></h1>
  <p>{run_record['timestamp']} · model <code>{run_record['model']}</code> · dataset <code>{run_record['dataset_version']}</code> · {baseline_line}</p>

  <div class="scorecard">
    <div class="metric"><div class="label">Pass rate</div><div class="value">{run_record['pass_rate']*100:.1f}%</div></div>
    <div class="metric"><div class="label">Category accuracy</div><div class="value">{run_record['category_accuracy']*100:.1f}%</div></div>
    <div class="metric"><div class="label">Pass rate Δ</div><div class="value">{diff.pass_rate_delta*100:+.1f}%</div></div>
    <div class="metric"><div class="label">Avg latency</div><div class="value">{run_record['avg_latency_seconds']:.1f}s</div></div>
    <div class="metric"><div class="label">Tokens</div><div class="value">{run_record['total_prompt_tokens']}/{run_record['total_completion_tokens']}</div></div>
  </div>

  <h2>Per-category accuracy delta</h2>
  <table><thead><tr><th>Category</th><th>Δ</th></tr></thead><tbody>{per_cat_rows}</tbody></table>

  <h2>Trend — pass rate over last {len(recent_runs)} runs</h2>
  {_trend_svg(recent_runs)}

  <h2>Regressions ({len(diff.regressions)})</h2>
  {regression_table}

  <h2>Improvements ({len(diff.improvements)})</h2>
  {improvement_table}
</body>
</html>"""


def save_html_report(run_record: dict, diff: RunDiff) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    html = generate_html_report(run_record, diff)
    path = REPORTS_DIR / f"{run_record['run_id']}_report.html"
    with path.open("w", encoding="utf-8") as f:
        f.write(html)
    return path
