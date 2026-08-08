"""
Run: GROQ_API_KEY=gsk_... python scripts/run_eval.py

Runs the full golden dataset through the current prompt config,
saves the run, diffs it against the previous run, and prints a
pass/warn/fail summary — this is the "did I just break something"
check you'd run before merging a prompt change.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PromptConfig
from src.dataset import GoldenDataset
from src.diff import diff_runs
from src.eval_runner import load_latest_run, run_eval, save_run
from src.report import save_html_report
from src.slack_alert import send_discord_alert, send_slack_alert

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "v1.yaml"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset" / "v1.json"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _write_ci_summary(run_record: dict, diff) -> None:
    """
    Writes eval_summary.md — picked up by the GitHub Actions workflow to
    post as a PR comment and to $GITHUB_STEP_SUMMARY. Kept as a separate
    function so the format can change without touching main().
    """
    status_emoji = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}[diff.severity]
    lines = [
        f"## {status_emoji} Eval Results — {run_record['prompt_version']} ({diff.severity.upper()})",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Pass rate | {run_record['pass_rate']*100:.1f}% |",
        f"| Category accuracy | {run_record['category_accuracy']*100:.1f}% |",
    ]
    if diff.baseline_run_id:
        lines.append(f"| Pass rate delta | {diff.pass_rate_delta*100:+.1f}% |")
        lines.append(f"| vs baseline | `{diff.baseline_run_id}` |")
    else:
        lines.append("| Baseline | none — this run establishes it |")

    if diff.regressions:
        lines.append("")
        lines.append(f"### {len(diff.regressions)} regression(s)")
        lines.append("| Case | Expected | Was | Now |")
        lines.append("|---|---|---|---|")
        for r in diff.regressions:
            lines.append(
                f"| `{r['case_id']}` | {r['expected_category']} | "
                f"{r['baseline_actual_category']} | {r['current_actual_category']} |"
            )

    if diff.improvements:
        lines.append("")
        lines.append(f"### {len(diff.improvements)} improvement(s)")
        lines.append(", ".join(f"`{r['case_id']}`" for r in diff.improvements))

    summary_path = Path(__file__).resolve().parent.parent / "eval_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # Also write directly to the GitHub Actions step summary if running in CI.
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


async def main(limit: int | None) -> None:
    prompt_config = PromptConfig.from_yaml(PROMPT_PATH)
    dataset = GoldenDataset.from_json(DATASET_PATH)

    if limit:
        dataset.cases = dataset.cases[:limit]
        print(f"⚠ --limit {limit}: running a partial dataset, do not save this as a baseline")

    print(f"Running eval: prompt {prompt_config.version_id} ({prompt_config.model}) "
          f"against dataset {dataset.dataset_version} ({len(dataset.cases)} cases)...")

    baseline = load_latest_run()
    run_record = await run_eval(prompt_config, dataset, is_partial=bool(limit))
    run_path = save_run(run_record)

    diff = diff_runs(baseline, run_record)

    print(f"\nSaved run: {run_path.name}")
    print(f"Pass rate:         {_pct(run_record['pass_rate'])}")
    print(f"Category accuracy: {_pct(run_record['category_accuracy'])}")
    print(f"Avg latency:       {run_record['avg_latency_seconds']:.2f}s")
    print(f"Tokens:            {run_record['total_prompt_tokens']} prompt / "
          f"{run_record['total_completion_tokens']} completion")

    if diff.baseline_run_id is None:
        print("\nNo prior run found — this is the new baseline.")
        return

    print(f"\nDiff vs baseline ({diff.baseline_run_id}):")
    print(f"  Pass rate delta:         {diff.pass_rate_delta * 100:+.1f}%")
    print(f"  Category accuracy delta: {diff.category_accuracy_delta * 100:+.1f}%")
    for cat, delta in sorted(diff.per_category_deltas.items()):
        print(f"    {cat:12s} {delta * 100:+.1f}%")

    status_label = {"ok": "PASS", "warning": "⚠ WARNING", "critical": "✗ CRITICAL"}[diff.severity]
    print(f"\nStatus: {status_label}")

    if diff.regressions:
        print(f"\n{len(diff.regressions)} regression(s) (passed before, failing now):")
        for r in diff.regressions:
            print(f"  - {r['case_id']}: expected={r['expected_category']!r} "
                  f"was={r['baseline_actual_category']!r} now={r['current_actual_category']!r}")

    if diff.improvements:
        print(f"\n{len(diff.improvements)} improvement(s) (failed before, passing now):")
        for r in diff.improvements:
            print(f"  - {r['case_id']}: now={r['current_actual_category']!r}")

    report_path = save_html_report(run_record, diff)
    print(f"\nHTML report: {report_path}")

    if not limit:
        # Only alert on full runs — partial --limit runs are dev-only, not signal.
        report_url = f"file://{report_path.resolve()}"
        sent_slack = send_slack_alert(run_record, diff, report_url=report_url)
        sent_discord = send_discord_alert(run_record, diff, report_url=report_url)
        if sent_slack:
            print("Slack alert sent.")
        if sent_discord:
            print("Discord alert sent.")

    _write_ci_summary(run_record, diff)

    # Exit code drives CI: nonzero fails the job, which blocks merge if this
    # check is required on the branch. Only critical regressions block —
    # warnings surface in the PR comment but don't stop the merge.
    if diff.severity == "critical":
        print("\n✗ CRITICAL regression — failing CI check.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases (dev/testing only)")
    args = parser.parse_args()
    asyncio.run(main(args.limit))
