"""
Run: GROQ_API_KEY=gsk_... python scripts/run_eval.py

Runs the full golden dataset through the current prompt config,
saves the run, diffs it against the previous run, and prints a
pass/warn/fail summary — this is the "did I just break something"
check you'd run before merging a prompt change.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PromptConfig
from src.dataset import GoldenDataset
from src.diff import diff_runs
from src.eval_runner import load_latest_run, run_eval, save_run
from src.report import save_html_report
from src.slack_alert import send_slack_alert

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "v1.yaml"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset" / "v1.json"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


async def main(limit: int | None) -> None:
    prompt_config = PromptConfig.from_yaml(PROMPT_PATH)
    dataset = GoldenDataset.from_json(DATASET_PATH)

    if limit:
        dataset.cases = dataset.cases[:limit]
        print(f"⚠ --limit {limit}: running a partial dataset, do not save this as a baseline")

    print(f"Running eval: prompt {prompt_config.version_id} ({prompt_config.model}) "
          f"against dataset {dataset.dataset_version} ({len(dataset.cases)} cases)...")

    baseline = load_latest_run()
    run_record = await run_eval(prompt_config, dataset)
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
        sent = send_slack_alert(run_record, diff, report_url=f"file://{report_path.resolve()}")
        if sent:
            print("Slack alert sent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases (dev/testing only)")
    args = parser.parse_args()
    asyncio.run(main(args.limit))
