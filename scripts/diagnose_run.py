"""
Run: python scripts/diagnose_run.py

Reads the most recent run in /runs and breaks down WHY the pass rate
is what it is: API/schema errors vs category mismatches vs low judge
scores. Run this whenever pass rate looks surprisingly low.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def main() -> None:
    run_files = sorted(RUNS_DIR.glob("*.json"))
    if not run_files:
        print("No runs found in /runs")
        return

    with run_files[-1].open("r", encoding="utf-8") as f:
        run = json.load(f)

    results = run["case_results"]
    total = len(results)

    errors = [r for r in results if r["error"]]
    category_mismatches = [r for r in results if not r["error"] and not r["category_match"]]
    low_summary_score = [
        r for r in results if not r["error"] and r["category_match"] and r["summary_score"] < 3
    ]
    passed = [r for r in results if r["case_passed"]]

    print(f"Run: {run_files[-1].name}")
    print(f"Total cases: {total}")
    print(f"  Passed:                    {len(passed)}")
    print(f"  Hard errors (API/schema):  {len(errors)}")
    print(f"  Category mismatches:       {len(category_mismatches)}")
    print(f"  Low summary score (<3):    {len(low_summary_score)}")

    if errors:
        print("\n--- ERROR BREAKDOWN ---")
        error_types = Counter(r["error"].split(":")[0] for r in errors)
        for err_type, count in error_types.most_common():
            print(f"  {err_type}: {count}")
        print("\n  Sample errors:")
        for r in errors[:5]:
            print(f"    {r['case_id']}: {r['error']}")

    if category_mismatches:
        print("\n--- CATEGORY MISMATCHES (first 10) ---")
        for r in category_mismatches[:10]:
            print(f"  {r['case_id']}: expected={r['expected_category']!r} got={r['actual_category']!r}")

    if low_summary_score:
        print("\n--- LOW SUMMARY SCORES (first 5) ---")
        for r in low_summary_score[:5]:
            print(f"  {r['case_id']}: score={r['summary_score']} reason={r['summary_score_reason']!r}")
            print(f"    expected: {r['expected_summary']}")
            print(f"    actual:   {r['actual_summary']}")


if __name__ == "__main__":
    main()
