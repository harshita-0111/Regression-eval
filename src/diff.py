"""
Phase 3: Comparison logic.

Diffs a new eval run against a baseline run. Flags per-case regressions
(pass -> fail) and improvements (fail -> pass), computes deltas, and
applies configurable significance thresholds so small noise doesn't
trigger false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Configurable thresholds, per project spec: warn at 3%, critical at 8%.
WARNING_THRESHOLD = 0.03
CRITICAL_THRESHOLD = 0.08


@dataclass
class RunDiff:
    baseline_run_id: str | None
    current_run_id: str
    pass_rate_delta: float
    category_accuracy_delta: float
    per_category_deltas: dict[str, float]
    regressions: list[dict] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)
    severity: str = "ok"  # "ok" | "warning" | "critical"

    def to_dict(self) -> dict:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "pass_rate_delta": self.pass_rate_delta,
            "category_accuracy_delta": self.category_accuracy_delta,
            "per_category_deltas": self.per_category_deltas,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "severity": self.severity,
        }


def _severity_for_delta(delta: float) -> str:
    """Delta is expected as a negative number for a drop in pass rate."""
    if delta <= -CRITICAL_THRESHOLD:
        return "critical"
    if delta <= -WARNING_THRESHOLD:
        return "warning"
    return "ok"


def diff_runs(baseline: dict | None, current: dict) -> RunDiff:
    if baseline is None:
        # No baseline yet — first run establishes the bar, nothing to diff against.
        return RunDiff(
            baseline_run_id=None,
            current_run_id=current["run_id"],
            pass_rate_delta=0.0,
            category_accuracy_delta=0.0,
            per_category_deltas={},
            severity="ok",
        )

    pass_rate_delta = current["pass_rate"] - baseline["pass_rate"]
    category_accuracy_delta = current["category_accuracy"] - baseline["category_accuracy"]

    per_category_deltas = {}
    baseline_cats = baseline.get("per_category_accuracy", {})
    current_cats = current.get("per_category_accuracy", {})
    for cat in set(baseline_cats) | set(current_cats):
        base_acc = baseline_cats.get(cat, {}).get("accuracy", 0.0)
        curr_acc = current_cats.get(cat, {}).get("accuracy", 0.0)
        per_category_deltas[cat] = curr_acc - base_acc

    baseline_by_id = {r["case_id"]: r for r in baseline["case_results"]}
    regressions, improvements = [], []
    for r in current["case_results"]:
        base_r = baseline_by_id.get(r["case_id"])
        if base_r is None:
            continue  # case is new since baseline, nothing to diff
        if base_r["case_passed"] and not r["case_passed"]:
            regressions.append(
                {
                    "case_id": r["case_id"],
                    "expected_category": r["expected_category"],
                    "baseline_actual_category": base_r["actual_category"],
                    "current_actual_category": r["actual_category"],
                    "baseline_summary": base_r["actual_summary"],
                    "current_summary": r["actual_summary"],
                }
            )
        elif not base_r["case_passed"] and r["case_passed"]:
            improvements.append(
                {
                    "case_id": r["case_id"],
                    "expected_category": r["expected_category"],
                    "current_actual_category": r["actual_category"],
                }
            )

    return RunDiff(
        baseline_run_id=baseline["run_id"],
        current_run_id=current["run_id"],
        pass_rate_delta=pass_rate_delta,
        category_accuracy_delta=category_accuracy_delta,
        per_category_deltas=per_category_deltas,
        regressions=regressions,
        improvements=improvements,
        severity=_severity_for_delta(pass_rate_delta),
    )
