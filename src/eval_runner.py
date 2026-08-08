"""
Phase 3: Eval engine — test runner.

Runs every case in the golden dataset through the classifier feature,
scores each dimension, and persists the run as a versioned JSON file
under /runs (SQLite is overkill for this scale — JSON is portable and
git-diffable, matching the project's "zero infra" data storage choice).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from .classifier import classify_email_with_metadata
from .config import ClassificationInput, PromptConfig
from .dataset import GoldenDataset
from .scoring import score_category_match, score_summary_quality

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

MAX_CONCURRENT_REQUESTS = 4


async def _run_single_case(case, prompt_config: PromptConfig, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        output, metadata = await classify_email_with_metadata(
            ClassificationInput(email_text=case.input_email), prompt_config
        )

    result: dict = {
        "case_id": case.case_id,
        "difficulty": case.difficulty,
        "expected_category": case.expected_category,
        "expected_summary": case.expected_summary,
        "error": metadata["error"],
        "latency_seconds": metadata["latency_seconds"],
        "prompt_tokens": metadata["prompt_tokens"],
        "completion_tokens": metadata["completion_tokens"],
    }

    if output is None:
        result.update(
            actual_category=None,
            actual_summary=None,
            category_match=False,
            summary_score=1,
            summary_score_reason="no_output",
            case_passed=False,
        )
        return result

    category_match = score_category_match(case.expected_category, output.category)

    if not category_match:
        result.update(
            actual_category=output.category,
            actual_summary=output.summary,
            category_match=False,
            summary_score=None,
            summary_score_reason="skipped_category_mismatch",
            case_passed=False,
        )
        return result

    summary_score, summary_reason = await score_summary_quality(case.expected_summary, output.summary)

    result.update(
        actual_category=output.category,
        actual_summary=output.summary,
        category_match=category_match,
        summary_score=summary_score,
        summary_score_reason=summary_reason,
        case_passed=summary_score >= 3,
    )
    return result


async def run_eval(prompt_config: PromptConfig, dataset: GoldenDataset, is_partial: bool = False) -> dict:
    """Run the full golden dataset through prompt_config and return the run record."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = [_run_single_case(case, prompt_config, semaphore) for case in dataset.cases]
    case_results = await asyncio.gather(*tasks)

    total = len(case_results)
    passed = sum(1 for r in case_results if r["case_passed"])
    category_correct = sum(1 for r in case_results if r["category_match"])

    per_category: dict[str, dict] = {}
    for r in case_results:
        cat = r["expected_category"]
        bucket = per_category.setdefault(cat, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if r["category_match"]:
            bucket["correct"] += 1
    for cat, bucket in per_category.items():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0

    run_record = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_config.version_id,
        "model": prompt_config.model,
        "dataset_version": dataset.dataset_version,
        "is_partial_run": is_partial,
        "total_cases": total,
        "pass_rate": passed / total if total else 0.0,
        "category_accuracy": category_correct / total if total else 0.0,
        "per_category_accuracy": per_category,
        "avg_latency_seconds": sum(
            r["latency_seconds"] for r in case_results if r["latency_seconds"] is not None
        ) / total if total else 0.0,
        "total_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in case_results),
        "total_completion_tokens": sum(r["completion_tokens"] or 0 for r in case_results),
        "case_results": case_results,
    }
    return run_record


def save_run(run_record: dict) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{run_record['run_id']}_{run_record['prompt_version']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)
    return path


def load_latest_run(exclude_run_id: str | None = None, include_partial: bool = False) -> dict | None:
    """
    Find the most recent FULL run in /runs — partial (--limit) runs are
    excluded by default since they're dev sanity checks, not real baselines.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    run_files = sorted(RUNS_DIR.glob("*.json"))
    if exclude_run_id:
        run_files = [f for f in run_files if exclude_run_id not in f.name]

    for path in reversed(run_files):
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        if include_partial or not record.get("is_partial_run", False):
            return record
    return None
