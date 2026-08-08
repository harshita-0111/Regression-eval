# LLM Regression Eval

CI for prompts. Every time `prompts/v1.yaml` changes, this pipeline runs the classifier against a 54-case golden dataset, scores it, diffs against the last known-good run, and blocks the merge if anything regressed past a threshold.

Mental model: unit tests, but for LLM behavior. A prompt change is a code change — this is the test suite for it.

## What it does

The classifier reads a support email, returns a category (`billing` / `technical` / `account` / `general`) and a summary. The classifier itself isn't the interesting part — the eval system around it is:

- **`data/golden_dataset/v1.json`** — 54 hand-labeled test cases, including hard ones (sarcasm, typos, mixed-language, near-empty input)
- **`src/eval_runner.py`** — runs every case, scores category match (binary) and summary quality (LLM-as-judge, 1–5)
- **`src/diff.py`** — compares against the last baseline, flags exactly which cases flipped, fires CRITICAL past an 8% pass-rate drop
- **`src/report.py` / `src/slack_alert.py`** — self-contained HTML report + Slack/Discord alert
- **`.github/workflows/eval-regression.yml`** — runs on every PR touching `/prompts` or `/data/golden_dataset`, posts results as a comment, fails the check on CRITICAL

## Key design decisions

**Golden dataset is hand-labeled, not LLM-generated.** LLM-writes-and-grades measures LLM agreement, not real-world correctness.

**Category and summary are scored differently.** Category has one right answer — exact match. Summaries are free text, so quality needs an LLM-as-judge against a rubric.

**Thresholds exist because 54 cases is a small sample.** One flip is a ~1.85% swing. Warn at 3%, fail at 8% — tunable via `EVAL_WARNING_THRESHOLD` / `EVAL_CRITICAL_THRESHOLD`.

**Judge call is skipped on category mismatch.** A wrong-category case fails regardless of summary quality, so scoring it anyway just burns tokens.

**Every call goes through a shared rate limiter, not just per-call retries.** Independent retries could all back off and retry simultaneously, re-triggering the limit. One shared pacer keeps the whole run under budget.

## Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
python scripts/run_eval.py
```

First run is the baseline. Every run after diffs against the last full run automatically.

Cheap sanity check (doesn't count as a baseline):
```bash
python scripts/run_eval.py --limit 5
```

## Adding a test case

Edit `data/golden_dataset/v1.json` following the existing schema, then:
```bash
python scripts/validate_dataset.py
```
Checks for duplicate IDs and prints category/difficulty coverage.

Label by root cause, not surface symptom — e.g. "reset email never arrived" is `technical` (delivery failure), not `account` (the visible symptom). See notes on `typo_001` for the reasoning.

## CI setup

Repo secrets needed: `GROQ_API_KEY`, `DISCORD_WEBHOOK_URL` (optional — CI still runs without it). Add a branch protection rule requiring the `eval` check to actually gate merges instead of just reporting.

## Debugging a bad run

```bash
python scripts/diagnose_run.py
```
Splits failures into hard errors (usually infra noise), category mismatches (real disagreements), and low judge scores. Check this before trusting a surprising pass-rate drop — it's more often a rate limit or timeout than a real regression.

## Known model behavior

`typo_001`, `account_006`, `account_008` are stable, known disagreements between the model and the labels — not bugs. `general_006` has flipped between runs at temperature 0; Groq doesn't guarantee bit-identical outputs across requests.

## Docker

```bash
docker build -t regression-eval .
docker run -e GROQ_API_KEY=gsk_... -e DISCORD_WEBHOOK_URL=... regression-eval
```
