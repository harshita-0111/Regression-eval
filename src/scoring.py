"""
Scoring functions for eval runs.

Category match is binary (exact string match) — no judgment call needed.
Summary quality needs an LLM-as-judge since two different phrasings can
both be "correct." Kept as a separate module so the judge prompt can be
iterated on independently of the eval runner's control flow.
"""

from __future__ import annotations

import json
import re

from .classifier import _call_with_retry, _get_client
from .config import Category

JUDGE_MODEL = "llama-3.3-70b-versatile"

_JUDGE_SYSTEM_PROMPT = """\
You are grading how well a generated summary captures the same information \
as a reference summary of a customer support email.

Score 1-5:
5 = captures the same core issue and key details, no meaningful loss of information
4 = captures the core issue, minor details missing or phrased differently
3 = captures the general topic but misses a meaningfully important detail
2 = only loosely related to the core issue, misses the main point
1 = unrelated or contradicts the reference summary

Respond ONLY with valid JSON: {"score": <int 1-5>, "reason": "<one short sentence>"}
"""


async def score_summary_quality(
    reference_summary: str, generated_summary: str
) -> tuple[int, str]:
    """
    Returns (score 1-5, one-line reason). Score defaults to 1 with an
    error reason if the judge call itself fails or returns malformed JSON —
    fail loud in the eval report, not silently as a false pass.
    """
    client = _get_client()
    user_content = (
        f"Reference summary: {reference_summary}\n"
        f"Generated summary: {generated_summary}"
    )

    try:
        response = await _call_with_retry(
            client.chat.completions.create,
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        score = int(parsed["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"score out of range: {score}")
        return score, str(parsed.get("reason", ""))
    except Exception as e:  # noqa: BLE001 — judge failures must not crash the eval run
        return 1, f"judge_error: {e}"


def score_category_match(expected: Category, actual: Category) -> bool:
    return expected == actual
