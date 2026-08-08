"""
Phase 1: The LLM feature under test.

A customer support email classifier. The prompt is fully externalized
via PromptConfig — this function has zero hardcoded prompt text, so
swapping /prompts/v1.yaml -> v2.yaml is the only thing that changes
between eval runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from openai import AsyncOpenAI, RateLimitError

from .config import ClassificationInput, ClassificationOutput, PromptConfig

_client: AsyncOpenAI | None = None

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 4.0

# Groq free tier: 30 requests/minute. A per-call retry alone isn't enough
# under sustained concurrent load — many tasks can all hit 429, back off,
# and then all retry at once, re-triggering the limit. A shared pacer
# spaces EVERY request (classify + judge, across all cases) so the whole
# run stays under budget instead of bursting.
_MIN_SECONDS_BETWEEN_REQUESTS = 2.1  # ~28 req/min, just under the 30 RPM ceiling
_rate_limiter_lock = asyncio.Lock()
_last_request_time = 0.0


async def _pace_request() -> None:
    global _last_request_time
    async with _rate_limiter_lock:
        now = time.monotonic()
        wait = _last_request_time + _MIN_SECONDS_BETWEEN_REQUESTS - now
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_time = time.monotonic()


async def _call_with_retry(coro_fn, *args, **kwargs):
    """
    Paces every request against the shared rate limiter, then retries
    on RateLimitError with exponential backoff as a second line of
    defense (e.g. other processes/orgs sharing the same key).
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        await _pace_request()
        try:
            return await coro_fn(*args, **kwargs)
        except RateLimitError as e:
            last_error = e
            wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            await asyncio.sleep(wait)
    raise last_error  # type: ignore[misc]


def _get_client() -> AsyncOpenAI:
    """
    Groq exposes an OpenAI-compatible /v1 endpoint, so the same AsyncOpenAI
    client works — just point base_url at Groq and use a GROQ_API_KEY.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        _client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return _client


async def classify_email(
    email_input: ClassificationInput,
    prompt_config: PromptConfig,
) -> ClassificationOutput:
    """
    Run one email through the classifier feature.

    Raises:
        ValueError: if the model response isn't valid JSON matching the schema.
    """
    client = _get_client()
    messages = prompt_config.as_messages()
    messages.append({"role": "user", "content": email_input.email_text})

    response = await _call_with_retry(
        client.chat.completions.create,
        model=prompt_config.model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw_content = response.choices[0].message.content
    if raw_content is None:
        raise ValueError("Model returned empty content")

    try:
        parsed = json.loads(raw_content)
        return ClassificationOutput(**parsed)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Model output failed schema validation: {raw_content!r}") from e


async def classify_email_with_metadata(
    email_input: ClassificationInput,
    prompt_config: PromptConfig,
) -> tuple[ClassificationOutput | None, dict]:
    """
    Eval-run variant of classify_email: captures latency and token usage
    alongside the parsed output, and never raises — schema/API failures
    are returned as (None, metadata_with_error) so a single bad case
    doesn't crash the whole eval run.
    """
    client = _get_client()
    messages = prompt_config.as_messages()
    messages.append({"role": "user", "content": email_input.email_text})

    start = time.monotonic()
    metadata: dict = {"error": None, "latency_seconds": None, "prompt_tokens": None, "completion_tokens": None}

    try:
        response = await _call_with_retry(
            client.chat.completions.create,
            model=prompt_config.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        metadata["latency_seconds"] = time.monotonic() - start

        if response.usage:
            metadata["prompt_tokens"] = response.usage.prompt_tokens
            metadata["completion_tokens"] = response.usage.completion_tokens

        raw_content = response.choices[0].message.content
        if raw_content is None:
            metadata["error"] = "empty_response"
            return None, metadata

        parsed = json.loads(raw_content)
        return ClassificationOutput(**parsed), metadata

    except Exception as e:  # noqa: BLE001 — a single case's failure must not kill the eval run
        metadata["latency_seconds"] = time.monotonic() - start
        metadata["error"] = f"{type(e).__name__}: {e}"
        return None, metadata


async def classify_email_text(email_text: str, prompt_config: PromptConfig) -> ClassificationOutput:
    """Convenience wrapper when you just have raw text, not a ClassificationInput."""
    return await classify_email(ClassificationInput(email_text=email_text), prompt_config)
