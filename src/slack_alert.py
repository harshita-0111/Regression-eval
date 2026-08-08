"""
Phase 4: Slack alerting.

Sends a structured message via Slack's incoming webhook API. Kept
dependency-free (raw urllib) so this doesn't force an extra package
just to POST some JSON.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .diff import RunDiff

_SEVERITY_EMOJI = {"ok": "✅", "warning": "⚠️", "critical": "🚨"}


def build_slack_payload(run_record: dict, diff: RunDiff, report_url: str | None = None) -> dict:
    emoji = _SEVERITY_EMOJI.get(diff.severity, "ℹ️")
    lines = [
        f"{emoji} *Eval run — {run_record['prompt_version']}* ({diff.severity.upper()})",
        f"Pass rate: *{run_record['pass_rate']*100:.1f}%*"
        + (f" ({diff.pass_rate_delta*100:+.1f}% vs baseline)" if diff.baseline_run_id else " (new baseline)"),
        f"Category accuracy: *{run_record['category_accuracy']*100:.1f}%*",
    ]
    if diff.regressions:
        case_ids = ", ".join(r["case_id"] for r in diff.regressions[:10])
        more = f" (+{len(diff.regressions) - 10} more)" if len(diff.regressions) > 10 else ""
        lines.append(f"*{len(diff.regressions)} regression(s):* {case_ids}{more}")
    if report_url:
        lines.append(f"<{report_url}|View full diff report>")

    return {"text": "\n".join(lines)}


def send_slack_alert(run_record: dict, diff: RunDiff, report_url: str | None = None) -> bool:
    """
    Returns True if the alert was sent successfully. Never raises —
    a failed alert shouldn't fail the eval run or CI job itself.
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("SLACK_WEBHOOK_URL not set — skipping Slack alert.")
        return False

    payload = build_slack_payload(run_record, diff, report_url)
    return _post_json(webhook_url, payload)


def send_discord_alert(run_record: dict, diff: RunDiff, report_url: str | None = None) -> bool:
    """
    Discord alternative to Slack — same webhook pattern, but Discord's
    payload key is "content" instead of "text", and Discord doesn't
    support Slack's <url|label> link syntax, so links go as plain text.
    """
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL not set — skipping Discord alert.")
        return False

    slack_payload = build_slack_payload(run_record, diff, report_url)
    # Discord doesn't support Slack's <url|label> syntax — convert to plain "label: url"
    text = slack_payload["text"]
    if report_url:
        text = text.replace(f"<{report_url}|View full diff report>", f"View full diff report: {report_url}")

    return _post_json(webhook_url, {"content": text})


def _post_json(webhook_url: str, payload: dict) -> bool:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except urllib.error.URLError as e:
        print(f"Webhook alert failed: {e}")
        return False
