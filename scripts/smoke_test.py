"""
Run: GROQ_API_KEY=gsk_... python scripts/smoke_test.py
Get a free key at: https://console.groq.com/keys

Sanity check for Phase 1 — loads v1.yaml, runs one email through
the classifier, prints the typed output. If this works, Phase 1
is done and you're ready for the golden dataset (Phase 2).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import classify_email_text
from src.config import PromptConfig

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "v1.yaml"

TEST_EMAIL = (
    "Hey, I've been trying to reset my password for an hour and the "
    "reset email never arrives. Can someone help?"
)


async def main() -> None:
    prompt_config = PromptConfig.from_yaml(PROMPT_PATH)
    print(f"Loaded prompt {prompt_config.version_id} ({prompt_config.model})")

    result = await classify_email_text(TEST_EMAIL, prompt_config)
    print("category:", result.category)
    print("summary:", result.summary)


if __name__ == "__main__":
    asyncio.run(main())
