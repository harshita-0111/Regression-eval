"""
Interface contract for the eval pipeline.

PromptConfig is loaded from a versioned YAML file in /prompts.
ClassificationInput / ClassificationOutput define the typed I/O
the eval engine consumes downstream — keep these stable, since
Phase 3 (eval engine) depends on this schema not changing under it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Category = Literal["billing", "technical", "account", "general"]


class FewShotExample(BaseModel):
    input: str
    output: "ClassificationOutput"


class PromptConfig(BaseModel):
    version_id: str
    timestamp: datetime
    model: str
    description: str = ""
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    @field_validator("system_prompt")
    @classmethod
    def _non_empty_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("system_prompt cannot be empty")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def as_messages(self) -> list[dict]:
        """Build the message list to send to the LLM: system prompt + few-shots."""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex.input})
            messages.append({"role": "assistant", "content": ex.output.model_dump_json()})
        return messages


class ClassificationInput(BaseModel):
    email_text: str


class ClassificationOutput(BaseModel):
    category: Category
    summary: str


# Resolve forward ref for FewShotExample.output
FewShotExample.model_rebuild()
