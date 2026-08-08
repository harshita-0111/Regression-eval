"""
Phase 2: Golden dataset schema + loader.

The golden dataset is versioned JSON — separate from prompt versions.
When you change the eval bar (add cases, fix a mislabeled case), bump
dataset_version so eval runs can be traced back to exactly which bar
they were scored against.

IMPORTANT (per project spec): test cases must be hand-written and
hand-labeled. Do NOT generate them with an LLM — the whole point of
this dataset is that it's human-verified ground truth. This module
only handles loading/validating, not generating.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import Category

Difficulty = Literal["easy", "medium", "hard", "ambiguous"]


class GoldenTestCase(BaseModel):
    case_id: str = Field(..., description="Stable unique ID, e.g. 'billing_001'")
    input_email: str
    expected_category: Category
    expected_summary: str
    difficulty: Difficulty = "easy"
    notes: str = Field(
        default="", description="Why this case matters / what it's testing for"
    )

    @field_validator("input_email", "expected_summary")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field cannot be empty")
        return v


class GoldenDataset(BaseModel):
    dataset_version: str
    description: str = ""
    cases: list[GoldenTestCase]

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[GoldenTestCase]) -> list[GoldenTestCase]:
        ids = [c.case_id for c in cases]
        dupes = [cid for cid, count in Counter(ids).items() if count > 1]
        if dupes:
            raise ValueError(f"Duplicate case_id(s) found: {dupes}")
        return cases

    @classmethod
    def from_json(cls, path: str | Path) -> "GoldenDataset":
        import json

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(**raw)

    def category_counts(self) -> dict[str, int]:
        return dict(Counter(c.expected_category for c in self.cases))

    def difficulty_counts(self) -> dict[str, int]:
        return dict(Counter(c.difficulty for c in self.cases))

    def summary(self) -> str:
        lines = [
            f"Dataset {self.dataset_version}: {len(self.cases)} cases",
            f"  by category:   {self.category_counts()}",
            f"  by difficulty: {self.difficulty_counts()}",
        ]
        return "\n".join(lines)
