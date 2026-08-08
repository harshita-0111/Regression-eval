"""
Run: python scripts/validate_dataset.py

Loads the golden dataset, validates schema (Pydantic will raise on
malformed entries or duplicate case_ids), and prints coverage stats
so you can see gaps while hand-writing cases toward 50-100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import GoldenDataset

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset" / "v1.json"


def main() -> None:
    dataset = GoldenDataset.from_json(DATASET_PATH)
    print(dataset.summary())

    if len(dataset.cases) < 50:
        print(f"\n⚠ Only {len(dataset.cases)} cases — target is 50-100 before real eval runs.")

    counts = dataset.category_counts()
    for cat in ("billing", "technical", "account", "general"):
        if counts.get(cat, 0) == 0:
            print(f"⚠ No cases for category: {cat}")


if __name__ == "__main__":
    main()
