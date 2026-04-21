"""Download Food-101 dataset and build USDA macro label CSV.

Orchestrates the full data acquisition pipeline:
  1. Downloads Food-101 images via HuggingFace datasets
  2. Queries USDA FoodData Central to build per-class macro labels

Run: python scripts/make_dataset.py

Requires USDA_API_KEY in environment (free at https://fdc.nal.usda.gov/api-key-signup/).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Step 1: verify the macro CSV exists or build it
    macro_csv = Path("data/processed/food101_macros.csv")
    if macro_csv.exists():
        print(f"Macro CSV already exists at {macro_csv} — skipping USDA lookup.")
        print("Delete it and re-run to refresh.")
    else:
        print("Building USDA macro label CSV...")
        subprocess.run(
            [sys.executable, "scripts/build_macro_lookup.py"],
            check=True,
        )

    # Step 2: verify HuggingFace dataset is accessible (downloaded lazily on first use)
    print("\nVerifying Food-101 dataset access via HuggingFace...")
    try:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset("ethz/food101", split="train[:1]", trust_remote_code=True)
        print(f"Food-101 accessible — sample label: {ds[0]['label']}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not load Food-101 ({exc}).")
        print("The dataset will be downloaded automatically on first training run.")

    print("\nDataset ready. Next step: python scripts/build_features.py")


if __name__ == "__main__":
    main()
