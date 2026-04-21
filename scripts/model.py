"""Train all three MealLens models sequentially.

Runs the full training pipeline:
  1. Naive baseline  → models/naive.json
  2. Classical ML    → models/classical.pkl
  3. Deep learning   → models/deep.pt

Run: python scripts/model.py [--skip-deep]

Use --skip-deep on CPU-only machines to skip the EfficientNet fine-tune
(which requires a GPU and takes ~2 hours on the full dataset).
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(script: str) -> None:
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print("=" * 60)
    subprocess.run([sys.executable, script], check=True)


def main(skip_deep: bool = False) -> None:
    run("scripts/train_naive.py")
    run("scripts/train_classical.py")

    if skip_deep:
        print("\nSkipping deep model training (--skip-deep flag set).")
        print("Pre-trained weights are available at models/deep.pt")
    else:
        run("scripts/train_deep.py")

    print("\nAll models trained. Start the app with: make serve")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-deep",
        action="store_true",
        help="Skip EfficientNet training (use pre-trained weights in models/deep.pt)",
    )
    args = parser.parse_args()
    main(skip_deep=args.skip_deep)
