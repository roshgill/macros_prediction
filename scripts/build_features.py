"""Extract classical ML features from Food-101 training images.

Runs the feature extraction pipeline defined in src/features.py over a
subset of the Food-101 training split and saves the result as a numpy
archive for inspection or offline classical model training.

Output: data/processed/features_train.npz  (X, y, class_names)

Run: python scripts/build_features.py [--n-per-class N]

Default: 200 images per class (20k total) — matches the classical model
training subset documented in CLAUDE.md §4.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(n_per_class: int = 200) -> None:
    from datasets import load_dataset  # noqa: PLC0415

    from src.data import load_macro_lookup  # noqa: PLC0415
    from src.features import extract_features  # noqa: PLC0415

    macro_csv = Path("data/processed/food101_macros.csv")
    if not macro_csv.exists():
        raise FileNotFoundError(
            f"{macro_csv} not found — run scripts/make_dataset.py first."
        )

    lookup = load_macro_lookup(macro_csv)

    print(f"Loading Food-101 train split (capped at {n_per_class} per class)...")
    ds = load_dataset("ethz/food101", split="train", trust_remote_code=True)

    from collections import defaultdict  # noqa: PLC0415

    class_counts: dict[int, int] = defaultdict(int)
    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []

    for item in ds:
        label: int = item["label"]
        if class_counts[label] >= n_per_class:
            continue
        feats = extract_features(item["image"])
        X_rows.append(feats)
        y_rows.append(lookup[label])
        class_counts[label] += 1
        if sum(class_counts.values()) % 1000 == 0:
            print(f"  {sum(class_counts.values())} images processed...")

    X = np.stack(X_rows)
    y = np.stack(y_rows)

    out = Path("data/processed/features_train.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y)
    print(f"\nSaved {X.shape[0]} feature vectors ({X.shape[1]}D) to {out}")
    print("Next step: python scripts/model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=200)
    args = parser.parse_args()
    main(n_per_class=args.n_per_class)
