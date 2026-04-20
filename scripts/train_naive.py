"""Train and evaluate the naive baseline model.

Predicts the global mean of each macro across the training set for every
input — zero learned parameters. Establishes the MAE floor.

Output: models/naive.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import MACRO_COLS, get_dataloaders

OUTPUT_PATH = Path("models/naive.json")


def train_naive() -> dict[str, float]:
    """Compute per-macro global mean over the training set.

    Returns:
        Dict mapping macro name to mean value.
    """
    print("Loading training data...")
    loaders = get_dataloaders(batch_size=256, num_workers=4)

    all_macros: list[np.ndarray] = []
    for batch in loaders["train"]:
        all_macros.append(batch["macros"].numpy())

    macros = np.concatenate(all_macros, axis=0)  # (N, 4)
    means = macros.mean(axis=0)

    return {col: float(means[i]) for i, col in enumerate(MACRO_COLS)}


def evaluate(means: dict[str, float], loader) -> dict[str, float]:
    """Compute per-macro MAE for a constant-mean predictor.

    Args:
        means: Dict of macro -> mean value.
        loader: DataLoader to evaluate on.

    Returns:
        Dict of macro -> MAE.
    """
    pred = np.array([means[c] for c in MACRO_COLS], dtype=np.float32)

    abs_errors: list[np.ndarray] = []
    for batch in loader:
        targets = batch["macros"].numpy()  # (B, 4)
        abs_errors.append(np.abs(targets - pred[None, :]))

    errors = np.concatenate(abs_errors, axis=0)  # (N, 4)
    return {col: float(errors[:, i].mean()) for i, col in enumerate(MACRO_COLS)}


def main() -> None:
    """Train naive baseline, evaluate, and save."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    means = train_naive()
    print("\nGlobal means (per 100g):")
    for k, v in means.items():
        print(f"  {k}: {v:.2f}")

    print("\nLoading val/test for evaluation...")
    loaders = get_dataloaders(batch_size=256, num_workers=4)

    for split in ("val", "test"):
        mae = evaluate(means, loaders[split])
        print(f"\n{split.upper()} MAE:")
        for k, v in mae.items():
            print(f"  {k}: {v:.2f}")

    OUTPUT_PATH.write_text(json.dumps(means, indent=2))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
