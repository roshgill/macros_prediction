"""Train and evaluate the classical ML model (XGBoost multi-output regressor).

Features: RGB/HSV histograms + LBP texture + channel stats (~230 dims).
Model: MultiOutputRegressor(XGBRegressor) with early stopping on val MAE.

Output: models/classical.pkl

Usage:
    python scripts/train_classical.py
    python scripts/train_classical.py --subset 200   # 200 images/class (~20k total)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.multioutput import MultiOutputRegressor  # used as predict wrapper
from xgboost import XGBRegressor

from src.data import MACRO_COLS, SEED, _subsample_per_class, load_macro_lookup
from src.features import extract_features

OUTPUT_PATH = Path("models/classical.pkl")


def build_feature_matrix(
    hf_split,
    macro_lookup: dict[int, np.ndarray],
    desc: str = "extracting",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from a HuggingFace dataset split.

    Args:
        hf_split: HuggingFace dataset split.
        macro_lookup: Class-id to macro array mapping.
        desc: Label for progress printing.

    Returns:
        X: float32 array (N, 230), y: float32 array (N, 4).
    """
    n = len(hf_split)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    print(f"{desc}: {n} images")
    t0 = time.time()
    for i in range(n):
        item = hf_split[i]
        feat = extract_features(item["image"])
        X_list.append(feat)
        y_list.append(macro_lookup[item["label"]])
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n} ({elapsed:.0f}s)")

    return np.stack(X_list).astype(np.float32), np.stack(y_list).astype(np.float32)


def main(subset_per_class: int | None = None) -> None:
    """Train XGBoost multi-output regressor and evaluate."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    macro_lookup = load_macro_lookup()

    print("Loading Food-101...")
    ds = load_dataset("ethz/food101")

    # Build train/val split (same seed/fraction as deep model for fair comparison)
    split = ds["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=SEED)
    train_hf = split["train"]
    val_hf = split["test"]
    test_hf = ds["validation"]

    if subset_per_class is not None:
        train_hf = _subsample_per_class(train_hf, subset_per_class)
        print(f"Using subset: {len(train_hf)} train images ({subset_per_class}/class)")

    X_train, y_train = build_feature_matrix(train_hf, macro_lookup, "Train")
    X_val, y_val = build_feature_matrix(val_hf, macro_lookup, "Val")
    X_test, y_test = build_feature_matrix(test_hf, macro_lookup, "Test")

    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")

    # Train one XGBRegressor per macro target with proper early stopping
    estimators: list[XGBRegressor] = []
    print("\nFitting XGBoost (one estimator per macro target)...")
    t0 = time.time()

    for i, col in enumerate(MACRO_COLS):
        print(f"\n  [{i+1}/4] {col}")
        est = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            device="cpu",
            early_stopping_rounds=20,
            eval_metric="mae",
        )
        est.fit(
            X_train, y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=50,
        )
        estimators.append(est)

    print(f"\nAll targets trained in {time.time() - t0:.0f}s")

    # Wrap in MultiOutputRegressor shell for a unified predict interface
    model = MultiOutputRegressor(XGBRegressor())
    model.estimators_ = estimators
    model.n_outputs_ = 4

    # Evaluate
    for split_name, X, y in [("VAL", X_val, y_val), ("TEST", X_test, y_test)]:
        preds = np.column_stack([est.predict(X) for est in estimators])
        mae = np.abs(preds - y).mean(axis=0)
        print(f"\n{split_name} MAE:")
        for i, col in enumerate(MACRO_COLS):
            print(f"  {col}: {mae[i]:.2f}")

    joblib.dump(estimators, OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Max images per class for training (e.g. 200). Omit for full dataset.",
    )
    args = parser.parse_args()
    main(subset_per_class=args.subset)
