"""Tests for src/data.py — data loading and macro join."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.data import (
    Food101MacroDataset,
    _subsample_per_class,
    compute_macro_stats,
    get_train_transforms,
    get_val_transforms,
    load_macro_lookup,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def macro_csv(tmp_path: Path) -> Path:
    """Write a minimal 3-class macro CSV for testing."""
    df = pd.DataFrame([
        {"class_id": 0, "class_name": "pizza", "kcal_per_100g": 266.0,
         "protein_g": 11.0, "carb_g": 33.0, "fat_g": 10.0,
         "fdc_id": 1, "fdc_name": "Pizza", "notes": "auto"},
        {"class_id": 1, "class_name": "sushi", "kcal_per_100g": 150.0,
         "protein_g": 6.0, "carb_g": 28.0, "fat_g": 1.0,
         "fdc_id": 2, "fdc_name": "Sushi", "notes": "auto"},
        {"class_id": 2, "class_name": "tacos", "kcal_per_100g": 210.0,
         "protein_g": 8.0, "carb_g": 25.0, "fat_g": 9.0,
         "fdc_id": 3, "fdc_name": "Tacos", "notes": "auto"},
    ])
    p = tmp_path / "food101_macros.csv"
    df.to_csv(p, index=False)
    return p


def make_dummy_hf_item(label: int) -> dict:
    """Create a minimal dict mimicking a HuggingFace Food-101 item."""
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    return {"image": img, "label": label}


class DummyHFSplit:
    """Minimal list-like object mimicking a HuggingFace dataset split."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx):
        # Support both integer indexing and column-name access (like HF datasets)
        if isinstance(idx, str):
            return [item[idx] for item in self._items]
        return self._items[idx]

    def select(self, indices: list[int]):
        return DummyHFSplit([self._items[i] for i in indices])


# ── tests ─────────────────────────────────────────────────────────────────────

def test_load_macro_lookup_shape(macro_csv: Path) -> None:
    lookup = load_macro_lookup(macro_csv)
    assert len(lookup) == 3
    assert lookup[0].shape == (4,)


def test_load_macro_lookup_values(macro_csv: Path) -> None:
    lookup = load_macro_lookup(macro_csv)
    np.testing.assert_allclose(lookup[0], [266.0, 11.0, 33.0, 10.0])


def test_compute_macro_stats(macro_csv: Path) -> None:
    lookup = load_macro_lookup(macro_csv)
    mean, std = compute_macro_stats(lookup)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert mean[0] == pytest.approx((266 + 150 + 210) / 3, rel=1e-5)


def test_transforms_output_shape() -> None:
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    for tfm in [get_train_transforms(), get_val_transforms()]:
        tensor = tfm(img)
        assert tensor.shape == (3, 224, 224)


def test_dataset_item_types(macro_csv: Path) -> None:
    lookup = load_macro_lookup(macro_csv)
    items = [make_dummy_hf_item(label=0), make_dummy_hf_item(label=1)]
    ds = Food101MacroDataset(DummyHFSplit(items), lookup, get_val_transforms())
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["macros"].shape == (4,)
    assert item["label"].dtype == torch.long


def test_dataset_macro_values_match_lookup(macro_csv: Path) -> None:
    lookup = load_macro_lookup(macro_csv)
    items = [make_dummy_hf_item(label=2)]
    ds = Food101MacroDataset(DummyHFSplit(items), lookup, get_val_transforms())
    np.testing.assert_allclose(ds[0]["macros"].numpy(), [210.0, 8.0, 25.0, 9.0])


def test_subsample_per_class_caps() -> None:
    items = [make_dummy_hf_item(label=i % 3) for i in range(30)]
    split = DummyHFSplit(items)
    result = _subsample_per_class(split, n=3)
    labels = [result[i]["label"] for i in range(len(result))]
    from collections import Counter
    counts = Counter(labels)
    assert all(v <= 3 for v in counts.values())
    assert len(result) == 9
