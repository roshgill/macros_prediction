"""Data loading and preprocessing for MealLens.

Loads Food-101 from HuggingFace, joins per-class macro labels from the
USDA lookup CSV, and returns DataLoaders ready for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
MACRO_CSV = Path("data/processed/food101_macros.csv")
MACRO_COLS = ["kcal_per_100g", "protein_g", "carb_g", "fat_g"]

# ImageNet normalization (used for pretrained EfficientNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms() -> transforms.Compose:
    """Augmented transforms for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic transforms for validation and test."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_macro_lookup(csv_path: Path = MACRO_CSV) -> dict[int, np.ndarray]:
    """Load the USDA macro CSV and return a dict mapping class_id -> macro array.

    Returns:
        Mapping from Food-101 class integer label to float32 array [kcal, protein, carb, fat].
    """
    df = pd.read_csv(csv_path)
    return {
        int(row["class_id"]): np.array(
            [row[c] for c in MACRO_COLS], dtype=np.float32
        )
        for _, row in df.iterrows()
    }


def compute_macro_stats(
    macro_lookup: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of macros across all classes for normalization.

    Returns:
        (mean, std) arrays of shape (4,).
    """
    values = np.stack(list(macro_lookup.values()))  # (101, 4)
    return values.mean(axis=0), values.std(axis=0)


class Food101MacroDataset(Dataset):
    """Food-101 dataset with per-class USDA macro labels.

    Args:
        hf_split: HuggingFace dataset split object.
        macro_lookup: Mapping from class_id to macro float32 array.
        transform: Torchvision transform to apply to images.
    """

    def __init__(
        self,
        hf_split,
        macro_lookup: dict[int, np.ndarray],
        transform: transforms.Compose,
    ) -> None:
        self.data = hf_split
        self.macro_lookup = macro_lookup
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        image: Image.Image = item["image"].convert("RGB")
        label: int = item["label"]
        macros = torch.tensor(self.macro_lookup[label], dtype=torch.float32)
        return {
            "image": self.transform(image),
            "label": torch.tensor(label, dtype=torch.long),
            "macros": macros,
        }


def get_dataloaders(
    macro_csv: Path = MACRO_CSV,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    num_workers: int = 4,
    subset_per_class: int | None = None,
) -> dict[Literal["train", "val", "test"], DataLoader]:
    """Build train/val/test DataLoaders for Food-101 + macro labels.

    Args:
        macro_csv: Path to the USDA macro lookup CSV.
        batch_size: Mini-batch size.
        val_fraction: Fraction of training data to hold out as validation.
        num_workers: DataLoader worker processes.
        subset_per_class: If set, cap each class to this many training images
            (useful for fast iteration; document in report if used).

    Returns:
        Dict with keys "train", "val", "test", each a DataLoader.
    """
    macro_lookup = load_macro_lookup(macro_csv)

    ds = load_dataset("ethz/food101")
    train_hf = ds["train"]
    test_hf = ds["validation"]  # Food-101 uses "validation" key for the test split

    # Stratified train/val split using HF's built-in method
    split = train_hf.train_test_split(
        test_size=val_fraction,
        stratify_by_column="label",
        seed=SEED,
    )
    train_hf_split = split["train"]
    val_hf_split = split["test"]

    # Optional per-class subset (for faster classical ML feature extraction)
    if subset_per_class is not None:
        train_hf_split = _subsample_per_class(train_hf_split, subset_per_class)

    train_ds = Food101MacroDataset(train_hf_split, macro_lookup, get_train_transforms())
    val_ds = Food101MacroDataset(val_hf_split, macro_lookup, get_val_transforms())
    test_ds = Food101MacroDataset(test_hf, macro_lookup, get_val_transforms())

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }


def _subsample_per_class(hf_split, n: int):
    """Keep at most n examples per class from a HuggingFace dataset split."""
    rng = np.random.default_rng(SEED)
    indices_by_class: dict[int, list[int]] = {}
    for i, label in enumerate(hf_split["label"]):
        indices_by_class.setdefault(label, []).append(i)

    selected: list[int] = []
    for label_indices in indices_by_class.values():
        arr = np.array(label_indices)
        if len(arr) > n:
            arr = rng.choice(arr, size=n, replace=False)
        selected.extend(arr.tolist())

    return hf_split.select(selected)
