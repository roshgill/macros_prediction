"""Unified inference interface for MealLens.

All three models (naive, classical, deep) are accessible through this module.
The deep model is the primary deployed model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.features import extract_features
from src.models import MealLensModel

# Defined inline to avoid importing src.data (which pulls in datasets at top level)
MACRO_COLS = ["kcal_per_100g", "protein_g", "carb_g", "fat_g"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

MODEL_PATH = Path("models/deep.pt")
STATS_PATH = Path("models/macro_stats.json")
NAIVE_PATH = Path("models/naive.json")
CLASSICAL_PATH = Path("models/classical.pkl")

MC_PASSES = 5  # MC dropout forward passes; 5 balances uncertainty vs <2s latency target

# Food-101 class names in label order (must match training label ordering)
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
    "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
    "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]


class DeepModelBundle:
    """Loaded deep model + normalisation stats, ready for inference."""

    def __init__(
        self,
        model: MealLensModel,
        macro_mean: np.ndarray,
        macro_std: np.ndarray,
    ) -> None:
        self.model = model
        self.macro_mean = torch.tensor(macro_mean, dtype=torch.float32, device=DEVICE)
        self.macro_std = torch.tensor(macro_std, dtype=torch.float32, device=DEVICE)
        self.transform = get_val_transforms()


def load_deep_model(
    model_path: Path = MODEL_PATH,
    stats_path: Path = STATS_PATH,
) -> DeepModelBundle:
    """Load trained EfficientNet-B0 weights and normalisation stats.

    Args:
        model_path: Path to deep.pt checkpoint.
        stats_path: Path to macro_stats.json.

    Returns:
        DeepModelBundle ready for inference.
    """
    stats = json.loads(stats_path.read_text())
    macro_mean = np.array(stats["mean"], dtype=np.float32)
    macro_std = np.array(stats["std"], dtype=np.float32)

    model = MealLensModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()

    return DeepModelBundle(model, macro_mean, macro_std)


def _enable_dropout(model: MealLensModel) -> None:
    """Set dropout layers to train mode for MC dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def predict(image: Image.Image, bundle: DeepModelBundle) -> dict:
    """Run deep model inference with MC dropout uncertainty.

    Args:
        image: PIL RGB image.
        bundle: Loaded DeepModelBundle from load_deep_model().

    Returns:
        Dict with per-100g macros, uncertainty, top_classes, inference_ms.
    """
    t0 = time.perf_counter()

    x = bundle.transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    macro_preds: list[np.ndarray] = []
    cls_logits_list: list[torch.Tensor] = []

    with torch.no_grad():
        # MC dropout: keep dropout active for uncertainty passes
        _enable_dropout(bundle.model)
        for _ in range(MC_PASSES):
            reg, cls = bundle.model(x)
            # Denormalise regression output
            pred_raw = reg * bundle.macro_std + bundle.macro_mean
            macro_preds.append(pred_raw.cpu().numpy()[0])
            cls_logits_list.append(cls.cpu())

    preds = np.stack(macro_preds)           # (MC_PASSES, 4)
    mean_preds = preds.mean(axis=0)         # (4,)
    std_preds = preds.std(axis=0)           # (4,)

    # Average class probabilities across passes
    cls_probs = F.softmax(torch.stack(cls_logits_list).mean(dim=0), dim=-1)[0].numpy()
    top3_idx = cls_probs.argsort()[::-1][:3]
    top_classes = [
        {"name": FOOD101_CLASSES[i].replace("_", " "), "confidence": float(cls_probs[i])}
        for i in top3_idx
    ]

    inference_ms = (time.perf_counter() - t0) * 1000

    return {
        "kcal_per_100g": float(mean_preds[0]),
        "protein_g": float(mean_preds[1]),
        "carb_g": float(mean_preds[2]),
        "fat_g": float(mean_preds[3]),
        "uncertainty": {
            "kcal": float(std_preds[0]),
            "protein": float(std_preds[1]),
            "carb": float(std_preds[2]),
            "fat": float(std_preds[3]),
        },
        "top_classes": top_classes,
        "inference_ms": round(inference_ms, 1),
    }


def predict_classical(image: Image.Image, estimators_path: Path = CLASSICAL_PATH) -> dict:
    """Run classical XGBoost inference.

    Args:
        image: PIL RGB image.
        estimators_path: Path to classical.pkl.

    Returns:
        Dict with per-100g macros (no uncertainty or class labels).
    """
    estimators = joblib.load(estimators_path)
    feats = extract_features(image).reshape(1, -1)
    preds = np.array([est.predict(feats)[0] for est in estimators])
    return {col: float(preds[i]) for i, col in enumerate(MACRO_COLS)}


def predict_naive(naive_path: Path = NAIVE_PATH) -> dict:
    """Return the global-mean macro prediction (naive baseline).

    Returns:
        Dict with per-100g macros.
    """
    return json.loads(naive_path.read_text())
