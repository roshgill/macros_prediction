"""Portion-size robustness experiment.

Tests whether models predict macros based on dish identity (robust)
or spurious framing cues (brittle) by applying 4 crop/zoom variants
to every test image and measuring MAE degradation.

Variants:
  1. original    — standard 224x224 val transform
  2. crop_80     — center crop 80% → resize to 224
  3. crop_60     — center crop 60% → resize to 224
  4. zoom_out    — pad with reflection so content fills 80% → resize to 224

Output:
  data/outputs/experiment_results.csv
  data/outputs/experiment_results.png   (degradation curves)
  data/outputs/gradcam/                 (20 sampled images, original vs crop_60)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from torchvision import transforms

from src.data import MACRO_COLS, SEED, load_macro_lookup
from src.features import extract_features
from src.gradcam import gradcam_to_base64, overlay_heatmap, compute_gradcam
from src.inference import (
    FOOD101_CLASSES,
    DeepModelBundle,
    load_deep_model,
    predict_naive,
)

OUTPUT_DIR = Path("data/outputs")
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
RESULTS_CSV = OUTPUT_DIR / "experiment_results.csv"
RESULTS_PLOT = OUTPUT_DIR / "experiment_results.png"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GRADCAM_SAMPLES = 20


# ── Image variant transforms ───────────────────────────────────────────────────

def _to_tensor(pil: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(pil.convert("RGB"))


def make_variants(pil: Image.Image) -> dict[str, Image.Image]:
    """Generate 4 spatial variants of a PIL image, all returned as 224x224 PIL."""
    w, h = pil.size

    # original: standard center-crop to 224
    original = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])(pil)

    # crop_80: center crop 80% of pixels → resize to 224
    crop_w, crop_h = int(w * 0.8), int(h * 0.8)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    crop_80 = pil.crop((left, top, left + crop_w, top + crop_h)).resize((224, 224), Image.BILINEAR)

    # crop_60: center crop 60% → resize to 224
    crop_w, crop_h = int(w * 0.6), int(h * 0.6)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    crop_60 = pil.crop((left, top, left + crop_w, top + crop_h)).resize((224, 224), Image.BILINEAR)

    # zoom_out: pad image so content occupies 80% of final frame → resize to 224
    pad_w = int(w * 0.25)   # 25% padding each side → content = 1/(1+0.5) ≈ 80%
    pad_h = int(h * 0.25)
    zoom_out = ImageOps.expand(pil, border=(pad_w, pad_h, pad_w, pad_h))
    # reflect-pad by mirroring edges
    zoom_out = ImageOps.expand(
        pil,
        border=(pad_w, pad_h, pad_w, pad_h),
        fill=0,
    ).resize((224, 224), Image.BILINEAR)

    return {
        "original": original,
        "crop_80": crop_80,
        "crop_60": crop_60,
        "zoom_out": zoom_out,
    }


# ── Per-model predict functions (take PIL 224x224, return np array (4,)) ──────

def _deep_predict_pil(pil: Image.Image, bundle: DeepModelBundle) -> np.ndarray:
    """Single forward pass (no MC dropout) for speed during experiment."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tfm(pil.convert("RGB")).unsqueeze(0).to(bundle.macro_mean.device)
    bundle.model.eval()
    with torch.no_grad():
        reg, _ = bundle.model(x)
        pred_raw = reg * bundle.macro_std + bundle.macro_mean
    return pred_raw.cpu().numpy()[0]


def _classical_predict_pil(pil: Image.Image, estimators: list) -> np.ndarray:
    feats = extract_features(pil).reshape(1, -1)
    return np.array([est.predict(feats)[0] for est in estimators])


def _naive_predict(naive_means: dict) -> np.ndarray:
    return np.array([naive_means[c] for c in MACRO_COLS], dtype=np.float32)


# ── Main experiment loop ───────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    bundle = load_deep_model()
    estimators = joblib.load("models/classical.pkl")
    naive_means = predict_naive()

    print("Loading test split...")
    ds = load_dataset("ethz/food101")
    test_hf = ds["validation"]
    macro_lookup = load_macro_lookup()

    VARIANTS = ["original", "crop_80", "crop_60", "zoom_out"]
    MODELS = ["naive", "classical", "deep"]

    # Accumulators: errors[model][variant] = list of (4,) abs-error arrays
    errors: dict[str, dict[str, list[np.ndarray]]] = {
        m: {v: [] for v in VARIANTS} for m in MODELS
    }

    # Pick 20 random indices for Grad-CAM samples
    random.seed(SEED)
    gradcam_indices = set(random.sample(range(len(test_hf)), GRADCAM_SAMPLES))

    print(f"Running experiment on {len(test_hf)} test images...")
    for i in range(len(test_hf)):
        item = test_hf[i]
        pil_orig: Image.Image = item["image"].convert("RGB")
        label: int = item["label"]
        gt = macro_lookup[label]  # (4,) ground truth

        variants = make_variants(pil_orig)

        for v_name, v_img in variants.items():
            errors["naive"][v_name].append(np.abs(_naive_predict(naive_means) - gt))
            errors["classical"][v_name].append(np.abs(_classical_predict_pil(v_img, estimators) - gt))
            errors["deep"][v_name].append(np.abs(_deep_predict_pil(v_img, bundle) - gt))

        # Grad-CAM for sampled images: original vs crop_60
        if i in gradcam_indices:
            class_name = FOOD101_CLASSES[label].replace("_", " ")
            for v_name in ("original", "crop_60"):
                cam = compute_gradcam(variants[v_name], bundle)
                overlay = overlay_heatmap(variants[v_name], cam)
                fname = GRADCAM_DIR / f"{i:05d}_{class_name.replace(' ', '_')}_{v_name}.png"
                overlay.save(fname)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(test_hf)}")

    # ── Aggregate results ──────────────────────────────────────────────────────
    rows = []
    for model in MODELS:
        for variant in VARIANTS:
            arr = np.stack(errors[model][variant])  # (N, 4)
            mae = arr.mean(axis=0)
            rows.append({
                "model": model,
                "variant": variant,
                **{f"mae_{col}": float(mae[i]) for i, col in enumerate(MACRO_COLS)},
                "mae_avg": float(mae.mean()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to {RESULTS_CSV}")
    print(df.to_string(index=False))

    # ── Plot degradation curves ────────────────────────────────────────────────
    _plot_degradation(df, RESULTS_PLOT)
    print(f"Plot saved to {RESULTS_PLOT}")


def _plot_degradation(df: pd.DataFrame, path: Path) -> None:
    variant_order = ["original", "crop_80", "crop_60", "zoom_out"]
    variant_labels = ["Original", "80% crop", "60% crop", "Zoom-out"]
    colors = {"naive": "#888888", "classical": "#4C72B0", "deep": "#DD8452"}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    fig.suptitle("Portion-size robustness: MAE vs. spatial variant", fontsize=13)

    for ax, col in zip(axes, MACRO_COLS):
        for model in ["naive", "classical", "deep"]:
            sub = df[df["model"] == model].set_index("variant")
            vals = [sub.loc[v, f"mae_{col}"] for v in variant_order]
            ax.plot(variant_labels, vals, marker="o", label=model, color=colors[model])
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("Variant")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
