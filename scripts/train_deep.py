"""Train EfficientNet-B0 with dual regression + classification heads.

Architecture:
  - Backbone: EfficientNet-B0 pretrained on ImageNet (via timm)
  - Regression head: pool → dropout(0.3) → Linear(1280,256) → ReLU → Linear(256,4)
  - Classification head: pool → dropout(0.3) → Linear(1280,101)

Loss: α·Huber(regression, normalised) + β·CrossEntropy(classification)
  α=1.0, β=0.3

Training:
  Phase 1 (2 epochs): freeze backbone, train heads
  Phase 2 (7 epochs): unfreeze top-2 blocks, cosine LR schedule

Output: models/deep.pt  (best val regression MAE checkpoint)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data import MACRO_COLS, compute_macro_stats, get_dataloaders, load_macro_lookup
from src.models import MealLensModel

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

ALPHA = 1.0   # regression loss weight
BETA = 0.3    # classification loss weight
PHASE1_EPOCHS = 2
PHASE2_EPOCHS = 7
BATCH_SIZE = 32
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
NUM_CLASSES = 101

OUTPUT_PATH = Path("models/deep.pt")
STATS_PATH = Path("models/macro_stats.json")  # mean/std for denormalisation at inference



def make_optimizer(model: MealLensModel, phase: int) -> AdamW:
    """Build AdamW with different LRs for heads vs backbone."""
    if phase == 1:
        return AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=HEAD_LR,
        )
    # Phase 2: separate param groups
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.reg_head.parameters()) + list(model.cls_head.parameters())
    return AdamW([
        {"params": head_params, "lr": HEAD_LR},
        {"params": backbone_params, "lr": BACKBONE_LR},
    ])


def run_epoch(
    model: MealLensModel,
    loader,
    optimizer: AdamW | None,
    macro_mean: torch.Tensor,
    macro_std: torch.Tensor,
    training: bool,
) -> dict[str, float]:
    """Run one epoch; return dict of loss/mae metrics."""
    model.train(training)
    huber = nn.HuberLoss()
    ce = nn.CrossEntropyLoss()

    total_loss = total_reg = total_cls = 0.0
    abs_errors: list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(DEVICE)
            macros = batch["macros"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            macro_norm = (macros - macro_mean) / macro_std
            reg_pred, cls_pred = model(images)

            loss_reg = huber(reg_pred, macro_norm)
            loss_cls = ce(cls_pred, labels)
            loss = ALPHA * loss_reg + BETA * loss_cls

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_reg += loss_reg.item()
            total_cls += loss_cls.item()

            # Denormalise for MAE in original units
            pred_raw = reg_pred.detach() * macro_std + macro_mean
            abs_errors.append(torch.abs(pred_raw - macros).cpu().numpy())

    n = len(loader)
    errors = np.concatenate(abs_errors, axis=0)
    mae_per_macro = {col: float(errors[:, i].mean()) for i, col in enumerate(MACRO_COLS)}
    return {
        "loss": total_loss / n,
        "loss_reg": total_reg / n,
        "loss_cls": total_cls / n,
        **mae_per_macro,
    }


def main() -> None:
    """Full two-phase training loop."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    # Macro normalisation stats from training labels
    macro_lookup = load_macro_lookup()
    mean_np, std_np = compute_macro_stats(macro_lookup)
    macro_mean = torch.tensor(mean_np, dtype=torch.float32, device=DEVICE)
    macro_std = torch.tensor(std_np, dtype=torch.float32, device=DEVICE)

    # Save stats for inference-time denormalisation
    STATS_PATH.write_text(json.dumps({
        "mean": mean_np.tolist(),
        "std": std_np.tolist(),
        "cols": MACRO_COLS,
    }))

    loaders = get_dataloaders(batch_size=BATCH_SIZE, num_workers=4)
    model = MealLensModel(pretrained=True).to(DEVICE)

    best_val_mae = float("inf")

    # ── Phase 1: frozen backbone ───────────────────────────────────────────────
    print(f"\n=== Phase 1: {PHASE1_EPOCHS} epochs (frozen backbone) ===")
    model.freeze_backbone()
    opt = make_optimizer(model, phase=1)
    scheduler = CosineAnnealingLR(opt, T_max=PHASE1_EPOCHS)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_m = run_epoch(model, loaders["train"], opt, macro_mean, macro_std, training=True)
        val_m = run_epoch(model, loaders["val"], None, macro_mean, macro_std, training=False)
        scheduler.step()

        val_mae_avg = np.mean([val_m[c] for c in MACRO_COLS])
        print(
            f"Epoch {epoch}/{PHASE1_EPOCHS} ({time.time()-t0:.0f}s) | "
            f"train_loss={train_m['loss']:.3f} | "
            f"val_loss={val_m['loss']:.3f} val_mae_avg={val_mae_avg:.2f}"
        )
        _print_maes(val_m)

        if val_mae_avg < best_val_mae:
            best_val_mae = val_mae_avg
            torch.save(model.state_dict(), OUTPUT_PATH)
            print("  → saved checkpoint")

    # ── Phase 2: top-2 blocks unfrozen ────────────────────────────────────────
    print(f"\n=== Phase 2: {PHASE2_EPOCHS} epochs (top-2 blocks unfrozen) ===")
    model.unfreeze_top_blocks(n=2)
    opt = make_optimizer(model, phase=2)
    scheduler = CosineAnnealingLR(opt, T_max=PHASE2_EPOCHS)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_m = run_epoch(model, loaders["train"], opt, macro_mean, macro_std, training=True)
        val_m = run_epoch(model, loaders["val"], None, macro_mean, macro_std, training=False)
        scheduler.step()

        val_mae_avg = np.mean([val_m[c] for c in MACRO_COLS])
        print(
            f"Epoch {epoch}/{PHASE2_EPOCHS} ({time.time()-t0:.0f}s) | "
            f"train_loss={train_m['loss']:.3f} | "
            f"val_loss={val_m['loss']:.3f} val_mae_avg={val_mae_avg:.2f}"
        )
        _print_maes(val_m)

        if val_mae_avg < best_val_mae:
            best_val_mae = val_mae_avg
            torch.save(model.state_dict(), OUTPUT_PATH)
            print("  → saved checkpoint")

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\n=== Test evaluation (best checkpoint) ===")
    model.load_state_dict(torch.load(OUTPUT_PATH, map_location=DEVICE))
    test_m = run_epoch(model, loaders["test"], None, macro_mean, macro_std, training=False)
    print("TEST MAE:")
    _print_maes(test_m)
    print(f"\nBest val MAE avg: {best_val_mae:.2f}")
    print(f"Model saved to {OUTPUT_PATH}")


def _print_maes(metrics: dict[str, float]) -> None:
    for col in MACRO_COLS:
        print(f"  {col}: {metrics[col]:.2f}")


if __name__ == "__main__":
    main()
