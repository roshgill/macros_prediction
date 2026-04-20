"""Model definitions for MealLens.

MealLensModel is imported by both train_deep.py and inference.py.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

NUM_CLASSES = 101


class MealLensModel(nn.Module):
    """EfficientNet-B0 with regression and classification heads.

    Regression head: pool → dropout(0.3) → Linear(1280,256) → ReLU → Linear(256,4)
    Classification head: pool → dropout(0.3) → Linear(1280,101)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features  # 1280

        self.reg_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (macro_preds [B,4], class_logits [B,101])."""
        feats = self.backbone(x)
        return self.reg_head(feats), self.cls_head(feats)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, n: int = 2) -> None:
        """Unfreeze the last n blocks of the EfficientNet backbone."""
        for block in list(self.backbone.blocks)[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.conv_head.parameters():
            p.requires_grad = True
        for p in self.backbone.bn2.parameters():
            p.requires_grad = True
