"""Grad-CAM heatmap generation for MealLens.

Computes Grad-CAM w.r.t. the sum of all 4 regression outputs,
hooked on the last conv layer (backbone.conv_head) of EfficientNet-B0.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

from src.data import get_val_transforms
from src.inference import DEVICE, DeepModelBundle


def compute_gradcam(
    image: Image.Image,
    bundle: DeepModelBundle,
) -> np.ndarray:
    """Compute Grad-CAM activation map for a single image.

    Args:
        image: PIL RGB image.
        bundle: Loaded DeepModelBundle.

    Returns:
        Float32 array of shape (H, W) with values in [0, 1],
        same spatial size as the 7x7 conv output upsampled to 224x224.
    """
    model = bundle.model
    model.eval()

    x = bundle.transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    x.requires_grad_(False)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach().clone())

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach().clone())

    handle_fwd = model.backbone.conv_head.register_forward_hook(forward_hook)
    handle_bwd = model.backbone.conv_head.register_full_backward_hook(backward_hook)

    reg_pred, _ = model(x)
    # Grad-CAM target: sum of all 4 macro outputs
    score = reg_pred.sum()
    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[0].squeeze(0)   # (C, H, W)
    grad = gradients[0].squeeze(0)    # (C, H, W)

    weights = grad.mean(dim=(1, 2))    # (C,)
    cam = (weights[:, None, None] * act).sum(dim=0)  # (H, W)
    cam = torch.clamp(cam, min=0)

    cam_np = cam.cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # Upsample to 224x224
    cam_tensor = torch.tensor(cam_np).unsqueeze(0).unsqueeze(0)
    cam_up = torch.nn.functional.interpolate(
        cam_tensor, size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    return cam_up.astype(np.float32)


def overlay_heatmap(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay Grad-CAM heatmap on the original image.

    Args:
        image: Original PIL image (resized to 224x224 internally).
        cam: Float32 (224, 224) array in [0, 1].
        alpha: Heatmap opacity.

    Returns:
        PIL RGBA image with heatmap overlay.
    """
    import matplotlib.cm as cm

    img_resized = image.convert("RGB").resize((224, 224))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap = colormap(cam)[:, :, :3].astype(np.float32)  # (H, W, 3), drop alpha

    blended = (1 - alpha) * img_arr + alpha * heatmap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def gradcam_to_base64(image: Image.Image, bundle: DeepModelBundle) -> str:
    """Compute Grad-CAM and return a base64-encoded PNG overlay.

    Args:
        image: PIL RGB image.
        bundle: Loaded DeepModelBundle.

    Returns:
        Base64-encoded PNG string.
    """
    cam = compute_gradcam(image, bundle)
    overlay = overlay_heatmap(image, cam)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
