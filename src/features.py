"""Classical ML feature extraction for MealLens.

Extracts a ~230-dimensional feature vector from a single RGB PIL image:
  - RGB histogram: 32 bins/channel → 96 features
  - HSV histogram: 32 bins/channel → 96 features
  - LBP texture histogram: 26 features
  - Per-channel mean + std (RGB + HSV): 12 features
  Total: 230 features
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

# LBP parameters
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS  # 24
LBP_N_BINS = LBP_N_POINTS + 2   # uniform + 1 non-uniform bin = 26

HIST_BINS = 32


def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Convert HxWx3 uint8 RGB array to HxWx3 float32 HSV in [0,1]."""
    img = Image.fromarray(rgb).convert("HSV")
    return np.array(img, dtype=np.float32) / 255.0


def extract_color_histograms(rgb: np.ndarray) -> np.ndarray:
    """Compute normalised RGB and HSV histograms.

    Args:
        rgb: HxWx3 uint8 array.

    Returns:
        Float32 array of length 192 (96 RGB + 96 HSV).
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    hsv = _rgb_to_hsv_array(rgb)

    feats = []
    for img in (rgb_f, hsv):
        for c in range(3):
            hist, _ = np.histogram(img[:, :, c], bins=HIST_BINS, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            feats.append(hist)

    return np.concatenate(feats)  # (192,)


def extract_lbp(rgb: np.ndarray) -> np.ndarray:
    """Compute normalised uniform LBP histogram on grayscale image.

    Args:
        rgb: HxWx3 uint8 array.

    Returns:
        Float32 array of length LBP_N_BINS (26).
    """
    gray = np.array(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255.0
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_N_BINS, range=(0, LBP_N_BINS))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist  # (26,)


def extract_channel_stats(rgb: np.ndarray) -> np.ndarray:
    """Compute per-channel mean and std for RGB and HSV.

    Args:
        rgb: HxWx3 uint8 array.

    Returns:
        Float32 array of length 12 (mean+std for 3 RGB + 3 HSV channels).
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    hsv = _rgb_to_hsv_array(rgb)

    feats = []
    for img in (rgb_f, hsv):
        for c in range(3):
            ch = img[:, :, c]
            feats.extend([ch.mean(), ch.std()])

    return np.array(feats, dtype=np.float32)  # (12,)


def extract_features(image: Image.Image) -> np.ndarray:
    """Extract the full ~230-dim feature vector from a PIL image.

    Resizes to 224x224 before extraction to ensure consistent feature size.

    Args:
        image: PIL RGB image (any size).

    Returns:
        Float32 array of length 230.
    """
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    rgb = np.array(img, dtype=np.uint8)

    color_hist = extract_color_histograms(rgb)  # (192,)
    lbp = extract_lbp(rgb)                       # (26,)
    stats = extract_channel_stats(rgb)           # (12,)

    return np.concatenate([color_hist, lbp, stats])  # (230,)
