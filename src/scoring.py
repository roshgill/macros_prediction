# scoring.py — Blueprint-aligned meal scoring (macro-based)
# Reference macros: Bryan Johnson's protocol, ~70kg / 183cm body
# Source: Bryan Johnson's published protocol (public web)
#
# LIMITATIONS (documented in report):
# - Macro-only scoring; cannot distinguish food quality, fat type, fiber, or processing
# - BSA scaling assumes similar body composition to Bryan (lean, high muscle mass);
#   users with higher body fat % have lower metabolic demand than BSA scaling estimates
# - No fat quality distinction (EVOO vs butter score identically)
# - No fiber or phytonutrient scoring despite protocol emphasis on both

from __future__ import annotations

import math

BRYAN_HEIGHT_CM = 183   # ~6'0"
BRYAN_WEIGHT_KG = 70
BRYAN_BSA = math.sqrt((BRYAN_HEIGHT_CM * BRYAN_WEIGHT_KG) / 3600)  # ≈ 1.89 m²

BRYAN_DAILY = {
    "kcal": 2250,
    "protein_g": 130,
    "carb_g": 206,
    "fat_g": 101,
}
BRYAN_RATIOS = {"protein": 0.25, "carb": 0.35, "fat": 0.40}
BRYAN_MEALS_PER_DAY = 3  # pre-workout, Super Veggie, final meal


def calculate_personal_targets(height_cm: float, weight_kg: float) -> dict:
    """Scale Bryan's targets by BSA ratio (Mosteller formula)."""
    user_bsa = math.sqrt((height_cm * weight_kg) / 3600)
    scale = user_bsa / BRYAN_BSA
    return {
        "daily_kcal": BRYAN_DAILY["kcal"] * scale,
        "daily_protein_g": BRYAN_DAILY["protein_g"] * scale,
        "daily_carb_g": BRYAN_DAILY["carb_g"] * scale,
        "daily_fat_g": BRYAN_DAILY["fat_g"] * scale,
        "per_meal_kcal": (BRYAN_DAILY["kcal"] * scale) / BRYAN_MEALS_PER_DAY,
        "per_meal_protein_g": (BRYAN_DAILY["protein_g"] * scale) / BRYAN_MEALS_PER_DAY,
        "user_bsa": round(user_bsa, 3),
        "scale_factor": round(scale, 3),
    }


def score_meal(meal_macros: dict, targets: dict) -> dict:
    """
    Four sub-scores, each 0–100. Overall is weighted average.

    Args:
        meal_macros: {
            "kcal_per_100g": float,  # directly from model (for density score)
            "kcal": float,           # per serving (portion weight applied)
            "protein_g": float,      # per serving
            "carb_g": float,         # per serving
            "fat_g": float,          # per serving
        }
        targets: output of calculate_personal_targets()

    Returns:
        Dict with overall score, subscores, verdict, and advice.
    """
    # 1. Protein adequacy
    expected_protein = targets["per_meal_protein_g"]
    protein_ratio = meal_macros["protein_g"] / expected_protein if expected_protein > 0 else 0
    if protein_ratio >= 1.0:
        protein_score = max(60, 100 - (protein_ratio - 1.0) * 40)
    else:
        protein_score = protein_ratio * 100

    # 2. Macro ratio alignment (protein/carb/fat distribution)
    total_macro_g = meal_macros["protein_g"] + meal_macros["carb_g"] + meal_macros["fat_g"]
    if total_macro_g > 0:
        meal_ratios = {
            "protein": meal_macros["protein_g"] / total_macro_g,
            "carb": meal_macros["carb_g"] / total_macro_g,
            "fat": meal_macros["fat_g"] / total_macro_g,
        }
        ratio_distance = sum(abs(meal_ratios[k] - BRYAN_RATIOS[k]) for k in BRYAN_RATIOS)
        macro_ratio_score = max(0, 100 - ratio_distance * 150)
    else:
        macro_ratio_score = 0

    # 3. Calorie density: Blueprint favors plant-heavy, lower-density meals
    kcal_per_100g = meal_macros["kcal_per_100g"]
    if kcal_per_100g <= 180:
        density_score = 100
    elif kcal_per_100g <= 300:
        density_score = 100 - (kcal_per_100g - 180) * 0.5
    else:
        density_score = max(0, 40 - (kcal_per_100g - 300) * 0.2)

    # 4. Meal size appropriateness
    expected_kcal = targets["per_meal_kcal"]
    kcal_ratio = meal_macros["kcal"] / expected_kcal if expected_kcal > 0 else 0
    if 0.7 <= kcal_ratio <= 1.3:
        size_score = 100
    elif kcal_ratio < 0.7:
        size_score = kcal_ratio / 0.7 * 100
    else:
        size_score = max(0, 100 - (kcal_ratio - 1.3) * 60)

    overall = (
        0.35 * protein_score +
        0.25 * macro_ratio_score +
        0.25 * density_score +
        0.15 * size_score
    )

    return {
        "overall": round(overall, 1),
        "subscores": {
            "protein": round(protein_score, 1),
            "macro_ratio": round(macro_ratio_score, 1),
            "calorie_density": round(density_score, 1),
            "portion_size": round(size_score, 1),
        },
        "verdict": _get_verdict(overall),
        "advice": _get_advice(protein_score, macro_ratio_score, density_score, size_score, meal_macros, targets),
    }


def _get_verdict(score: float) -> str:
    if score >= 80:
        return "Aligned"
    if score >= 60:
        return "Close"
    if score >= 40:
        return "Off-target"
    return "Not aligned"


def _get_advice(
    p: float,
    m: float,
    d: float,
    s: float,
    macros: dict,
    targets: dict,
) -> str:
    """Return one-line actionable feedback based on lowest sub-score."""
    lowest = min([(p, "protein"), (m, "ratio"), (d, "density"), (s, "size")], key=lambda x: x[0])
    score, category = lowest
    if score > 75:
        return "Well-balanced meal. Aligned with protocol targets."
    advice_map = {
        "protein": (
            f"Low protein for your size ({macros['protein_g']:.0f}g). "
            f"Protocol targets {targets['per_meal_protein_g']:.0f}g per meal."
        ),
        "ratio": "Macro balance skewed. Protocol targets ~25% protein, 35% carbs, 40% fat.",
        "density": (
            f"High calorie density ({macros['kcal_per_100g']:.0f} kcal/100g). "
            "Protocol favors plant-heavy meals under 180 kcal/100g."
        ),
        "size": "Portion size outside typical meal range for your body weight.",
    }
    return advice_map[category]
