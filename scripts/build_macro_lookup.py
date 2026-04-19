"""Build food101_macros.csv by querying the USDA FoodData Central API.

Produces one row per Food-101 class with per-100g macro values.
Run once: python scripts/build_macro_lookup.py

Requires USDA_API_KEY environment variable (free at https://fdc.nal.usda.gov/api-key-signup/).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

USDA_API_KEY = os.environ.get("USDA_API_KEY", "")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_FOOD_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"

OUTPUT_PATH = Path("data/processed/food101_macros.csv")

# Food-101 class names (101 classes, underscores as separators)
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

# Manual disambiguation notes for common edge cases — agent augments this at runtime
MANUAL_NOTES: dict[str, str] = {
    "caesar_salad": "matched to FNDDS entry including dressing",
    "french_fries": "matched to fried, not baked",
    "cheese_plate": "averaged cheddar, brie, gouda entries as representative mixed plate",
    "chicken_curry": "matched to FNDDS chicken curry with sauce, not dry spice blend",
    "poutine": "matched to FNDDS poutine (fries + gravy + cheese curds)",
    "foie_gras": "matched to duck liver pate as closest FNDDS entry",
    "bibimbap": "matched to FNDDS Korean bibimbap mixed rice bowl",
    "pad_thai": "matched to FNDDS pad thai with shrimp",
    "peking_duck": "matched to roasted duck skin + meat composite",
}

NUTRIENT_IDS = {
    "kcal": 1008,     # Energy (kcal)
    "protein_g": 1003,  # Protein
    "carb_g": 1005,   # Carbohydrate, by difference
    "fat_g": 1004,    # Total lipid (fat)
}


def search_food(query: str, data_type: str = "Survey (FNDDS)") -> list[dict]:
    """Search USDA FoodData Central for a food query."""
    params = {
        "query": query,
        "dataType": data_type,
        "pageSize": 5,
        "api_key": USDA_API_KEY,
    }
    resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("foods", [])


def get_macros_from_food(food: dict) -> dict[str, float]:
    """Extract macro nutrients from a USDA food search result."""
    nutrient_map = {n["nutrientId"]: n["value"] for n in food.get("foodNutrients", [])}
    return {
        "kcal_per_100g": nutrient_map.get(NUTRIENT_IDS["kcal"], 0.0),
        "protein_g": nutrient_map.get(NUTRIENT_IDS["protein_g"], 0.0),
        "carb_g": nutrient_map.get(NUTRIENT_IDS["carb_g"], 0.0),
        "fat_g": nutrient_map.get(NUTRIENT_IDS["fat_g"], 0.0),
    }


def class_to_query(class_name: str) -> str:
    """Convert Food-101 class name to a human-readable search query."""
    return class_name.replace("_", " ")


def build_macro_row(class_id: int, class_name: str) -> dict:
    """Query USDA and return a macro row for one Food-101 class."""
    query = class_to_query(class_name)
    note = MANUAL_NOTES.get(class_name, "")

    # Try FNDDS first (as-consumed), fall back to Foundation Foods
    foods = search_food(query, data_type="Survey (FNDDS)")
    data_type_used = "FNDDS"
    if not foods:
        foods = search_food(query, data_type="Foundation")
        data_type_used = "Foundation"
    if not foods:
        foods = search_food(query, data_type="SR Legacy")
        data_type_used = "SR Legacy"

    if not foods:
        print(f"  [WARN] No results for '{class_name}' — filling zeros")
        return {
            "class_id": class_id,
            "class_name": class_name,
            "kcal_per_100g": 0.0,
            "protein_g": 0.0,
            "carb_g": 0.0,
            "fat_g": 0.0,
            "fdc_id": None,
            "fdc_name": None,
            "notes": note or "NO USDA MATCH — manual review required",
        }

    best = foods[0]
    macros = get_macros_from_food(best)

    if not note:
        note = f"auto-matched via {data_type_used}"

    print(
        f"  [{class_id:03d}] {class_name} -> '{best['description']}' "
        f"(fdc_id={best['fdcId']}) | "
        f"kcal={macros['kcal_per_100g']:.0f} pro={macros['protein_g']:.1f} "
        f"carb={macros['carb_g']:.1f} fat={macros['fat_g']:.1f}"
    )
    return {
        "class_id": class_id,
        "class_name": class_name,
        **macros,
        "fdc_id": best["fdcId"],
        "fdc_name": best["description"],
        "notes": note,
    }


def main() -> None:
    """Build the macro lookup CSV for all 101 Food-101 classes."""
    if not USDA_API_KEY:
        raise EnvironmentError("USDA_API_KEY environment variable not set")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume from partial output if it exists
    existing: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        df_existing = pd.read_csv(OUTPUT_PATH)
        existing = {row["class_name"]: row.to_dict() for _, row in df_existing.iterrows()}
        print(f"Resuming: {len(existing)}/101 classes already done")

    rows = []
    for class_id, class_name in enumerate(FOOD101_CLASSES):
        if class_name in existing:
            rows.append(existing[class_name])
            continue
        print(f"Fetching {class_name}...")
        row = build_macro_row(class_id, class_name)
        rows.append(row)
        time.sleep(0.25)  # stay well under 1000 req/hour limit

    df = pd.DataFrame(rows)
    df = df[["class_id", "class_name", "kcal_per_100g", "protein_g", "carb_g", "fat_g", "fdc_id", "fdc_name", "notes"]]
    df.to_csv(OUTPUT_PATH, index=False)

    # Checkpoint validation
    nulls = df["kcal_per_100g"].isna().sum() + (df["kcal_per_100g"] == 0).sum()
    noted = (df["notes"].str.len() > 0).sum()
    print(f"\nCheckpoint: {len(df)} rows, {nulls} zero/null kcal, {noted} with notes")
    assert len(df) == 101, f"Expected 101 rows, got {len(df)}"
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
