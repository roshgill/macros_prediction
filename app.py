"""Optimal — Blueprint-aligned meal analysis app.

Endpoints:
  GET  /          → serves frontend/index.html
  GET  /health    → {"status": "ok"}
  GET  /samples   → list of sample image URLs
  POST /predict   → multipart image + serving_g + body_weight_kg → macro + score JSON
  POST /gradcam   → multipart image → {"heatmap": base64 PNG}
"""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.gradcam import gradcam_to_base64
from src.inference import load_deep_model, predict
from src.llm import analyze_meal
from src.scoring import calculate_personal_targets, score_meal

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
SAMPLES_DIR = Path("frontend/samples")
FRONTEND_DIR = Path("frontend")

app = FastAPI(title="Optimal", version="1.0.0")

# Load model once at startup — not per request
_bundle = load_deep_model()

# Serve static assets (CSS, JS, sample images) under /static
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/samples")
def samples() -> dict:
    """Return URLs for the sample images shown in the UI try-row."""
    files = (
        sorted(SAMPLES_DIR.glob("*.jpg"))
        + sorted(SAMPLES_DIR.glob("*.jpeg"))
        + sorted(SAMPLES_DIR.glob("*.png"))
    )
    return {"samples": [f"/static/samples/{f.name}" for f in files]}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    serving_g: float = Form(200.0),
    body_weight_kg: float = Form(70.0),
    height_cm: float = Form(183.0),
) -> JSONResponse:
    """Accept a food image and return predicted macros + Blueprint alignment score."""
    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image — upload a valid JPEG or PNG")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file: {e}")

    try:
        result = predict(img, _bundle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Compute per-serving macros for scoring
    factor = serving_g / 100.0
    meal_macros = {
        "kcal_per_100g": result["kcal_per_100g"],
        "kcal":          result["kcal_per_100g"] * factor,
        "protein_g":     result["protein_g"]     * factor,
        "carb_g":        result["carb_g"]         * factor,
        "fat_g":         result["fat_g"]           * factor,
    }
    targets = calculate_personal_targets(height_cm, body_weight_kg)
    result["score"] = score_meal(meal_macros, targets)
    result["targets"] = targets

    return JSONResponse(content=result)


@app.post("/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a food image and return a base64 Grad-CAM heatmap overlay."""
    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file: {e}")

    try:
        heatmap = gradcam_to_base64(img, _bundle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM error: {e}")

    return JSONResponse(content={"heatmap": heatmap})


@app.post("/analyze")
async def analyze_endpoint(
    dish_name: str = Form(...),
    kcal_per_100g: float = Form(...),
    protein_g: float = Form(...),
    carb_g: float = Form(...),
    fat_g: float = Form(...),
    body_weight_kg: float = Form(70.0),
    height_cm: float = Form(183.0),
) -> JSONResponse:
    """Run GPT-4o Blueprint analysis for a predicted meal."""
    macros = {
        "kcal_per_100g": kcal_per_100g,
        "protein_g": protein_g,
        "carb_g": carb_g,
        "fat_g": fat_g,
    }
    targets = calculate_personal_targets(height_cm, body_weight_kg)
    try:
        result = analyze_meal(dish_name, macros, targets, body_weight_kg, height_cm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")
    return JSONResponse(content=result)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
