# MealLens

**Upload a photo of any dish — get predicted macros (calories, protein, carbs, fat) per 100g in under 2 seconds.**

A fine-tuned EfficientNet-B0 identifies the dish from 101 Food-101 categories and predicts nutritional content. Adjust the portion weight slider to convert per-100g values to per-serving totals. An LLM-powered Blueprint protocol analyzer then scores the meal against personalised longevity targets.

**Live demo:** [meallens on Railway](https://macros-prediction-production.up.railway.app)

---

## Model Performance (MAE per 100g)

| Model | kcal | Protein (g) | Carbs (g) | Fat (g) |
|---|---|---|---|---|
| Naive baseline | 97.97 | 5.26 | 16.04 | 7.01 |
| Classical (XGBoost) | 90.29 | 4.92 | 14.76 | 6.77 |
| **Deep (EfficientNet-B0)** | **39.55** | **2.36** | **6.43** | **3.26** |

The deep model beats classical on all four macros. Uncertainty estimated via 5-pass MC dropout.

---

## Setup

```bash
git clone https://github.com/roshgill/macros_prediction
cd macros_prediction

pip install -r requirements.txt

cp .env.example .env
# Add USDA_API_KEY and OPENAI_API_KEY to .env
```

---

## Run the App

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000
```

Or via Makefile:

```bash
make serve
```

---

## Data Pipeline

```bash
# 1. Download Food-101 images and build USDA macro label CSV
python scripts/make_dataset.py

# 2. Extract classical ML features
python scripts/build_features.py

# 3. Train all three models
python scripts/model.py
```

Or step by step:

```bash
make macros          # USDA macro lookup → data/processed/food101_macros.csv
make train-naive     # global-mean baseline → models/naive.json
make train-classical # XGBoost on hand-crafted features → models/classical.pkl
make train-deep      # EfficientNet-B0 fine-tune → models/deep.pt
make experiment      # portion-size robustness experiment → data/outputs/
```

---

## Project Structure

```
├── README.md
├── requirements.txt
├── Makefile
├── setup.py                  ← installs the meallens package
├── app.py                    ← FastAPI app (served on Railway)
├── scripts/
│   ├── make_dataset.py       ← download Food-101 + build USDA macro CSV
│   ├── build_features.py     ← extract classical ML features
│   ├── model.py              ← train all three models
│   ├── build_macro_lookup.py ← raw USDA API query script
│   ├── train_naive.py
│   ├── train_classical.py
│   ├── train_deep.py
│   └── run_experiment.py
├── src/
│   ├── data.py               ← HuggingFace dataset loader + macro join
│   ├── features.py           ← RGB/HSV histograms + LBP texture features
│   ├── models.py             ← MealLensModel (EfficientNet dual head)
│   ├── inference.py          ← predict() with MC dropout uncertainty
│   ├── gradcam.py            ← Grad-CAM heatmap generation
│   ├── scoring.py            ← BSA-scaled per-meal macro targets
│   └── llm.py                ← GPT-4o Blueprint protocol analyzer
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
│   ├── deep.pt               ← EfficientNet-B0 weights (17MB)
│   ├── classical.pkl         ← XGBoost multi-output regressor
│   ├── naive.json            ← global mean macros
│   └── macro_stats.json      ← normalisation mean/std
├── data/
│   ├── raw/                  ← Food-101 images (gitignored)
│   ├── processed/
│   │   └── food101_macros.csv ← 101-class USDA macro labels
│   └── outputs/
│       ├── experiment_results.csv
│       └── gradcam/          ← Grad-CAM sample outputs
├── notebooks/
│   └── 01_eda.ipynb          ← macro distribution + feature exploration
└── tests/
    └── test_data.py
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/health` | GET | Health check |
| `/samples` | GET | Sample image URLs for the try-row |
| `/predict` | POST | Image → macros + dish classification |
| `/gradcam` | POST | Image → Grad-CAM heatmap (base64 PNG) |
| `/analyze` | POST | Macros + user metrics → Blueprint score + LLM advice |

---

## Data Sources

- **Images:** [Food-101](https://huggingface.co/datasets/ethz/food101) — 101,000 images, 101 classes (Bossard et al., 2014)
- **Macro labels:** [USDA FoodData Central](https://fdc.nal.usda.gov/) — FNDDS "as-consumed" entries (CC0)

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `USDA_API_KEY` | Training only | USDA FoodData Central API key (free) |
| `OPENAI_API_KEY` | Runtime | GPT-4o for Blueprint analysis |
| `DATABASE_URL` | Runtime | Neon PostgreSQL + pgvector for RAG |
