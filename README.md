# MealLens

Upload a photo of a meal, get predicted macros (calories, protein, carbs, fat) per 100g with a portion-weight slider for per-serving totals.

## Setup

```bash
cp .env.example .env
# fill in USDA_API_KEY
make setup
```

## Run

```bash
make macros          # build macro label CSV (one-time)
make train-naive     # train naive baseline
make train-classical # train XGBoost model
make train-deep      # train EfficientNet-B0 model
make experiment      # run portion-size robustness experiment
make serve           # start FastAPI server
```

## Test

```bash
make test
make lint
```

## Project structure

```
scripts/   training + data scripts
src/       library code (data, features, models, inference, gradcam)
frontend/  single-page UI
models/    saved weights
data/      raw images (gitignored) + processed CSVs + experiment outputs
tests/     pytest suite
report/    written report and slides
```
