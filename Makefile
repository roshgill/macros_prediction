setup:
	pip install -r requirements.txt

macros:
	python scripts/build_macro_lookup.py

train-naive:
	python scripts/train_naive.py

train-classical:
	python scripts/train_classical.py

train-deep:
	python scripts/train_deep.py

experiment:
	python scripts/run_experiment.py

serve:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ --cov=src --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
