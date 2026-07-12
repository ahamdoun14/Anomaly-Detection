.PHONY: install test lint app train evaluate

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

app:
	streamlit run app.py

train:
	python scripts/train.py --dataset-path "$${MVTEC_DATASET_PATH}" --category bottle

evaluate:
	python scripts/evaluate.py --dataset-path "$${MVTEC_DATASET_PATH}" --category bottle
