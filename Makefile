# Project commands (DOC-03 §17.2). PYTHON defaults to the active interpreter;
# override it, e.g. `make test PYTHON=.venv/Scripts/python`.
PYTHON ?= python

.PHONY: setup lint format typecheck test validate-data split train

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

# Hash check and ingestion schema, plus the validation report (M2).
validate-data:
	$(PYTHON) -m house_price.data.validate

# Create the development/holdout split once; later runs load and verify it (M2).
split:
	$(PYTHON) -m house_price.data.split

# house-price train: cross-validate the baselines and log them to MLflow (M6).
train:
	$(PYTHON) -m house_price.cli train
