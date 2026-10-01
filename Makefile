# Project commands (DOC-03 §17.2). PYTHON defaults to the active interpreter;
# override it, e.g. `make test PYTHON=.venv/Scripts/python`.
PYTHON ?= python

.PHONY: setup lint format typecheck test validate-data split train evaluate smoke repro-check freeze

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

# DOC-05 M10: coverage of the house_price package must stay >= 80% (pyproject.toml).
test:
	$(PYTHON) -m pytest --cov=house_price --cov-report=term-missing:skip-covered --cov-fail-under=80

# Hash check and ingestion schema, plus the validation report (M2).
validate-data:
	$(PYTHON) -m house_price.data.validate

# Create the development/holdout split once; later runs load and verify it (M2).
split:
	$(PYTHON) -m house_price.data.split

# house-price train: baselines (M6), ablation with the RC-02 check and development checks
# (M7), tuning, blend and 7-candidate comparison (M8), selection and the diagnostic report
# (M9). Exit code 3 = RC-02 stop (review the ablation). Development set only.
train:
	$(PYTHON) -m house_price.cli train

# house-price evaluate: the ONE final holdout evaluation (M13 Release Run only, DOC-05
# RC-01). Refuses without a completed human diagnostic review (DN-19).
evaluate:
	$(PYTHON) -m house_price.cli evaluate

# DN-17 smoke run: train --smoke then evaluate --smoke on a development-set sample with a
# holdout substitute (never the real holdout); everything in hpp-smoke and artifacts/smoke/.
smoke:
	$(PYTHON) -m house_price.cli train --smoke
	$(PYTHON) -m house_price.cli evaluate --smoke

# AC-033: two full train runs in isolated project copies, compared; writes
# reports/reproducibility/repro_check.json. Pass CONFIG_DIR to use an execution config.
CONFIG_DIR ?= configs
repro-check:
	$(PYTHON) -m house_price.repro --config-dir $(CONFIG_DIR)

# house-price freeze: release models/staging as models/$(VERSION) after every gate QG-10 to
# QG-16 passes (DOC-03 §16.2). M13 Release Run only.
freeze:
	$(PYTHON) -m house_price.cli freeze --version $(VERSION)
