# Project commands (DOC-03 §17.2). PYTHON defaults to the active interpreter;
# override it, e.g. `make test PYTHON=.venv/Scripts/python`.
PYTHON ?= python

.PHONY: setup lint format typecheck test validate-data split train evaluate smoke repro-check freeze serve predict docker-build docker-test docker-push

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

# Serving (M11, DOC-04 §18.3). MODEL_DIR is the artifact directory: models/$(VERSION) for a
# frozen release; for the smoke artifact (never released) also pass
# HPP_ALLOW_NON_RELEASE=true, e.g.
#   make serve MODEL_DIR=artifacts/smoke/model HPP_ALLOW_NON_RELEASE=true
MODEL_DIR ?= models/$(VERSION)
PORT ?= 8000
HPP_ALLOW_NON_RELEASE ?= false

# Uvicorn, one worker, structured logs only (SD-09); the lifespan verifies the artifact
# before the service accepts requests (DOC-04 §5).
serve:
	HPP_MODEL_DIR=$(MODEL_DIR) HPP_CONFIG_DIR=$(CONFIG_DIR) HPP_ALLOW_NON_RELEASE=$(HPP_ALLOW_NON_RELEASE) PORT=$(PORT) $(PYTHON) -m uvicorn house_price.api.app:app --host 127.0.0.1 --port $(PORT) --workers 1 --no-access-log

# Batch CLI (DOC-04 §10): make predict INPUT=properties.csv OUTPUT=predictions.csv
# Exit codes 0 success, 2 invalid input (no output written), 3 runtime failure; at most
# 20 rows per file (MAX_BATCH_SIZE).
predict:
	HPP_ALLOW_NON_RELEASE=$(HPP_ALLOW_NON_RELEASE) $(PYTHON) -m house_price.cli predict --input $(INPUT) --output $(OUTPUT) --model-dir $(MODEL_DIR) --config-dir $(CONFIG_DIR)

# Docker (M12, DOC-04 §11, §16.5, §17). Without VERSION the smoke artifact
# artifacts/smoke/model is baked in (CI and local container tests only; never pushed);
# with VERSION=x.y.z the frozen release models/x.y.z (M13). DOCKER_MODEL_DIR overrides.
DOCKER_ARGS = $(if $(VERSION),--version $(VERSION)) $(if $(DOCKER_MODEL_DIR),--model-dir $(DOCKER_MODEL_DIR))

# Two-stage build; fails unless the OCI version label equals metadata.model_version.
docker-build:
	$(PYTHON) -m house_price.deploy build $(DOCKER_ARGS)

# Container test (DOC-04 §16.5 steps 2-6): default port and PORT=10000, /health, UID != 0,
# example price == direct prediction, wrong HPP_MODEL_DIR fails startup, image contents.
docker-test: docker-build
	$(PYTHON) -m house_price.deploy test $(DOCKER_ARGS)

# Release push to ghcr.io/<owner>/house-price-api:$(VERSION) (M13 only). Refuses smoke,
# non-release and candidate artifacts, an existing tag, and CI. Needs docker login ghcr.io.
docker-push:
	$(PYTHON) -m house_price.deploy push --version $(VERSION)
