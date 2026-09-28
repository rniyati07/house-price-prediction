# DOC-05: Implementation Roadmap and Acceptance Criteria

**Project:** House Price Prediction — End-to-End ML Regression System
**Document ID:** DOC-05
**Version:** 1.1
**Status:** Approved baseline — revision 1.1 (dataset alignment: the project uses the full 2,930-row Ames file; see `docs/data_card.md`)
**Date:** 2026-09-28
**Authoritative sources, in precedence order:** `house-price-prediction-adr.md` → DOC-01 v1.2 → DOC-02 v1.2 → DOC-03 v1.1 → DOC-04 v1.1

---

# 1. Purpose and Scope

## 1.1 Why This Roadmap Exists

DOC-01 to DOC-04 describe **what** the system is and **how** each part works. They do not say in which order to build it. Building a system like this in the wrong order causes real problems: code gets rewritten because a later component needed something earlier code did not provide, or the holdout set gets used before the system is ready and can never be used honestly again.

This document turns the approved designs into an ordered plan of 13 milestones. For each milestone it states what to build, what to test, what to commit, and what evidence proves the milestone is done.

## 1.2 Relationship to the Other Documents

| Document | Role in this roadmap |
|---|---|
| ADR (ADR-01 to ADR-20) | The decisions every milestone implements. The roadmap changes none of them |
| DOC-01 | The requirements (FR, NFR) and acceptance criteria (AC) that milestones deliver and verify |
| DOC-02 | The EDA questions and deliverables (E-01 to E-36) built in M3, M4, and M7 |
| DOC-03 | The training-system design built in M1 to M10 |
| DOC-04 | The serving and deployment design built in M11 to M13 |

Where this roadmap refers to a component, file, or command, it uses the exact name from DOC-03 or DOC-04. If the roadmap and a design document ever disagree, the design document wins and the roadmap is corrected.

## 1.3 Roadmap Clarifications

Putting the designs in a real order exposed four places where the documents need a precise execution rule. These rules do not change any design; they state how existing requirements are satisfied in sequence.

| RC | Topic | Clarification |
|---|---|---|
| RC-01 | When the real holdout is used | The real final holdout evaluation happens **once**, in the Release Run (M13), on the final commit. M9 and M10 build and test the evaluation and artifact machinery using smoke mode (DOC-03 DN-17), which never touches the real holdout. See Section 2.5 for the reasoning |
| RC-02 | First ablation run | On the first `make train`, `features.yaml` has no ablation outcome yet, so `train` stops after ablation and asks for the outcome to be reviewed and committed (DOC-03 §6.6 step 5). On later runs, `train` recomputes the ablation and stops if the result differs from the committed outcome |
| RC-03 | "Clean tree" during release | `evaluate` writes new report files into `reports/evaluation/`. The clean-tree check in `freeze` treats changes inside `reports/evaluation/` as expected output of `evaluate`; any other change fails the check. These reports are committed immediately after `freeze` |
| RC-04 | Smoke training in CI | The dataset is not committed, so CI cannot sample the real development set as DOC-03 DN-17 describes. In CI, smoke mode runs on a committed synthetic fixture that follows `schema.yaml`; locally, `make smoke` uses the real development-set sample. Both exercise every stage (Section 5.5) |

## 1.4 Intended Usage

- Work through the milestones in order. Each depends on the ones before it.
- At the start of a milestone, read its section and the design sections it cites.
- At the end of a milestone, go through its acceptance table (Section 7) and collect the evidence before moving on.
- Use Section 5 to see how the test suite grows, Section 6 for commit and tag conventions, and Section 10 before the release.

## 1.5 Out of Scope

This document does not add requirements, tools, models, or design decisions. It contains no traceability matrix; traceability lives in DOC-01 §10, DOC-03 §18, and DOC-04 §19.

---

# 2. Implementation Philosophy

## 2.1 Incremental Development

The system is built bottom-up, in the same order that data flows through it:

```
foundation → data → understanding (EDA) → features → preprocessing → baselines
  → candidates → tuning → selection → artifact → API → container → release
```

Each layer uses only layers already built and tested. When you build the preprocessing pipeline (M5), the feature engineer (M4) is already correct, so any bug you see is in the new code.

## 2.2 Validation Before Progression

A milestone is not finished when the code runs; it is finished when its acceptance table (Section 7) is satisfied and the evidence is committed. Moving on with a known failure is how small bugs become expensive ones: a wrong imputation discovered in M9 means redoing M5 to M9.

## 2.3 Reproducibility Principles

From the first milestone:

- **One environment:** every command runs through `uv` with the locked dependencies (NFR-002).
- **One seed:** the global seed 42 from `configs/validation.yaml` (DOC-03 DN-02).
- **No hidden constants:** thresholds, sizes, and budgets live in configuration (NFR-020).
- **Every run is recorded:** from M6 on, every model evaluation is logged to MLflow with its commit, data hash, and config hash (FR-036).
- **Nothing generated by hand:** splits, folds, reports, and artifacts are produced by commands, never edited manually.

## 2.4 Quality-First Implementation

- Tests are written in the same milestone as the code they test, not at the end.
- Ruff, mypy, and the test suite run on every commit (pre-commit) and every push (CI) from M1.
- Test coverage is measured from M1 and enforced at 80% from M10 (Section 5.3).

## 2.5 Protecting the Holdout (RC-01)

The holdout set may be evaluated **exactly once** for the selected model [ADR-09, AC-038]. That single evaluation must produce the released artifact's reported metrics, and the released artifact must be produced by the final code: its `git_commit` must be the release tag's commit (DOC-01 §11.9), and it is unpickled by the serving code in the same image (DOC-04 §11).

If the real holdout were evaluated in M9, the artifact would come from a commit that does not yet contain the API. Releasing it later would mean either a mismatched commit, or evaluating the holdout a second time. Neither is acceptable.

The roadmap therefore separates **building** from **executing**:

| Activity | Where | Touches real holdout? |
|---|---|---|
| Build and test selection, holdout evaluation, gates, temporal diagnostic, refit, artifact, freeze | M9, M10 (smoke mode and unit tests) | No |
| Real `make train`, including selection and the diagnostic review | M9 (rehearsal) and M13 (release) | No; training uses the development set only |
| Real `make evaluate` and `make freeze` | M13 Release Run only | **Yes, once** |

Because training is deterministic (NFR-001), the M13 selection is expected to match M9. If it does not, that is a reproducibility failure that must be fixed before the holdout is touched.

## 2.6 Learning Objectives

Each milestone lists its learning goals. Across the project you will practice, in this order:

1. Project structure, packaging, configuration, and automated quality checks
2. Treating data as a contract
3. Reading a dataset before modeling it
4. Stateless transformers and the scikit-learn estimator API
5. Leakage-safe preprocessing with `Pipeline` and `ColumnTransformer`
6. Honest validation, baselines, and experiment tracking
7. Model families, regularization, and ablation
8. Bayesian hyperparameter optimization
9. Decision-making under uncertainty (the one-standard-error rule)
10. Artifacts with provenance
11. API contracts and training–serving consistency
12. Containers and cloud deployment
13. Release discipline

## 2.7 Milestone-Based Execution

| Milestone | Theme | Main design sections | Rough effort |
|---|---|---|---|
| M1 | Project foundation | DOC-03 §5.1, §17 | 1–2 days |
| M2 | Data ingestion and validation | DOC-03 §5 | 2–3 days |
| M3 | Exploratory data analysis | DOC-02 §4 to §13 | 3–4 days |
| M4 | Data preparation and feature engineering | DOC-03 §6 | 2 days |
| M5 | Preprocessing architecture | DOC-03 §7 | 2–3 days |
| M6 | Baseline modeling and MLflow | DOC-03 §8.2, §8.3, §9, §14 | 2 days |
| M7 | Candidate models and ablation | DOC-03 §6.5, §6.6, §8 | 2–3 days |
| M8 | Hyperparameter tuning | DOC-03 §10 | 2–3 days |
| M9 | Model selection and evaluation machinery | DOC-03 §11 to §13 | 3 days |
| M10 | Final artifact machinery | DOC-03 §15, §16 | 2 days |
| M11 | FastAPI serving layer and batch CLI | DOC-04 §5 to §10 | 3–4 days |
| M12 | Dockerization and deployment preparation | DOC-04 §11, §12 | 2 days |
| M13 | Testing, documentation, and release | DOC-04 §16, §17 | 3 days |

Effort is indicative for a student working part-time; it is not a deadline.

---

# 3. Repository Initialization

This section is the step-by-step setup that M1 wraps into a milestone. Do it once, carefully: everything later depends on it.

## 3.1 Repository Setup

1. Create a GitHub repository named `house-price-prediction` with a `main` branch.
2. Clone it locally.
3. Add a `.gitignore` covering: `data/raw/`, `data/processed/*.csv`, `artifacts/`, `mlruns/`, `models/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.coverage`, `htmlcov/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`.
4. Add a license file of your choice for the code, and note in the README that the dataset (the Ames Housing data published by De Cock, 2011) is not redistributed and that users obtain it themselves under its published terms (see the data card).

## 3.2 Project Structure Creation

Create the ADR-17 layout, with the module files named in DOC-03 §17.1 and DOC-04 §18.1 (empty modules with a one-line docstring are fine for now):

```
house-price-prediction/
├── configs/                      # YAML configuration (filled milestone by milestone)
├── data/
│   ├── raw/                      # train.csv goes here (gitignored)
│   └── processed/                # split files (CSVs gitignored; manifest tracked)
├── notebooks/                    # EDA notebooks (M3)
├── src/house_price/
│   ├── __init__.py
│   ├── config.py
│   ├── tracking.py
│   ├── cli.py
│   ├── data/        (__init__, errors, load, schema, scope, split, profile, validate)
│   ├── features/    (__init__, semantic, engineer)
│   ├── pipelines/   (__init__, branches, build)
│   ├── models/      (__init__, registry, ablation, tuning, selection, train)
│   ├── evaluation/  (__init__, metrics, cv, diagnostics, holdout, temporal, gates)
│   ├── persistence/ (__init__, artifact, metadata)
│   └── api/         (__init__, app, schemas, predict, settings, logging, errors)
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── quality/
│   └── fixtures/
├── reports/
│   ├── figures/
│   ├── eda/
│   ├── data_validation/
│   ├── selection/
│   └── evaluation/
├── docs/
│   ├── adr/
│   └── (DOC-01 … DOC-05)
├── .github/workflows/
├── Makefile
├── pyproject.toml
└── README.md
```

## 3.3 Environment Setup

1. Install `uv`.
2. Pin Python 3.12 for the project (`uv python pin 3.12`).
3. Create the environment with `uv sync`.

## 3.4 Dependency Management

All dependencies go in `pyproject.toml`; `uv.lock` is committed and is the only source of installed versions [ADR-16].

| Group | Packages | Why each is needed |
|---|---|---|
| Runtime | pandas, numpy, scipy, scikit-learn, lightgbm, joblib, pyyaml, pydantic (v2), pandera, fastapi, uvicorn | The approved stack [ADR-08, ADR-10, ADR-14, ADR-15, ADR-16] |
| Training | optuna, mlflow, matplotlib | Tuning [ADR-13]; tracking [ADR-16]; the figures required by DOC-02 §13 and DOC-03 §11.6 |
| Development | pytest, pytest-cov, ruff, mypy, pre-commit, httpx, jupyter, types-PyYAML | Testing and quality tools [ADR-16, ADR-18]; `httpx` is required by FastAPI's `TestClient`; `jupyter` runs the EDA notebooks [ADR-17]; `types-PyYAML` gives mypy type information |

`matplotlib`, `httpx`, `jupyter`, and `types-PyYAML` are supporting libraries needed to carry out already-approved work (figures, API tests, notebooks, type checking); they introduce no new architecture.

The Docker image installs only the runtime group (DOC-04 §11.2: `uv sync --frozen --no-dev`). Define training and development groups so that the image stays small.

## 3.5 Configuration Initialization

Create the configuration files with the content known at this point. Each is completed by a later milestone:

| File | Created in M1 with | Completed in |
|---|---|---|
| `configs/data.yaml` | Raw path, processed paths, missing tokens `["NA", ""]` (DN-18), scope column and threshold (`GrLivArea`, 4000); hash and expected in-scope row count (2,925) left as required fields to be filled | M2 |
| `configs/validation.yaml` | Seed 42, holdout fraction 0.2, 10 bins, 5 folds, 3 repeats, tolerance 1e-6, split-balance tolerance 2 pp (IN-07) | M1 (complete) |
| `configs/schema.yaml` | Structure only | M2 |
| `configs/features.yaml` | Structure only | M4, M5, M7 |
| `configs/models.yaml` | Structure only | M6, M7, M8 |

`src/house_price/config.py` defines one Pydantic model per file (with `extra="forbid"`), a loader, and the `config_hash` function (DOC-03 §5.1).

## 3.6 Tooling Initialization

| Tool | Setup |
|---|---|
| Ruff | Lint and format rules in `pyproject.toml` |
| mypy | Strict-enough settings for `src/` (disallow untyped definitions in `src/`; tests may be looser) |
| pytest | `testpaths = ["tests"]`; coverage configured for `src/house_price` |
| pre-commit | Hooks: Ruff (lint and format), mypy on `src/`, trailing whitespace, end-of-file fixer, YAML check, large-file check |
| Makefile | `setup`, `lint`, `typecheck`, `test` targets now; later targets added by their milestones |
| GitHub Actions | `ci.yml` with jobs: lint → typecheck → test (smoke-train and Docker stages are added in M9 and M12) |
| CLI | `pyproject.toml` script entry `house-price = "house_price.cli:main"`, with `argparse` subcommands added milestone by milestone |

## 3.7 Expected Outputs

- A repository that installs with one command (`make setup`) on a clean machine
- An empty but importable `house_price` package
- CI running green on `main`

## 3.8 Acceptance Criteria

| ID | Criterion |
|---|---|
| RI-1 | `make setup` on a fresh clone succeeds and installs hooks |
| RI-2 | `uv run python -c "import house_price"` succeeds |
| RI-3 | `make lint`, `make typecheck`, and `make test` pass |
| RI-4 | CI is green on the first push |
| RI-5 | `uv.lock` is committed; Python is pinned to 3.12 |

---

# 4. Milestone Roadmap

Each milestone below follows the same template. Its full acceptance table, with verification methods, evidence, and failure conditions, is in Section 7.

---

## M1 – Project Foundation

**Purpose.** Create a working, tested, automatically checked project skeleton so that every later milestone starts from a known-good base.

**Learning goals.**
- Python packaging with a `src/` layout and why it prevents accidental imports [ADR-17]
- Reproducible environments with a lockfile [ADR-16]
- Configuration as validated data (YAML + Pydantic)
- Automated quality gates: pre-commit and CI

**Dependencies.** None.

**Tasks.**
1. Complete Section 3 (repository initialization).
2. Implement `config.py`: Pydantic models for `data.yaml` and `validation.yaml`, a loader that fails clearly on unknown keys, missing keys, or wrong types (NFR-019), and `config_hash`.
3. Add the `house-price` CLI entry with an empty `argparse` parser that prints help.
4. Convert the ADR into MADR files: `docs/adr/0000-foundational-decisions.md` and one file per ADR-01 to ADR-20, each with status, context, decision, and consequences (FR-052). The content already exists; this is formatting.
5. Copy DOC-01 to DOC-05 into `docs/`.
6. Write a README skeleton: title, one-paragraph description, setup instructions (the rest arrives in M13).
7. Write the first tests.

**Files created.**
`pyproject.toml`, `uv.lock`, `.gitignore`, `.pre-commit-config.yaml`, `Makefile`, `.github/workflows/ci.yml`, `README.md`, `configs/*.yaml`, `src/house_price/**` (skeleton), `src/house_price/config.py`, `src/house_price/cli.py`, `docs/adr/*.md`, `docs/DOC-0*.md`, `tests/unit/test_config.py`, `tests/unit/test_package.py`.

**Files modified.** None (first milestone).

**Tests.**
- `test_package.py`: the package imports; the CLI prints help and exits 0.
- `test_config.py`: valid configs load; an unknown key, a missing key, and a wrong type each raise a clear error; `config_hash` is stable for identical content and changes when content changes.

**Expected outputs.** A green CI run; `make setup`, `make lint`, `make typecheck`, `make test` all working.

**Acceptance criteria.** M1-1 to M1-6 (Section 7.1).

**Checkpoint commit message.** `chore(repo): initialize project structure, environment, and tooling`
Milestone commit: `chore(milestone): complete M1 project foundation`; tag `m1-complete`.

**Recommended verification.** Clone the repository into a new directory and run `make setup && make lint && make typecheck && make test`. Everything must pass without manual steps.

---

## M2 – Data Ingestion and Validation

**Purpose.** Load the raw dataset exactly as specified, prove its integrity, enforce the data contract, apply the scope rule, and create the one permanent development/holdout split.

**Learning goals.**
- Data as a contract: schemas, allowed values, ranges [ADR-04]
- Why parsing rules matter (the `MasVnrType` "None" hazard, DOC-02 §6.5)
- Scope rules versus cleaning (DOC-02 §9.4)
- Stratified splitting and why the holdout must be locked [ADR-09]

**Dependencies.** M1.

**Tasks.**
1. Place the full Ames Housing file (De Cock, 2,930 rows × 82 columns) at `data/raw/train.csv`. Compute its SHA-256 and write it into `configs/data.yaml`, together with the verified in-scope row count (2,925).
2. Write `configs/schema.yaml` for all 82 columns: canonical name and raw header (`source_name`), dtype, nullable, allowed values, range, and role (`model_input`, `excluded`, `identifier`, `target`), following the range classes in DOC-03 §5.3. `Id` (raw header `Order`) and `PID` are identifiers. Nullable columns are the 27 that contain missing values in the file (DOC-04 §6.4). While writing allowed values, compare the data dictionary codes with the values that actually occur in the file and include the file's spellings where they differ (DOC-03 §5.3).
3. Implement `data/load.py`: `load_raw()` with file-existence and hash verification, explicit missing tokens (DN-18), text-first reading, header normalization to canonical names, and casting (FR-001 to FR-003).
4. Implement `data/schema.py`: build the Pandera ingestion schema from `schema.yaml`, lazy validation, and a builder for the model-input subset that M11 will reuse (FR-004, FR-005).
5. Implement `data/scope.py`: `apply_scope_rule()` with the configured 2,925-row assertion and the removal record (FR-006).
6. Implement `data/split.py`: `create_or_load_split()` with 10 decile bins, seed 42, persisted CSVs, `split_manifest.json`, the disjointness check, and the 2-point balance check (FR-007).
7. Add a `__main__` block to `data/split.py` so the split can be created with `python -m house_price.data.split` (`make split`). Add `data/validate.py` and `make validate-data`: hash check and ingestion schema, plus the automated validation report and initial profiling (missing values, column profile, duplicate detection, data-quality flags, dataset metadata) written to `reports/data_validation/`. Profiling only reports; it never changes data (DOC-03 §5.4).
8. Run the split **once**. Commit `data/processed/split_manifest.json`.
9. Draft `docs/data_card.md`: source, license, the scope rule, missing-value semantics, and the documented known issues (parsing hazard, the basement and garage anomalies, the remodel-date floor, dictionary-versus-file spellings), each marked "documented — to be verified in M3" (FR-053).

**Files created.**
`configs/schema.yaml` (complete), `src/house_price/data/{errors,load,schema,scope,split,profile,validate}.py`, `data/processed/split_manifest.json`, `reports/data_validation/*` (validation report and profiling tables), `docs/data_card.md`, `tests/conftest.py`, `tests/unit/test_load.py`, `tests/unit/test_schema.py`, `tests/unit/test_scope_split.py`, `tests/unit/test_profile.py`, `tests/unit/test_validate.py`, `tests/fixtures/raw_sample.csv` (a small sample of raw-format rows for fast tests).

**Files modified.** `configs/data.yaml` (hash, expected in-scope count), `config.py` (schema model), `tests/unit/test_config.py` (schema model), `Makefile` (`validate-data`, `split`), `pyproject.toml` (`data` test marker).

**Tests.**
- Raw file unchanged after loading (AC-001).
- Modified copy of the file → error, non-zero exit, nothing written (AC-002).
- Per-column missing counts stable; `MasVnrType` counts only `NA` tokens (AC-003).
- Five schema rejection cases (AC-004).
- Ingestion and inference schemas read the same allowed values (AC-005).
- 2,925 rows after the scope rule; the 5 removed `Id`s recorded (AC-006).
- Split sizes, disjointness, persistence, and balance (AC-007 to AC-009).
- File-existence checks, hash generation, duplicate detection, profiling, and validation-report generation (including nothing written after a hash mismatch).

Tests that need the real file are marked `@pytest.mark.data` and skipped when `data/raw/train.csv` is absent (as in CI, which never has the dataset). Fixture-based tests always run.

**Expected outputs.** A validated dataset; a persisted split of 2,340/585 rows; a committed split manifest; a validation report in `reports/data_validation/`; a draft data card.

**Acceptance criteria.** M2-1 to M2-8 (Section 7.2).

**Checkpoint commit messages.**
- `feat(data): add raw loader with hash verification and explicit NA parsing`
- `feat(data): add schema config and Pandera ingestion validation`
- `feat(data): add scope rule and persisted stratified split`
- `docs(data): add draft data card`
Milestone commit: `chore(milestone): complete M2 data ingestion and validation`; tag `m2-complete`.

**Recommended verification.** Run `make validate-data`, then `make split` (`python -m house_price.data.split`) twice: the second run must load, not regenerate. Open `split_manifest.json` and confirm the counts. Temporarily edit one byte of a copy of the CSV, point the config at it, and confirm training-side loading refuses it.

---

## M3 – Exploratory Data Analysis

**Purpose.** Understand the dataset before modeling it, and turn DOC-02's documented properties into project EDA findings with evidence.

**Learning goals.**
- Asking questions before plotting (DOC-02 §4)
- Target skew and why log space helps [ADR-02]
- Absent versus unknown missingness [ADR-05]
- Correlation, collinearity, cardinality, rare categories
- Protecting the holdout during analysis (FR-009)

**Dependencies.** M2 (the validated dataset and the split).

**Tasks.**
1. Create the EDA notebooks. Every notebook imports loaders from `house_price.data`; no function used by training or serving is defined in a notebook (FR-010).

| Notebook | Deliverables (DOC-02 §13) | Data used |
|---|---|---|
| `01_data_integrity.ipynb` | E-01 to E-04 | Full raw file (no target relationships) |
| `02_target.ipynb` | E-05 to E-07 (development set); E-08 (split balance, the permitted exception) | Development set |
| `03_missing_values.ipynb` | E-09 to E-12 | E-09 to E-11 raw (target-free); E-12 development set |
| `04_numeric.ipynb` | E-13 to E-18 | Development set |
| `05_categorical.ipynb` | E-19 to E-23 | Development set |
| `06_outliers_scope.ipynb` | E-24 to E-26 | Full raw file (the ADR-06 scope-review exception) |
| `08_leakage_time.ipynb` | E-31 to E-34 | Development set |
| `99_eda_report.ipynb` | E-35 (draft), E-36 | Reads the saved outputs above |

   Notebook `07` is reserved for M4 (engineered features need `FeatureEngineer`).
2. Save figures to `reports/figures/eda/` and tables to `reports/eda/` with names that start with the deliverable ID (for example `E-05_target_statistics.csv`).
3. In `99_eda_report.ipynb`, create the **confirmation register** (E-35): for every documented property in DOC-02, record "confirmed" with the observed value or "not confirmed" with the discrepancy. Answer Q1–Q16, Q18, and Q19. Q17 stays open until M4 and M7.
4. Handle discrepancies: any "not confirmed" item is written into the data card's known issues. If it affects an ADR-fixed number (for example, the scope rule does not remove exactly 5 rows), stop and follow the ADR supersession process before continuing (DOC-02 §9.2).
5. Update the data card with confirmed known issues, including the basement and garage anomalies identified by `Id` (E-11).

**Files created.** `notebooks/01…06, 08, 99`, `reports/figures/eda/*`, `reports/eda/*`.

**Files modified.** `docs/data_card.md`.

**Tests.** No new unit tests (notebooks contain no production logic). A review-style check replaces them: search notebook sources for the holdout path (AC-011) and for function or class definitions used by the package (AC-012).

**Expected outputs.** All deliverables E-01 to E-26 and E-31 to E-34 present; a draft E-35 with every answer except Q17; E-36 fed into the data card.

**Acceptance criteria.** M3-1 to M3-6 (Section 7.3).

**Checkpoint commit messages.**
- `feat(eda): add data integrity and target analysis notebooks`
- `feat(eda): add missing-value, numeric, and categorical analyses`
- `feat(eda): add outlier, leakage, and time analyses`
- `docs(eda): add EDA report with confirmation register`
Milestone commit: `chore(milestone): complete M3 exploratory data analysis`; tag `m3-complete`.

**Recommended verification.** Restart each notebook's kernel and run all cells top to bottom; outputs must regenerate without error. Search: `grep -l "holdout" notebooks/*.ipynb` must list only notebooks that use the permitted exceptions, and only for those purposes.

---

## M4 – Data Preparation and Feature Engineering

**Purpose.** Implement the two stateless transformers that encode the project's missing-value semantics and engineered features, and validate the features against the EDA hypotheses.

**Learning goals.**
- The scikit-learn transformer API (`fit`, `transform`, `get_feature_names_out`)
- Why stateless steps cannot leak [ADR-05]
- Turning domain intuition into tested formulas [ADR-07]

**Dependencies.** M2 (schema and data), M3 (confirmed missingness patterns and feature hypotheses).

**Tasks.**
1. **Cleaning check.** Confirm that the only cleaning is casting in `load.py` and the scope rule (DOC-03 §5.4). No value repair is added anywhere.
2. Fill `configs/features.yaml`: semantic-filler column lists (DOC-03 §6.2), the ordinal map and its 10 columns (DN-13), the list of 12 engineered features, and the initial column groups for both branches (numeric, ordinal, nominal, dropped = `GarageYrBlt`).
3. Implement `features/semantic.py`: `SemanticNAFiller` (FR-011).
4. Implement `features/engineer.py`: `FeatureEngineer` with the 12 formulas, the ordinal map, the `MSSubClass` cast, and the defensive behaviors (null `GarageYrBlt` with a garage → `GarageAge` missing; negative ages passed through) (FR-012, FR-013).
5. Add notebook `07_feature_engineering.ipynb` for E-27 to E-29, using `FeatureEngineer` from the package on the development set.
6. Update E-35 with Q17 evidence from E-27 and E-28 (the ablation part of Q17 is added in M7).

**Files created.** `src/house_price/features/{semantic,engineer}.py`, `tests/unit/test_semantic.py`, `tests/unit/test_engineer.py`, `tests/fixtures/feature_rows.csv` (hand-built rows with known expected outputs), `notebooks/07_feature_engineering.ipynb`.

**Files modified.** `configs/features.yaml`, `config.py` (features model), `notebooks/99_eda_report.ipynb`.

**Tests.**
- Every "feature absent" case filled correctly; no missing values remain in filled columns (AC-014).
- Input frame not mutated; output identical regardless of what `fit` saw; no fitted state (AC-015).
- Each of the 12 formulas matches a hand-computed value, including the no-garage case (AC-016).
- Ordinal map values and the `MSSubClass` cast (AC-017).
- The documented basement and garage anomaly patterns receive `"None"` (and `0` for their numerics) (DOC-03 §6.2).
- `clone()` works on both transformers.

**Expected outputs.** Two tested transformers; E-27 to E-29 produced.

**Acceptance criteria.** M4-1 to M4-6 (Section 7.4).

**Checkpoint commit messages.**
- `feat(features): add stateless semantic NA filler`
- `feat(features): add feature engineer with approved features and ordinal map`
- `feat(eda): add engineered feature assessment (E-27 to E-29)`
Milestone commit: `chore(milestone): complete M4 data preparation and feature engineering`; tag `m4-complete`.

**Recommended verification.** In a scratch session, pass one development row with no garage through both transformers and read every output column by eye against the formulas.

---

## M5 – Preprocessing Architecture

**Purpose.** Assemble the full raw-to-dollars pipeline for both model families, with fitted imputation, encoding, scaling, and the log-target wrapper, and prove it cannot leak.

**Learning goals.**
- `Pipeline`, `ColumnTransformer`, and `TransformedTargetRegressor`
- Why preprocessing differs for linear and tree models [ADR-08]
- Unseen categories and feature order
- Writing a leakage test

**Dependencies.** M4.

**Tasks.**
1. Implement `pipelines/branches.py`: `build_column_transformer(branch, feature_config)` for the linear and tree branches exactly as DOC-03 §7.2 to §7.5 specify, with `set_output(transform="pandas")`.
2. Implement `pipelines/build.py`: `build_pipeline(candidate, config)` producing `TransformedTargetRegressor(log1p/expm1)` around `SemanticNAFiller → FeatureEngineer → ColumnTransformer → estimator`. For M5, test it with a simple `Ridge` and a simple `LightGBM` estimator; the full candidate registry arrives in M6 to M8.
3. Implement model-input selection: a function that takes a frame and returns exactly the 77 model-input columns in schema order (DN-11). Every fit and predict call goes through it.
4. Add the group-coverage check: the three column lists plus the dropped group equal the `FeatureEngineer` output columns (DOC-03 §7.5).
5. Lay the training–serving consistency foundations: a test that saves a fitted pipeline with `joblib` to a temporary file, reloads it, and compares predictions exactly.

**Files created.** `src/house_price/pipelines/{branches,build}.py`, `tests/unit/test_pipeline.py`, `tests/unit/test_leakage.py`.

**Files modified.** `configs/features.yaml` (branch groups), `src/house_price/data/schema.py` (model-input column order helper, if not added in M2).

**Tests.**
- Excluded columns (`SaleType`, `SaleCondition`, `Id`, `PID`) are not pipeline inputs; rows are kept (AC-018).
- No target encoder, polynomial features, PCA, or automated feature generator (AC-020).
- `predict` on raw rows returns finite, positive dollars (AC-021).
- Linear output is scaled; tree output is not; unseen category → −1 in the tree branch (AC-022).
- Allowed-but-unseen category is scored by both branches (AC-023).
- Missing-indicator column present; imputed value equals the fitting data's median (AC-024).
- **Leakage test:** fitted statistics unchanged by transforming other data, and equal to those from fitting on the fitting data alone (AC-025).
- `clone()` works on the full pipeline (AC-026).
- Moving a column between groups in YAML changes its treatment (AC-027).
- Joblib round trip is exact.
- Group-coverage check passes; reordering request columns does not change predictions.

**Expected outputs.** A single pipeline builder that every later milestone uses.

**Acceptance criteria.** M5-1 to M5-6 (Section 7.5).

**Checkpoint commit messages.**
- `feat(pipeline): add linear and tree column transformers`
- `feat(pipeline): add end-to-end pipeline builder with log-target wrapper`
- `test(pipeline): add leakage, unseen-category, and round-trip tests`
Milestone commit: `chore(milestone): complete M5 preprocessing architecture`; tag `m5-complete`.

**Recommended verification.** Fit the Ridge test pipeline on the development set and print `get_feature_names_out()` of the column transformer. Read through the names and check that every expected group appears and `GarageYrBlt` does not.

---

## M6 – Baseline Modeling

**Purpose.** Build the cross-validation machinery, the metrics, and MLflow tracking, and use them to measure the two baselines that define "good enough".

**Learning goals.**
- Why baselines come first [ADR-10]
- Repeated stratified CV and shared folds (DN-03)
- Mean ± standard error instead of a single number (DN-04)
- Experiment tracking with MLflow [ADR-16]

**Dependencies.** M5.

**Tasks.**
1. Implement `evaluation/metrics.py`: log-RMSE, MAE, MAPE, R² (DOC-03 §12.1).
2. Implement `evaluation/cv.py`: fold generation on development-set deciles with `RepeatedStratifiedKFold(5, 3, seed)`, saved to `artifacts/cv/folds.json` keyed by `Id`; the CV runner with a fresh pipeline per fold, 15 scores, mean, SE, and out-of-fold predictions (FR-024, FR-025).
3. Implement `tracking.py`: the MLflow wrapper with the standard tags (`pipeline_run_id`, `git_commit`, `git_dirty`, `data_sha256`, `split_manifest_sha256`, `config_hash`, `seed`, `stage`, `candidate`) and experiment names from DOC-03 §14.2 (FR-036).
4. Implement `models/registry.py` with the two baselines, `dummy_median` and `linear_2feat` (DOC-03 §8.2, §8.3).
5. Implement the first version of `models/train.py` and the `train` subcommand: load → validate → scope → split (load) → folds → baselines CV → log. The orchestrator grows in later milestones; at M6 it ends after the baselines.
6. Add `make train`.
7. Run `make train` and inspect the runs in the MLflow UI (`uv run mlflow ui`).

**Files created.** `src/house_price/evaluation/{metrics,cv}.py`, `src/house_price/tracking.py`, `src/house_price/models/{registry,train}.py`, `tests/unit/test_metrics.py`, `tests/unit/test_cv.py`, `tests/unit/test_tracking.py`.

**Files modified.** `src/house_price/cli.py` (`train`), `configs/models.yaml` (baseline entries), `Makefile` (`train`).

**Tests.**
- Metric formulas against hand-computed values.
- CV runner: 15 scores; a new pipeline object per fold; OOF predictions for every development `Id` in every repeat; SE formula (AC-029).
- Folds identical across two generations with the same seed; different with a different seed.
- Every tracked run carries all standard tags (AC-032, partially; completed in M8).

**Expected outputs.** Two baseline results in `hpp-baselines`, each with 15 fold scores, mean, and SE. The dummy baseline's mean should be close to the standard deviation of log price in the development set (DOC-03 §8.2); the two-feature baseline should be clearly lower.

**Acceptance criteria.** M6-1 to M6-6 (Section 7.6).

**Checkpoint commit messages.**
- `feat(eval): add metrics and repeated stratified CV with shared folds`
- `feat(tracking): add MLflow wrapper with standard lineage tags`
- `feat(models): add baseline candidates and first train orchestration`
Milestone commit: `chore(milestone): complete M6 baseline modeling`; tag `m6-complete`.

**Recommended verification.** Open the MLflow UI, find both baseline runs, and confirm the 15 fold metrics, `cv_mean`, `cv_se`, and all tags. Compare the dummy baseline's `cv_mean` with the standard deviation of `log1p(SalePrice)` in the development set.

---

## M7 – Candidate Model Development

**Purpose.** Add the four model candidates to the registry, run the feature ablation that fixes each branch's feature set (DN-05), and finish the EDA report's Q17.

**Learning goals.**
- L2 versus L1 regularization; bagging versus boosting [ADR-10]
- Ablation as evidence for features [ADR-07]
- Why ablation is decided per branch

**Dependencies.** M6.

**Tasks.**
1. Add `ridge`, `lasso`, `random_forest`, and `lightgbm` to the registry with their branches and simplicity tiers (DOC-03 §8.1). Put in `models.yaml`: Lasso `max_iter` 50,000; Random Forest fixed settings; LightGBM determinism settings (DN-14) and the ablation reference parameters (DOC-03 §6.6).
2. Implement the grid-evaluation helper in `models/tuning.py` for Ridge and Lasso (25 log-spaced values; DOC-03 §10.3, §10.4). Ablation uses it to choose Ridge's reference `alpha`; M8 reuses it for tuning.
3. Implement `models/ablation.py` (DOC-03 §6.6): per branch, the full-feature score and one leave-one-feature-out score for each of the 12 engineered features, on the shared folds; the retention rule (IN-04); the ablation table logged to `hpp-ablation`.
4. Extend `train`: after the baselines, run ablation. Apply **RC-02**: if `features.yaml` has no ablation outcome, stop and print the table; if it has one and the new result differs, stop and report the difference.
5. Run `make train`. Review the ablation table, write the retained sets into `features.yaml` (`linear.dropped_engineered`, `tree.dropped_engineered`), and commit.
6. Run `make train` again: it must pass the ablation check and stop at the end of the stages implemented so far.
7. Save the ablation table as E-30 and complete Q17 in `99_eda_report.ipynb`.
8. Run each of the four candidates through the CV runner once with its reference configuration, as a development check that every registry entry fits and predicts. Log these under `hpp-cv-comparison` with the tag `stage=development_check`; they are not used for selection.

**Files created.** `src/house_price/models/ablation.py`, `src/house_price/models/tuning.py` (grid helper), `tests/unit/test_registry.py`, `tests/unit/test_ablation.py`, `reports/eda/E-30_ablation.csv`.

**Files modified.** `src/house_price/models/{registry,train}.py`, `configs/models.yaml`, `configs/features.yaml` (ablation outcomes), `notebooks/99_eda_report.ipynb`.

**Tests.**
- Registry: every candidate builds a pipeline in the correct branch; tiers correct; baselines marked ineligible for selection.
- Ablation on a synthetic dataset where one feature is pure noise and another is informative: the retention rule keeps and removes the right features.
- Ablation outcome check (RC-02): matching outcome passes; differing outcome stops `train`.

**Expected outputs.** A committed per-branch feature set; E-30; E-35 complete.

**Acceptance criteria.** M7-1 to M7-6 (Section 7.7).

**Checkpoint commit messages.**
- `feat(models): add Ridge, Lasso, Random Forest, and LightGBM candidates`
- `feat(models): add per-branch feature ablation`
- `chore(config): record ablation outcome for linear and tree branches`
- `docs(eda): complete EDA report with ablation evidence`
Milestone commit: `chore(milestone): complete M7 candidate models and ablation`; tag `m7-complete`.

**Recommended verification.** Read the ablation table line by line. For each removed feature, find its E-27/E-28 plot and write one sentence in the EDA report explaining why the evidence and the hypothesis agree or disagree.

---

## M8 – Hyperparameter Tuning

**Purpose.** Tune each candidate within its approved budget, build the blend, and produce the final 15-fold comparison of all seven candidates.

**Learning goals.**
- Grid search versus Bayesian optimization (TPE) [ADR-13]
- Search-space design and edge warnings
- Reproducibility of tuning (seeded sampler, sequential trials, no pruning)
- Nested MLflow runs

**Dependencies.** M7.

**Tasks.**
1. Complete `models/tuning.py`:
   - Ridge and Lasso: the 25-point grids (DOC-03 §10.3, §10.4), each point a nested run, with an `edge_warning` tag when the best value is on a grid edge.
   - Random Forest: Optuna TPE with seed 42, 30 trials, the DOC-03 §10.5 search space.
   - LightGBM: Optuna TPE with seed 42, 100 trials, the DOC-03 §10.6 search space and fixed determinism settings.
   - All studies: sequential (`n_jobs=1`), no pruning, in-memory storage, objective = mean log-RMSE over the shared 15 folds, each trial a nested MLflow run, best configuration saved to `artifacts/tuning/<candidate>_best.json`.
2. Add the blend to the registry: `TransformedTargetRegressor` around a `VotingRegressor` of the better linear model and LightGBM, weights 0.5/0.5 (DN-07).
3. Extend `train`: tuning, then the final CV comparison of all seven candidates on the shared folds, logged to `hpp-cv-comparison` with `stage=cv_comparison`.
4. Add the holdout isolation test (AC-030) now that tuning exists.
5. Run `make train`. Record the total duration.

**Files created.** `tests/unit/test_tuning.py`, `tests/integration/test_holdout_isolation.py`.

**Files modified.** `src/house_price/models/{tuning,registry,train}.py`, `configs/models.yaml` (search spaces, budgets), `.gitignore` (confirm `artifacts/`).

**Tests.**
- Search spaces loaded from configuration match DOC-03 §10.
- Two studies with the same seed on a tiny dataset produce identical trial parameters and scores.
- A tuning run succeeds while the holdout file is unreadable; a static check confirms only `evaluation/holdout.py` references the holdout path (AC-030).
- The blend fits, predicts dollars, and averages the two inner pipelines in log space.

**Expected outputs.** Tuning studies in `hpp-tuning` (25 + 25 grid points, about 30 and exactly 100 trials); `_best.json` files; seven final CV results in `hpp-cv-comparison`.

**Acceptance criteria.** M8-1 to M8-6 (Section 7.8).

**Checkpoint commit messages.**
- `feat(tuning): add Ridge and Lasso grid tuning with nested MLflow runs`
- `feat(tuning): add Optuna TPE tuning for Random Forest and LightGBM`
- `feat(models): add fixed-weight log-space blend`
- `test(tuning): add holdout isolation and tuning reproducibility tests`
Milestone commit: `chore(milestone): complete M8 hyperparameter tuning`; tag `m8-complete`.

**Recommended verification.** In the MLflow UI, open the LightGBM tuning parent run and confirm exactly 100 nested trials. Plot trial number against score to see the sampler converge. Check every `_best.json` for `edge_warning`.

---

## M9 – Model Selection and Holdout Evaluation

**Purpose.** Implement the selection framework, diagnostic gates, and the complete evaluation machinery (final holdout evaluation, baseline reference scoring, quality gates, temporal diagnostic). Make the real model decision on the development set. Build and test the holdout evaluation **without** using the real holdout (RC-01).

**Learning goals.**
- The one-standard-error rule and simplicity preference [ADR-11]
- Reading residual, decile, and neighborhood diagnostics
- Why the holdout is a measurement, not a tool [ADR-09]
- Reproducibility as a testable property (AC-033)

**Dependencies.** M8.

**Tasks.**
1. Implement `models/selection.py`: the 1-SE rule with tiers and the Ridge tie-break (DN-08), the blend admission rule, the selection record (DOC-03 §11.7), and the check that `diagnostic_review.yaml` is a pass for the current `selection_record_id` (DN-19).
2. Implement `evaluation/diagnostics.py`: OOF averaging across repeats (DN-09); residual, price-decile, and neighborhood plots saved to `reports/selection/` and logged to `hpp-selection`.
3. Extend `train` so that it ends with selection and the diagnostic report. `train` is now complete (DOC-03 §3.2).
4. Implement `evaluation/holdout.py` (final evaluation and baseline reference scoring, with every precondition in DOC-03 §12.2, including the refusal of a second final evaluation), `evaluation/gates.py` (DOC-03 §12.4), and `evaluation/temporal.py` (DN-10).
5. Implement the `evaluate` subcommand up to and including the temporal diagnostic (refit and artifact arrive in M10).
6. Implement smoke mode (DN-17) for `train` and `evaluate`, with the smoke holdout substitute, the auto-generated smoke review, and the `hpp-smoke` experiment. Add `make smoke`, and add a **smoke-train stage** to CI.
7. Implement `make repro-check` (AC-033): run `train` twice into separate run groups; compare split `Id` sets, selected model, selected hyperparameters (exact), and all CV metrics (within 1e-6).
8. **Real selection (rehearsal).** Run `make train`. Read the diagnostic report and fill in `reports/selection/diagnostic_review.yaml` with a judgement and reasoning for each gate. Commit the selection record, diagnostic report, and review together. This is the project's model decision; it is made using only the development set.
9. Run `make repro-check` for real and fix any nondeterminism now.
10. **Do not run the real `make evaluate`.** It runs once, in the M13 Release Run.

**Files created.** `src/house_price/models/selection.py`, `src/house_price/evaluation/{diagnostics,holdout,gates,temporal}.py`, `tests/unit/test_selection.py`, `tests/unit/test_gates.py`, `tests/unit/test_temporal.py`, `tests/integration/test_holdout_once.py`, `tests/integration/test_smoke_train.py`, `reports/selection/{selection_record.json, diagnostic plots, diagnostic_review.yaml}`.

**Files modified.** `src/house_price/models/train.py`, `src/house_price/cli.py` (`evaluate`, `--smoke`), `Makefile` (`evaluate`, `smoke`, `repro-check`), `.github/workflows/ci.yml` (smoke-train stage).

**Tests.**
- 1-SE rule on synthetic score tables: best is simplest; a simpler model within 1 SE wins; Ridge versus Lasso tie; blend admitted only with a margin above 1 SE (AC-035, AC-036).
- Diagnostic report contains all three plots and the tables required by AC-037.
- `evaluate` refuses to run without a passing review; refuses a second final evaluation for the same selection record (AC-038, smoke mode).
- Gates on synthetic metrics: each threshold passes and fails at its boundary (AC-039 to AC-041 logic).
- Temporal split uses only development rows and correct years (AC-043 logic).
- End-to-end smoke training and evaluation succeed.

**Expected outputs.** A committed selection record and a signed diagnostic review; a reproducibility report showing a pass; a CI pipeline that runs smoke training on every push.

**Acceptance criteria.** M9-1 to M9-8 (Section 7.9).

**Checkpoint commit messages.**
- `feat(selection): add one-standard-error selection and blend admission`
- `feat(selection): add out-of-fold diagnostics and review gate`
- `feat(eval): add holdout evaluation, baseline reference, gates, and temporal diagnostic`
- `feat(cli): add smoke mode and reproducibility check`
- `ci: add smoke training stage`
- `chore(selection): record selection and diagnostic review (rehearsal)`
Milestone commit: `chore(milestone): complete M9 model selection and evaluation machinery`; tag `m9-complete`.

**Recommended verification.** Read `selection_record.json` and redo the 1-SE decision by hand from the listed means and SEs. Confirm the MLflow store contains **no** run tagged `run_kind=final_holdout_evaluation` outside `hpp-smoke`.

---

## M10 – Final Artifact Creation

**Purpose.** Implement the production refit, the joblib artifact, `metadata.json`, the round-trip check, and the freeze command, and validate them end to end in smoke mode.

**Learning goals.**
- An artifact is a model plus provenance [ADR-14]
- Integrity hashes and why pickles are a trust boundary
- Semantic versioning of models

**Dependencies.** M9.

**Tasks.**
1. Implement `persistence/metadata.py`: a Pydantic model for every field in DOC-03 §15.3, a builder, and the `input_schema` and `schema_hash` generation from `schema.yaml`.
2. Implement `persistence/artifact.py`: save, SHA-256, exact round-trip check, and a **verified load** function (hash check before `joblib.load`, library-version check) that M11 reuses for both the API and the CLI.
3. Extend `evaluate`: after the temporal diagnostic, refit the selected configuration on all in-scope rows, write `models/staging/`, run the round trip, and log `run_kind=production_refit` (FR-034, FR-038, FR-039).
4. Implement `freeze` (DOC-03 §16.2): semantic-version validation, refusal of an existing version directory, the gate checks QG-10 to QG-16, rejection of smoke artifacts, byte-for-byte copy, `is_release=true`, and the `hpp-release` run. Apply **RC-03** in the clean-tree check.
5. Add `make freeze`. Extend the CI smoke stage to run `evaluate --smoke` and keep the smoke artifact as a CI job artifact for the Docker stage (M12).
6. Turn on the coverage threshold: `--cov-fail-under=80` in `make test` and CI.
7. Declare the **training-code freeze** (Section 6.5): from now on, changes to `data/`, `features/`, `pipelines/`, `models/`, `evaluation/`, `persistence/`, or training configuration require re-running `make train` and the M9 review before the release.

**Files created.** `src/house_price/persistence/{metadata,artifact}.py`, `tests/integration/test_artifact.py`, `tests/unit/test_freeze.py`, `tests/unit/test_metadata.py`.

**Files modified.** `src/house_price/evaluation/holdout.py` or `models/train.py` (refit step), `src/house_price/cli.py` (`freeze`), `Makefile`, `.github/workflows/ci.yml`, `pyproject.toml` (coverage threshold).

**Tests.**
- Metadata: every required field present and non-empty; data hash and git SHA correct (AC-044, AC-045).
- Round trip exact (AC-046).
- Refit row count equals the input rows (2,925 in a real run; the sample size in smoke mode) (AC-042 logic).
- Freeze refuses: an invalid version string, an existing version, a failed gate, a smoke artifact, a dirty tree outside `reports/evaluation/`, and a HEAD that differs from `git_commit`.
- Verified load refuses a tampered artifact and a mismatched library version.

**Expected outputs.** A smoke artifact produced on every CI run; a freeze command proven to refuse everything it should.

**Acceptance criteria.** M10-1 to M10-7 (Section 7.10).

**Checkpoint commit messages.**
- `feat(artifact): add metadata model and artifact persistence with integrity hash`
- `feat(eval): add production refit and staging artifact`
- `feat(cli): add freeze command with release gates`
- `ci: run smoke evaluation and keep smoke artifact`
Milestone commit: `chore(milestone): complete M10 artifact machinery`; tag `m10-complete`.

**Recommended verification.** Run `make smoke`, open the smoke `metadata.json`, and check every field against DOC-03 §15.3. Try `make freeze VERSION=0.0.1` with the smoke artifact in staging and confirm it is refused with a clear message.

---

## M11 – FastAPI Serving Layer

**Purpose.** Serve the pipeline through the four endpoints, with the generated request models, startup verification, structured logging, and the batch CLI, all proven consistent with direct pipeline predictions.

**Learning goals.**
- API contracts and generated validation models [ADR-15]
- Startup verification and failing fast
- Training–serving consistency as a test
- Structured logging

**Dependencies.** M10 (verified load and the smoke artifact).

**Tasks.**
1. Create `configs/api_example.json`: the 77 model inputs of one development-set row (SD-15), with `NaN` written as `null`.
2. Implement `api/settings.py` (SD-08, including `HPP_ALLOW_NON_RELEASE`).
3. Implement `api/schemas.py`: `build_request_models(schema)` with `create_model`, aliases (SD-02), all fields required, nullable as `T | None` (SD-01), strict numerics (SD-03), `Literal` categoricals, `extra="forbid"`; batch model with 1–100 items (SD-04); response models.
4. Implement `api/logging.py`: JSON formatter and the named events in DOC-04 §13.
5. Implement `api/predict.py`: frame builder in schema order with `None` → `NaN`, the guard (SD-16), the domain flag from `metadata.scope_rule`, response assembly.
6. Implement `api/errors.py`: 422 logging handler (locations and types only), generic 500 handler with `request_id`.
7. Implement `api/app.py`: the lifespan with the 13 startup steps (DOC-04 §5.2), request-ID and timing middleware, the four routes, and the OpenAPI example.
8. Implement the batch CLI `predict` subcommand (DOC-04 §10): verified load, CSV parsing with DN-18 tokens, header normalization with the ingestion mapping (raw dataset layout accepted), SD-12 column rules, lazy Pandera validation with the inference schema, exit codes 0/2/3, output CSV.
9. Add `make serve` and `make predict`.
10. Add the import-rule test: `house_price.api` imports nothing from `features`, `pipelines`, `models`, or `evaluation` (DOC-04 §18.1).

**Files created.** `configs/api_example.json`, `src/house_price/api/{settings,schemas,logging,predict,errors,app}.py`, `tests/integration/test_api.py`, `tests/unit/test_contract_parity.py`, `tests/integration/test_training_serving_consistency.py`, `tests/integration/test_cli_predict.py`, `tests/integration/test_startup.py`, `tests/unit/test_import_rules.py`, `tests/fixtures/consistency_rows.csv`.

**Files modified.** `src/house_price/cli.py` (`predict`), `Makefile` (`serve`, `predict`).

**Tests.** Use the smoke artifact (or an equivalent fixture artifact) with `HPP_ALLOW_NON_RELEASE=true`.
- All DOC-04 §16.1 API tests (AC-048 to AC-056, AC-058).
- Contract parity between Pydantic and Pandera (DOC-04 §9.3).
- Training–serving consistency: API equals direct `predict` exactly (AC-057).
- CLI: valid file, invalid file (exit 2, no output, full report), column rules (including a raw-layout file scored without renaming, and `Id`/`PID` passed through), CLI equals API (AC-062, AC-063).
- Startup refusals: hash mismatch, version mismatch, non-release without override, schema-hash mismatch.
- Guard violation → 500; unexpected exception contained.

**Expected outputs.** `make serve` starts a verified service; `/docs` shows the four endpoints with the example; `make predict` scores a CSV.

**Acceptance criteria.** M11-1 to M11-8 (Section 7.11).

**Checkpoint commit messages.**
- `feat(api): add generated request models and settings`
- `feat(api): add prediction path, guard, and structured logging`
- `feat(api): add lifespan startup verification and endpoints`
- `feat(cli): add batch prediction command`
- `test(api): add API, contract parity, consistency, and startup tests`
Milestone commit: `chore(milestone): complete M11 FastAPI serving layer`; tag `m11-complete`.

**Recommended verification.** Start `make serve` with the smoke artifact, open `http://127.0.0.1:8000/docs`, run the example through `/predict`, then remove `PoolQC` from the body and confirm the 422 points at it. Watch the terminal: exactly one JSON line per prediction request, none for `/health`.

---

## M12 – Dockerization and Deployment

**Purpose.** Package the service as the two-stage, non-root image designed in DOC-04, test it locally and in CI, and prepare the GHCR and Render configuration so that the release deploy is a routine step.

**Learning goals.**
- Multi-stage builds and layer caching
- Why the artifact is baked in and why the runtime needs `libgomp1` [ADR-15, SD-14]
- Running as non-root
- How a prebuilt image reaches Render (SD-06, SD-07)

**Dependencies.** M11.

**Tasks.**
1. Write the `Dockerfile` (DOC-04 §11.2): `builder` stage with `uv sync --frozen --no-dev` from the lockfile first, then the package; `runtime` stage with `libgomp1`, `appuser` (UID 10001), the venv, `src/`, `configs/`, and the model directory last; build args `MODEL_DIR` and `MODEL_VERSION`; OCI version label; `EXPOSE 8000`; Python standard-library `HEALTHCHECK`; `USER appuser`; the `sh -c` Uvicorn command with one worker and no access log (SD-09).
2. Write `.dockerignore` (DOC-04 §11.2).
3. Add Make targets: `docker-build` (checks the label equals `metadata.model_version`), `docker-test` (DOC-04 §16.5 steps 2–6, also running the container with `PORT=10000` to mimic Render's port handling), and `docker-push` (refuses a tag that already exists in GHCR).
4. Add the CI Docker stage: download the smoke artifact, build, run with `HPP_ALLOW_NON_RELEASE=true`, test, stop. Never push.
5. Prepare deployment:
   - Confirm you can authenticate to GHCR from your machine with a personal access token that has package write scope.
   - Write `docs/deployment.md` with the exact Render settings from DOC-04 §12.6 (image URL pattern, health check path `/health`, auto-deploy off, `HPP_LOG_LEVEL=INFO`, free instance type) and the release and rollback procedures from DOC-04 §12.3 and §17.4.
6. **Do not push a smoke image or create the Render service with a smoke image.** Only release artifacts may run outside CI (SD-08, NFR-025). The first push and the first Render deploy happen in the M13 Release Run with the real image.

**Files created.** `Dockerfile`, `.dockerignore`, `docs/deployment.md`.

**Files modified.** `Makefile` (`docker-build`, `docker-test`, `docker-push`), `.github/workflows/ci.yml` (Docker stage).

**Tests.** The container test (CI and `make docker-test`): `/health` 200 within the start period; process UID ≠ 0 (AC-059); the example prediction equals the smoke pipeline's direct prediction (AC-060, CI form); the container works with `PORT=10000`.

**Expected outputs.** A reproducible image build; a green CI Docker stage; a written, reviewed deployment procedure.

**Acceptance criteria.** M12-1 to M12-6 (Section 7.12).

**Checkpoint commit messages.**
- `build(docker): add two-stage non-root image with baked-in artifact`
- `ci: add Docker build and container test stage`
- `docs(deploy): add GHCR and Render deployment procedure`
Milestone commit: `chore(milestone): complete M12 dockerization and deployment preparation`; tag `m12-complete`.

**Recommended verification.** Build twice in a row and confirm the second build reuses the dependency layer. Run `docker image inspect` and confirm the version label and the user. Run the container with a wrong `HPP_MODEL_DIR` and confirm it exits with a `startup.failed` log line.

---

## M13 – Testing, Documentation, and Release

**Purpose.** Complete the test suite and documentation, then perform the one-time Release Run that evaluates the holdout, freezes the artifact, builds and deploys the image, and verifies the live service.

**Learning goals.**
- Release discipline: one path, every gate, no shortcuts
- Model cards and honest limitations [ADR-19]
- Operating a deployed service

**Dependencies.** M1 to M12.

**Tasks — Part A: Completion (before the Release Run).**
1. Run the full test suite; fill any gaps so that every category in FR-058 has tests and coverage of `src/` is at least 80% (AC-064, AC-065).
2. Confirm Ruff and mypy are clean (AC-067) and pre-commit passes on all files (AC-068).
3. Add docstrings to every public function and class (AC-073).
4. Write the README (except the results table): problem statement, architecture diagram (from DOC-03 §4.1 and DOC-04 §4.1), quickstart (setup → validate → train → evaluate → serve → call the API), link to the live service (filled after deployment).
5. Draft `docs/model_card.md` (Mitchell et al. format): intended use; out-of-scope uses (real-world valuation, homes above 4,000 sq ft, markets outside Ames 2006–2010); the refit statement; the retransformation bias; the SE limitation; placeholders only for numbers that come from the Release Run.
6. Verify OpenAPI: the four endpoints and the valid example (AC-058).
7. Finalize the data card (AC-071).

**Tasks — Part B: Release Run.** Follow exactly (DOC-04 §17.2):

| # | Step | Command / action | Stop if |
|---|---|---|---|
| R1 | Clean `main`; CI green | `git status`; GitHub Actions | Any change or red CI |
| R2 | Reproducibility check | `make repro-check` | Any discrete difference or metric difference > 1e-6 |
| R3 | Train | `make train` | Any gate QG-01 to QG-08 fails; the selection differs from the M9 rehearsal (investigate before continuing) |
| R4 | Diagnostic review | Fill in `diagnostic_review.yaml` for the new `selection_record_id`; commit it with the selection record and plots: `chore(release): record diagnostic review for vX.Y.Z` | Any gate judged fail |
| R5 | Evaluate (**the one real holdout evaluation**) | `make evaluate` | Any quality gate fails: record it and follow ADR supersession; do not re-select |
| R6 | Freeze | `make freeze VERSION=X.Y.Z` | Any freeze check fails |
| R7 | Tag | `git tag -a vX.Y.Z -m "Release vX.Y.Z"`; `git push --tags` | — |
| R8 | Commit evaluation reports (RC-03) | `docs(release): add evaluation reports for vX.Y.Z` | — |
| R9 | Build image | `make docker-build VERSION=X.Y.Z` | Label differs from metadata |
| R10 | Test image | `make docker-test VERSION=X.Y.Z` (release artifact, no override) | Any container test fails |
| R11 | Push image | `make docker-push VERSION=X.Y.Z` | Tag already exists |
| R12 | Create or update the Render service | Per `docs/deployment.md`; set the image tag to `X.Y.Z`; deploy | Health check never passes |
| R13 | Verify live service | `curl https://<service>.onrender.com/health`; `curl …/model-info` | Version differs from `X.Y.Z` (AC-061) |
| R14 | Finish documentation | README results table and model card numbers from `metadata.json`; live URL; release notes with the R13 output: `docs(release): publish results and model card for vX.Y.Z` | Any number differs from `metadata.json` |
| R15 | Final Definition of Done | Section 8 | Any unchecked item |

**Files created.** `docs/model_card.md`, `docs/release_notes/vX.Y.Z.md`, `models/X.Y.Z/` (gitignored), `reports/evaluation/*`.

**Files modified.** `README.md`, `docs/data_card.md`, `docs/deployment.md` (confirmed Render behavior, DOC-04 §12.5), any file needing docstrings or tests.

**Tests.** Full suite locally and in CI; container test with the release image; live checks in R13.

**Expected outputs.** A tagged release; a frozen artifact; a pushed image; a live service reporting the release version; complete documentation.

**Acceptance criteria.** M13-1 to M13-10 (Section 7.13).

**Checkpoint commit messages.** As listed in R4, R8, R14, plus `test: close coverage gaps`, `docs: add docstrings to public API`, `docs(readme): complete README`, `docs(model-card): add model card draft`.
Milestone commit: `chore(milestone): complete M13 testing, documentation, and release`; tag `m13-complete` (in addition to the release tag `vX.Y.Z`).

**Recommended verification.** From a different machine or a fresh clone: follow the README quickstart to a successful local prediction, then call the live service with the example payload and compare the price with the local prediction from the same release artifact.

---

# 5. Testing Strategy Across Milestones

## 5.1 When Tests Appear

Tests are written in the milestone that creates the code they test. No milestone is complete with untested new modules.

| Milestone | Test files introduced | Test type |
|---|---|---|
| M1 | `test_package.py`, `test_config.py` | Unit |
| M2 | `test_load.py`, `test_schema.py`, `test_scope_split.py`, `test_profile.py`, `test_validate.py` (and schema-model cases in `test_config.py`) | Unit (fixture and `data`-marked) |
| M3 | Notebook review checks | Review (AC-011, AC-012) |
| M4 | `test_semantic.py`, `test_engineer.py` | Unit |
| M5 | `test_pipeline.py`, `test_leakage.py` | Unit, leakage |
| M6 | `test_metrics.py`, `test_cv.py`, `test_tracking.py` | Unit |
| M7 | `test_registry.py`, `test_ablation.py` | Unit |
| M8 | `test_tuning.py`, `test_holdout_isolation.py` | Unit, integration |
| M9 | `test_selection.py`, `test_gates.py`, `test_temporal.py`, `test_holdout_once.py`, `test_smoke_train.py` | Unit, integration, smoke |
| M10 | `test_metadata.py`, `test_artifact.py`, `test_freeze.py` | Unit, integration |
| M11 | `test_api.py`, `test_contract_parity.py`, `test_training_serving_consistency.py`, `test_cli_predict.py`, `test_startup.py`, `test_import_rules.py` | API, schema, consistency, serving |
| M12 | CI container test | Docker |
| M13 | `tests/quality/test_quality_gate.py` (reads the release metrics); gap-filling tests | Quality gate |

## 5.2 How Testing Evolves

1. **M1–M5: correctness of pieces.** Small, fast unit tests with hand-built fixtures. Each transformer and builder is tested in isolation, so failures point directly at the broken piece.
2. **M5: the first ML-specific test.** The leakage test is the most important test in the project. It is written as soon as the pipeline exists, before any model result can mislead you.
3. **M6–M9: correctness of procedures.** Tests check that CV, tuning, selection, and gates follow their rules, using synthetic score tables and tiny datasets, so they do not depend on real model quality.
4. **M9–M10: end-to-end smoke tests.** Smoke mode runs the whole training system in CI on every push.
5. **M11: serving contracts.** API, parity, consistency, and startup tests use the smoke artifact.
6. **M12: the container.** The same checks, run against the built image.
7. **M13: quality gates on real results.** The only tests that depend on the real model's performance, run against the release metrics.

## 5.3 Coverage Policy

- Measured from M1 and shown in every CI run.
- Enforced at 80% of `src/` from M10 (when most modules exist), so earlier milestones are not blocked by modules that are still skeletons (AC-065, NFR-014).

## 5.4 Unit, Integration, API, and Serving Tests

| Kind | What it proves | Where it lives | Example |
|---|---|---|---|
| Unit | One function or class behaves as specified | `tests/unit/` | `FeatureEngineer` formulas |
| Integration | Several modules work together | `tests/integration/` | Tuning never opens the holdout |
| Smoke | The whole training system runs end to end | `tests/integration/test_smoke_train.py`, CI stage | `train --smoke`, `evaluate --smoke` |
| API | Endpoints follow their contracts | `tests/integration/test_api.py` | Batch of 101 → 422 |
| Serving | Startup, consistency, and containment behave correctly | `tests/integration/test_startup.py`, consistency tests | Tampered artifact refused |
| Docker | The image runs as designed | CI Docker stage, `make docker-test` | Non-root, `/health` 200 |
| Quality gate | Real metrics meet targets | `tests/quality/` | Holdout log-RMSE ≤ 0.13 |

## 5.5 Tests That Need the Dataset

The dataset is not committed and is not available in CI. Tests that need the real file are marked `@pytest.mark.data` and are skipped in CI; they must pass locally before each milestone commit that touches data code. Everything else uses fixtures in `tests/fixtures/` or the smoke pipeline, which runs on a sample built from committed fixture data.

**Note on CI smoke training (RC-04).** Because CI has no dataset, the CI smoke stage (DN-17) runs on a committed synthetic fixture file that follows `schema.yaml` (generated once by a test helper and committed under `tests/fixtures/`). Locally, `make smoke` uses the real development set sample. Both exercise every stage.

---

# 6. Git and Version Control Strategy

## 6.1 Commit Philosophy

- **Small and focused.** One commit does one thing: a feature, a test set, a configuration change, or a documentation change.
- **Always green.** Every commit on `main` passes pre-commit and CI.
- **Generated evidence is committed with its decision.** The ablation outcome is committed with its table; the selection record with its review.
- **Never commit** data, `mlruns/`, `models/`, `artifacts/`, or secrets.

## 6.2 Branching

- `main` is always releasable up to the current milestone.
- Work on a short-lived branch per milestone: `m<N>/<short-name>` (for example `m5/preprocessing`).
- Merge to `main` through a pull request once CI is green, even when working alone: the PR view makes you read your own diff.

## 6.3 Commit Naming Convention

Format: `<type>(<scope>): <imperative summary>`

| Type | Use for |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fixes |
| `test` | Tests only |
| `refactor` | Code change with no behavior change |
| `docs` | Documentation, notebooks' narrative, reports |
| `build` | Dockerfile, dependencies |
| `ci` | Workflow changes |
| `chore` | Configuration, milestone markers, housekeeping |

Scopes: `repo`, `config`, `data`, `eda`, `features`, `pipeline`, `models`, `tuning`, `selection`, `eval`, `tracking`, `artifact`, `api`, `cli`, `docker`, `deploy`, `readme`, `model-card`, `release`, `milestone`.

Examples: `feat(data): add scope rule and persisted stratified split`; `fix(pipeline): keep pandas output from one-hot encoder`; `chore(config): record ablation outcome for linear and tree branches`.

## 6.4 Checkpoint and Milestone Commits

- **Checkpoint commits** are the commits listed under each milestone. They are natural resting points where all tests pass.
- **Milestone commit:** when a milestone's acceptance table is fully satisfied, add an empty commit `chore(milestone): complete M<N> <milestone name>` whose body lists the evidence locations (for example "Evidence: reports/eda/, MLflow experiment hpp-baselines"), then an annotated tag `m<N>-complete`.

## 6.5 Training-Code Freeze

From the end of M10, the training side (`data/`, `features/`, `pipelines/`, `models/`, `evaluation/`, `persistence/`, `tracking.py`, and training configuration) should change only to fix bugs. Any such change must be followed by `make train` and a fresh M9-style review before the Release Run. This keeps the M9 rehearsal meaningful and avoids surprises at release.

## 6.6 Release Tagging

- Release tags: annotated, `vX.Y.Z` (semantic versioning, DOC-03 §15.4), created in Release Run step R7.
- The tag's commit is the commit that contains the diagnostic review for that release, and it equals `metadata.git_commit` (DOC-01 §11.9).
- Image tag = model version = release tag without the `v` (DOC-04 §17.3).
- Tags are never moved or deleted. A fix produces a new version.

---

# 7. Acceptance Criteria Framework

## 7.0 How to Use These Tables

Each milestone has a table of criteria. A milestone is complete only when **every** criterion is met and its evidence exists. "DOC-01 AC" shows which DOC-01 acceptance criteria the milestone satisfies; the milestone criteria are how you check them during implementation. Where a DOC-01 AC can only be fully verified on real release results, it is marked "(logic)" in earlier milestones and "(final)" in M13.

A **failure condition** is a result that means "stop and fix before continuing".

## 7.1 M1 – Project Foundation

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M1-1 | Fresh clone installs with one command | `make setup` in a new directory | Terminal output | Any manual step needed |
| M1-2 | Package and CLI importable | `test_package.py` | CI log | Import error |
| M1-3 | Config models reject bad YAML | `test_config.py` | CI log | Unknown key or wrong type accepted |
| M1-4 | Lint, typecheck, tests green locally and in CI | `make lint typecheck test`; Actions | CI run link | Any red job |
| M1-5 | pre-commit hooks installed and passing | `pre-commit run --all-files` | Terminal output | Hook failure (AC-068 groundwork) |
| M1-6 | ADR in MADR format | File review | `docs/adr/` | Missing ADR file or section (AC-070) |

## 7.2 M2 – Data Ingestion and Validation (DOC-01 AC-001 to AC-009)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M2-1 | Raw file never modified | Hash before and after (AC-001) | Test output | Hash changes |
| M2-2 | Hash mismatch stops everything | Modified-copy test (AC-002) | Test output | Anything written after mismatch |
| M2-3 | Stable missing counts with explicit tokens | AC-003 test | Test output; E-04 later | `MasVnrType` count includes `"None"` text |
| M2-4 | Schema rejects the five fault types and accepts the raw file | AC-004 tests | Test output | Raw file fails, or a fault passes |
| M2-5 | One configuration source for allowed values | AC-005 test | Test output | Two separate value lists |
| M2-6 | 2,925 in-scope rows; the 5 removed IDs recorded | AC-006 | Scope record | Any other count |
| M2-7 | Split sizes, disjointness, persistence, balance ≤ 2 pp | AC-007 to AC-009 | `split_manifest.json`; test output | Regeneration on second run; imbalance |
| M2-8 | Data card drafted | Review | `docs/data_card.md` | Missing section (AC-071 groundwork) |

## 7.3 M3 – Exploratory Data Analysis (DOC-01 AC-010 to AC-013)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M3-1 | E-01 to E-26 and E-31 to E-34 exist | Checklist against DOC-02 §13 | `reports/eda/`, `reports/figures/eda/` | Any missing deliverable |
| M3-2 | Holdout not used for target analysis | Source search (AC-011) | Search output | Holdout path in a non-exception notebook |
| M3-3 | No production logic in notebooks | Code review (AC-012) | Review note in milestone commit | Function used by `src/` defined in a notebook |
| M3-4 | Confirmation register covers every documented property | Review of E-35 | `99_eda_report.ipynb` | Property without a confirmed/not-confirmed status |
| M3-5 | Discrepancies handled | Review | Data card; any ADR supersession | ADR-fixed number contradicted and ignored |
| M3-6 | Notebooks rerun cleanly | Restart and run all | Commit with fresh outputs | Error or stale output |

## 7.4 M4 – Data Preparation and Feature Engineering (DOC-01 AC-014 to AC-017)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M4-1 | Semantic filling correct | AC-014 test | Test output | Any listed column still missing |
| M4-2 | Transformers stateless and non-mutating | AC-015 tests | Test output | Output depends on fit data; input mutated |
| M4-3 | All 12 formulas exact | AC-016 test | Test output | Any mismatch |
| M4-4 | Ordinal map and `MSSubClass` correct | AC-017 tests | Test output | Wrong code or numeric `MSSubClass` |
| M4-5 | No value repair outside the pipeline | Code review | Review note | Clipping or editing in `data/` |
| M4-6 | E-27 to E-29 produced; Q17 partly answered | Review | Notebook 07; E-35 | Missing deliverable |

## 7.5 M5 – Preprocessing Architecture (DOC-01 AC-018, AC-020 to AC-027)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M5-1 | Pipeline takes the 77 columns and returns dollars | AC-018, AC-021 tests | Test output | Excluded column required; non-positive output |
| M5-2 | Branches behave as designed | AC-022, AC-023 tests | Test output | Scaled tree output; crash on unseen category |
| M5-3 | Imputation fitted on fitting data only | AC-024, AC-025 tests | Test output | Leakage test fails |
| M5-4 | Clone and round trip work | AC-026; round-trip test | Test output | Any difference |
| M5-5 | Groups driven by YAML and fully covered | AC-027; coverage check | Test output | Column silently dropped |
| M5-6 | No forbidden techniques | AC-020 | Test output | Any forbidden step present |

## 7.6 M6 – Baseline Modeling (DOC-01 AC-029, AC-032 partial)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M6-1 | Metrics correct | `test_metrics.py` | Test output | Any mismatch |
| M6-2 | 15 folds, fresh pipeline per fold | `test_cv.py` | Test output | Reused fitted object |
| M6-3 | Folds shared and deterministic | Fold tests | `folds.json` | Different folds with same seed |
| M6-4 | Every run tagged | `test_tracking.py`; MLflow UI | Screenshot or run listing | Missing tag |
| M6-5 | Both baselines logged with 15 scores, mean, SE | MLflow UI | `hpp-baselines` runs | Missing metric |
| M6-6 | Baseline results plausible | Compare dummy mean with log-price SD | Note in milestone commit | Dummy far from SD (indicates a bug) |

## 7.7 M7 – Candidate Model Development (DOC-01 AC-019)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M7-1 | Four candidates registered with correct branch and tier | `test_registry.py` | Test output | Wrong branch or tier |
| M7-2 | Ablation follows DN-05 and IN-04 | `test_ablation.py`; review | Ablation table | Retention rule applied incorrectly |
| M7-3 | Ablation outcome committed per branch | Review | `features.yaml` commit | Outcome applied automatically without commit |
| M7-4 | RC-02 check works | Test; second `make train` | Terminal output | Stale outcome accepted |
| M7-5 | E-30 saved; E-35 complete | Review | `reports/eda/E-30_ablation.csv`; notebook 99 | Q17 unanswered |
| M7-6 | Every candidate fits and predicts on real data | Development-check runs | `hpp-cv-comparison` (`development_check`) | Any failure |

## 7.8 M8 – Hyperparameter Tuning (DOC-01 AC-028 to AC-032)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M8-1 | Budgets and spaces match DOC-03 §10 | Tests; MLflow | Tuning runs | Wrong count or range (AC-031) |
| M8-2 | Tuning reproducible | Same-seed test | Test output | Different trials |
| M8-3 | Holdout never read | AC-030 test | Test output | Holdout referenced or opened |
| M8-4 | All trials logged with provenance | AC-032 check | MLflow | Missing tag or metric |
| M8-5 | Blend built as DN-07 | Blend test | Test output | Fitted weights or dollar-space averaging |
| M8-6 | Final CV comparison for exactly 7 candidates | AC-028, AC-029 | `hpp-cv-comparison` | Missing or extra candidate |

## 7.9 M9 – Model Selection and Holdout Evaluation (DOC-01 AC-033, AC-035 to AC-038, AC-039 to AC-041 and AC-043 (logic))

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M9-1 | Selection logic correct | AC-035, AC-036 tests | Test output | Wrong candidate on synthetic tables |
| M9-2 | Diagnostic report complete | AC-037 review | `reports/selection/` | Missing plot or table |
| M9-3 | Holdout preconditions enforced | `test_holdout_once.py` | Test output | Evaluate runs without review, or twice |
| M9-4 | Gates and temporal logic correct | `test_gates.py`, `test_temporal.py` | Test output | Boundary errors; holdout rows in temporal split |
| M9-5 | Smoke training runs in CI | CI | CI run link | Red smoke stage |
| M9-6 | Reproducibility check passes | `make repro-check` (AC-033) | Repro report | Any discrete or > 1e-6 difference |
| M9-7 | Real selection and review committed | Review | Selection record, review file | Review missing reasoning |
| M9-8 | Real holdout untouched | MLflow search | No `final_holdout_evaluation` run outside `hpp-smoke` | Any such run |

## 7.10 M10 – Final Artifact Creation (DOC-01 AC-042, AC-044 to AC-047 (logic))

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M10-1 | Metadata complete and correct | `test_metadata.py`, AC-044, AC-045 | Smoke `metadata.json` | Missing or empty field |
| M10-2 | Round trip exact | AC-046 | Test output | Any difference |
| M10-3 | Refit uses all rows passed to it | Refit test (AC-042 logic) | Test output | Wrong row count |
| M10-4 | Freeze refuses every invalid case | `test_freeze.py` | Test output | Any invalid freeze accepted |
| M10-5 | Verified load refuses tampered or mismatched artifacts | `test_artifact.py` | Test output | Tampered file loaded |
| M10-6 | Smoke artifact produced in CI | CI | CI job artifact | Missing |
| M10-7 | Coverage ≥ 80% enforced | CI | Coverage report | Below threshold |

## 7.11 M11 – FastAPI Serving Layer (DOC-01 AC-048 to AC-058, AC-062, AC-063)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M11-1 | Four endpoints meet their contracts | `test_api.py` | Test output | Any AC-048 to AC-051 failure |
| M11-2 | Validation rejects and accepts correctly | AC-052, AC-053 tests; SD-01, SD-03 tests | Test output | Bad input predicted, or valid `null` rejected |
| M11-3 | Domain flag correct at the boundary | AC-054 | Test output | Wrong flag at 4000 or 4001 |
| M11-4 | Model loaded once; one log line per prediction | AC-055, AC-056 | Test output | Extra loads or log lines |
| M11-5 | API equals direct pipeline prediction | AC-057 | Test output | Any difference |
| M11-6 | Pydantic and Pandera agree | Contract parity test | Test output | Any disagreement |
| M11-7 | Startup refuses bad artifacts and contracts | `test_startup.py` | Test output | Service starts |
| M11-8 | CLI behaves as designed and equals the API | AC-062, AC-063 | Test output | Partial output on invalid file; mismatch |

## 7.12 M12 – Dockerization and Deployment (DOC-01 AC-059, AC-060)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M12-1 | Image builds with the designed stages and layers | `make docker-build`; build log | Build log | Dependencies reinstalled on a code-only change |
| M12-2 | Runs as non-root | Container test (AC-059) | CI log | UID 0 |
| M12-3 | Healthy and correct in a container | Container test (AC-060, CI form) | CI log | `/health` not 200; price differs |
| M12-4 | Works with Render-style `PORT` | `PORT=10000` test | Test log | Not listening on the given port |
| M12-5 | CI Docker stage green; image never pushed from CI | CI | Workflow file; CI run | Push step present in CI |
| M12-6 | Deployment procedure written and GHCR access confirmed | Review | `docs/deployment.md` | Missing setting or rollback step |

## 7.13 M13 – Testing, Documentation, and Release (DOC-01 AC-039 to AC-043 (final), AC-061, AC-064 to AC-073)

| ID | Mandatory criterion | Verification | Evidence | Failure condition |
|---|---|---|---|---|
| M13-1 | All test categories present and passing; coverage ≥ 80% | AC-064, AC-065 | CI report | Missing category or below 80% |
| M13-2 | Lint, types, hooks clean; docstrings complete | AC-067, AC-068, AC-073 | CI; hook output | Any error |
| M13-3 | Release Run R1–R6 completed in order, holdout evaluated once | Release notes; MLflow | One `final_holdout_evaluation` run for the release | Steps out of order; second evaluation |
| M13-4 | Quality gates pass | AC-039 to AC-041 (final) | `quality_gates.json` | Any gate fails |
| M13-5 | Temporal diagnostic and refit recorded | AC-042, AC-043 (final) | Metadata; MLflow | Missing record; wrong row count |
| M13-6 | Frozen artifact and tag consistent | AC-044 to AC-047 (final) | `models/X.Y.Z/`; tag | Tag commit ≠ `git_commit` |
| M13-7 | Release image tested and pushed | AC-059, AC-060 with release artifact | `make docker-test` log | Test failure |
| M13-8 | Live service verified | AC-061 | Release notes with R13 output | Version mismatch |
| M13-9 | Documentation complete and consistent | AC-069 to AC-072 | README, model card, data card | Any number differs from metadata |
| M13-10 | Definition of Done complete | Section 8 | Checked list in release notes | Any unchecked item |

---

# 8. Final Definition of Done

The project is done when every box is checked.

**Data**
- [ ] Raw file unchanged and hash-verified on every run
- [ ] Ingestion schema passes on the raw file; five fault types rejected
- [ ] One configuration source for allowed values and ranges
- [ ] 2,925 in-scope rows; split persisted, disjoint, balanced; manifest committed

**EDA**
- [ ] Deliverables E-01 to E-36 present
- [ ] Confirmation register complete; discrepancies in the data card
- [ ] Holdout protected; notebooks contain no production logic

**Feature engineering**
- [ ] Semantic filler and feature engineer stateless and tested
- [ ] All 12 engineered features exact; ordinal map and `MSSubClass` correct
- [ ] Ablation outcome committed per branch

**Preprocessing**
- [ ] One raw-to-dollars pipeline per candidate
- [ ] Linear and tree branches as designed; unseen categories handled
- [ ] Leakage test passes

**Modeling**
- [ ] All 7 candidates evaluated on shared 5×3 folds
- [ ] Tuning budgets and spaces as designed; holdout never read during tuning
- [ ] Selection by the 1-SE rule and blend rule; diagnostic review recorded with reasoning

**Evaluation**
- [ ] One final holdout evaluation for the release; baseline reference scoring once, used for no decision
- [ ] Holdout log-RMSE ≤ 0.13; MAPE ≤ 10%; both baselines beaten by a clear margin (IN-11)
- [ ] Temporal diagnostic recorded as not used for selection

**MLflow**
- [ ] Every run tagged with lineage; experiments as in DOC-03 §14.2
- [ ] Release run holds the frozen artifact and metadata

**Artifacts**
- [ ] Refit on 2,925 rows; round trip exact
- [ ] `metadata.json` complete; `model_sha256`, `schema_hash`, versions correct
- [ ] Frozen version immutable; tag commit = `git_commit`

**API**
- [ ] Four endpoints meet their contracts; validation strict; domain flag correct
- [ ] Startup verification refuses every bad case
- [ ] API equals direct pipeline prediction; CLI equals API
- [ ] One structured log line per prediction request

**Docker**
- [ ] Two-stage image, `libgomp1`, non-root, artifact baked in, label = version
- [ ] Container test passes with the release artifact

**Deployment**
- [ ] Image pushed to GHCR with the release tag
- [ ] Render service live over HTTPS, health check `/health`, auto-deploy off
- [ ] `/model-info` reports the release version

**Testing**
- [ ] All eight DOC-01 test categories present and passing
- [ ] Coverage of `src/` ≥ 80%; Ruff and mypy clean; pre-commit passing
- [ ] CI runs lint, typecheck, tests, smoke training, and Docker build on every push

**Documentation**
- [ ] README with results, architecture diagram, quickstart, and live link
- [ ] ADR-000 and ADR-01 to ADR-20 in MADR format
- [ ] Data card and model card complete and consistent with metadata
- [ ] OpenAPI docs with a valid example; docstrings on public APIs
- [ ] Release notes, including live verification output

**Reproducibility**
- [ ] `make repro-check` passed on the release commit
- [ ] A fresh clone reaches a local prediction using only the README

---

# 9. Risks and Mitigation

## 9.1 Leakage Risks

| Risk | How it happens | Mitigation | Milestone |
|---|---|---|---|
| Statistics learned from validation data | Imputing or scaling outside the pipeline | All fitted steps in the pipeline; fresh pipeline per fold; leakage test | M5, M6 |
| Holdout influences decisions | Looking at holdout prices in EDA; tuning or selecting with it | Single reader module; isolation test; RC-01 (real evaluation only at release) | M3, M8, M9 |
| Transaction-outcome features | Including `SaleType` or `SaleCondition` | 77-column model-input schema; AC-018 test | M5 |
| Feature choices fitted to the whole dataset | Ablation or EDA on all rows | Ablation on development CV only | M7 |
| Re-running evaluation after a disappointing result | Human temptation | `evaluate` refuses a second evaluation; release checklist says record and escalate | M9, M13 |

## 9.2 Schema Mismatch Risks

| Risk | Mitigation | Milestone |
|---|---|---|
| Data dictionary spellings differ from the file | Allowed values include verified file spellings; E-02 audit | M2, M3 |
| API and training contracts drift | One `schema.yaml`; generated models; schema hash checked at startup; parity test | M2, M11 |
| Column order differs at serving | Frame built in schema order; name-based selection; `feature_names_in_` check | M5, M11 |
| Parser treats `"None"` as missing | Explicit tokens (DN-18); AC-003 | M2 |
| Library upgrade changes pickled objects | Lockfile; versions in metadata; startup version check | M1, M10, M11 |

## 9.3 Deployment Risks

| Risk | Mitigation | Milestone |
|---|---|---|
| LightGBM fails in the slim image | `libgomp1` installed (SD-14); container test runs a real prediction | M12 |
| Service listens on the wrong port on Render | `sh -c` with `${PORT:-8000}`; `PORT=10000` test | M12 |
| Artifact missing from the image | Build arg check; startup refuses to run without a verified artifact | M12 |
| Smoke artifact reaches production | `is_release` check at startup; freeze rejects smoke; CI never pushes | M10, M12 |
| Bad deploy replaces a good one | Health check gates the deploy; behavior confirmed at first release; rollback by tag | M13 |
| Cold starts on the free tier | Accepted (ADR-15); documented in the README | M13 |

## 9.4 Reproducibility Risks

| Risk | Mitigation | Milestone |
|---|---|---|
| Nondeterministic LightGBM threading | DN-14 settings; fixed `n_jobs` | M8 |
| Optuna order effects | Sequential trials, seeded sampler, no pruning | M8 |
| Split regenerated accidentally | `create_or_load_split` never overwrites; manifest committed | M2 |
| Uncommitted changes baked into results | `git_dirty` tag; clean-tree checks in `evaluate` and `freeze` | M6, M10 |
| Training code changes after the rehearsal | Training-code freeze (Section 6.5) | M10 onward |
| Nondeterminism discovered too late | `make repro-check` in M9 and again at release | M9, M13 |

## 9.5 Evaluation Risks

| Risk | Mitigation | Milestone |
|---|---|---|
| Over-reading small CV differences | Mean ± SE everywhere; 1-SE rule | M6, M9 |
| SE too optimistic (correlated folds) | Documented limitation in the model card | M13 |
| Tuning optimism in CV scores | Locked holdout gives the unbiased estimate | M13 |
| Diagnostic gates judged carelessly | Written reasoning required for each gate; review committed | M9, M13 |
| Holdout gate fails at release | Recorded, escalated through ADR supersession; no re-selection | M13 |
| Small temporal test set over-interpreted | Interpretation guidance in DOC-03 §13.4; labelled "not used for selection" | M13 |

---

# 10. Release Readiness Checklist

Complete this list immediately before Release Run step R5 (the holdout evaluation), then again before step R12 (deployment), as indicated.

## 10.1 Before R5 (the one real holdout evaluation)

**Testing verification**
- [ ] CI green on the release candidate commit (all five stages)
- [ ] Full local test run including `@pytest.mark.data` tests
- [ ] Coverage ≥ 80%; Ruff and mypy clean

**Reproducibility verification**
- [ ] `make repro-check` passed on this commit
- [ ] Selection matches the M9 rehearsal, or the difference is explained and fixed

**Selection verification**
- [ ] `diagnostic_review.yaml` for the current `selection_record_id` records pass for all three gates, with reasoning
- [ ] Review, selection record, and plots committed; working tree clean

**Holdout verification**
- [ ] No `final_holdout_evaluation` run exists outside `hpp-smoke`

## 10.2 After R6, before R12

**Artifact verification**
- [ ] `models/X.Y.Z/` exists with `model.joblib` and `metadata.json`
- [ ] `is_release=true`; `model_sha256` matches the file
- [ ] `git_commit` = the tagged commit; `git_dirty=false`
- [ ] `training_rows` = 2,925; `schema_hash` matches `schema.yaml`

**Metric verification**
- [ ] Holdout log-RMSE ≤ 0.13
- [ ] Holdout MAPE ≤ 10%
- [ ] Clear margin over both baselines in CV and on the holdout (IN-11)
- [ ] Temporal diagnostic recorded

**API verification**
- [ ] Local `make serve` with the release artifact starts without the override flag
- [ ] Example prediction returns 200 with version `X.Y.Z`
- [ ] `/model-info` equals `metadata.json`

**Docker verification**
- [ ] `make docker-test VERSION=X.Y.Z` passes with the release artifact
- [ ] Image label = `X.Y.Z`; process user is not root
- [ ] Tag `X.Y.Z` does not already exist in GHCR

**Documentation verification**
- [ ] README quickstart verified on a clean clone
- [ ] Model card and data card complete; numbers ready to copy from `metadata.json`

## 10.3 After R12

**Deployment verification**
- [ ] Render deploy succeeded; health check passing
- [ ] `https://<service>.onrender.com/health` → 200
- [ ] `/model-info` version = `X.Y.Z`
- [ ] Example payload returns the same price as the local release container
- [ ] Render logs show `service.ready` and one `prediction.completed` line for the test request
- [ ] Release notes contain the verification output; README links to the live service
