# DOC-01: Product Requirements and Acceptance Criteria

**Project:** House Price Prediction — End-to-End ML Regression System
**Document ID:** DOC-01
**Version:** 1.2
**Status:** Approved baseline — revision 1.2 (dataset alignment: the project uses the full 2,930-row Ames file; see `docs/data_card.md`)
**Date:** 2026-09-28
**Approved project change (M11):** the batch prediction maximum is **20 records** (API batch endpoint and batch CLI), superseding the previous 100-record limit; see FR-043, FR-050, NFR-017, AC-051 and ADR-15.
**Authoritative source:** `house-price-prediction-adr.md` (ADR-000: Foundational Decisions, ADR-01 to ADR-20)
**Related documents:** DOC-02 Data Understanding and EDA

---

## How to Read This Document

This document turns the approved Architecture Decision Record (ADR) into requirements that can be built, tested, and audited. It does not make new architectural decisions. Every requirement cites the ADR entry it comes from, in the form `[ADR-xx]`.

In a few places the ADR states an intent in qualitative terms (for example, "beat baselines by a clear margin"), or leaves a detail unspecified that a test needs (for example, a numerical tolerance). To make acceptance criteria objectively testable, this document gives such statements an operational definition. Each of these is labelled **Interpretation note (IN-xx)**, explains why it is needed, and uses only mechanisms the ADR already approved.

**Governing principle:** *these criteria operationalize ADR intent; they do not change ADR architecture.* If an interpretation ever appears to conflict with the ADR, the ADR wins and the interpretation is corrected.

**Two kinds of criteria.** Every requirement and acceptance criterion in this document is one of:

- **ADR-direct:** the threshold, count, tool, or procedure is stated in the ADR itself (for example: the scope rule, the 80/20 split, 5×3 repeated CV, the tuning budgets, holdout log-RMSE ≤ 0.13, MAPE ≤ 10%, 80% coverage, batch limit of 20 (project change, M11; previously 100), Python 3.12 slim, non-root container). These carry no label.
- **Interpretation:** DOC-01 adds precision the ADR does not literally specify (a numeric tolerance, a boundary case, a verification procedure, or an operational definition of a qualitative phrase). These carry an **IN-xx** tag where they appear, and are indexed below.

**Interpretation note index.**

| IN | Where it applies | ADR statement being operationalized | DOC-01 operationalization |
|---|---|---|---|
| IN-01 | FR-003, AC-003 | Data is a contract [ADR-04]; NA means "absent" per the data dictionary [ADR-05] | Explicit, documented missing-token parsing so counts do not depend on library defaults |
| IN-02 | FR-009, AC-011 | Holdout is locked and evaluated once [ADR-09] | EDA analyses relating features to `SalePrice` use the development set only |
| IN-03 | FR-014, AC-018 | Transaction-outcome columns excluded [ADR-02] | `Id` and `PID` also excluded as record identifiers (the ADR does not name them) |
| IN-04 | FR-015, AC-019 | Feature kept "only if a CV ablation shows it doesn't hurt" [ADR-07] | "Doesn't hurt" means: removing the feature does not lower mean CV log-RMSE |
| IN-05 | FR-043, AC-051 | Batch endpoint accepts "up to 20" [ADR-15; project change, M11: supersedes the previous 100-record limit] | Lower bound of 1 property; an empty batch is rejected with 422 |
| IN-06 | AC-007 | 20% stratified holdout [ADR-09] | Holdout size is 20% of the in-scope rows, rounded down or up by the splitter; for 2,925 rows this is exactly 2,340/585 |
| IN-07 | AC-009 | Stratified on binned log price [ADR-09] | Each bin's share differs by at most 2 percentage points between the sets |
| IN-08 | AC-022 | Linear branch scales numeric features [ADR-08] | Scaled columns have mean ≈ 0 and SD ≈ 1 within 1e-6 on the fitting data |
| IN-09 | NFR-001, AC-033 | Fixed seeds, pinned versions, reproducibility [ADR-09, ADR-13, ADR-14, ADR-16] | Same selected model and hyperparameters; CV metrics equal within an absolute tolerance of 1e-6 |
| IN-10 | FR-032, AC-037 | "No strong pattern", "no systematic bias", "no extreme error" [ADR-11] | Required evidence artifacts plus a written, reviewable pass/fail judgement per gate (no numeric threshold is invented) |
| IN-11 | Section 8.5, AC-041 | Beat baselines "by a clear margin" [ADR-12] | CV margin greater than one SE of the baseline's estimate, and lower holdout log-RMSE than each baseline |
| IN-12 | Section 8.5, FR-033, AC-038, AC-041 | Holdout evaluated once [ADR-09, ADR-11] | Separates the single final evaluation of the selected model from one-time, non-decisional baseline reference scoring |
| IN-13 | NFR-017 | No latency target; no load testing [ADR-15, ADR-18] | Latency measured and logged; no SLO imposed |
| IN-14 | AC-062 | Batch CLI validates against the API contract [ADR-15] | A file containing any invalid row is rejected as a whole, with the invalid rows reported |

**Requirement language.** "Must" means mandatory for release. "Should" does not appear in the requirement catalogs: everything listed is required.

**Terminology used throughout.**

| Term | Meaning |
|---|---|
| Raw dataset | The full Ames Housing file (De Cock), `data/raw/train.csv` (2,930 rows, 82 columns); used in place of the 1,460-row Kaggle `train.csv` named in ADR-01, by project-owner decision recorded in the data card [ADR-01] |
| In-scope dataset | The raw dataset after the scope rule `GrLivArea ≤ 4000` is applied (2,925 rows) [ADR-06] |
| Development set | The 80% stratified portion of the in-scope dataset used for EDA of target relationships, CV, tuning, and selection [ADR-09] |
| Holdout set | The 20% stratified portion of the in-scope dataset, locked; the selected model is evaluated on it exactly once [ADR-09] |
| Final holdout evaluation | The single evaluation of the selected model on the holdout set; it produces the reported performance and gates the release [ADR-09, ADR-11] |
| Baseline reference scoring | A one-time computation of the two baselines' holdout scores after selection is final, used only for the comparison in Section 8.5; it drives no decision (IN-12) |
| Log-RMSE | Root mean squared error computed on `log1p(SalePrice)`; the primary metric [ADR-12] |
| Pipeline / artifact | The single fitted object that accepts raw feature rows and returns prices in dollars [ADR-08, ADR-14] |
| Production artifact | The pipeline refit on all 2,925 in-scope rows after the final holdout evaluation, saved with its metadata, and served by the API [ADR-11, ADR-14] |
| Candidate | One of the approved models or the approved blend [ADR-10] |

---

# 1. Executive Summary

## 1.1 Project Purpose

The project builds a complete machine learning system that estimates the sale price of a residential property from its structural, quality, location, and other attributes. It covers the full path from a raw CSV file to a deployed, versioned, tested prediction service.

The project is primarily a learning vehicle. Its purpose is to build and demonstrate real ML engineering skill: leakage-free preprocessing, honest validation, disciplined model comparison, reproducible packaging, and production-style serving [ADR-03]. Leaderboard-maximizing techniques are deliberately excluded where they would reduce clarity or realism [ADR-02, ADR-07, ADR-10].

## 1.2 Business Objective

Provide a pricing-guidance tool, a simple Automated Valuation Model (AVM), that returns an estimated fair market value for a residential property using only information available before a sale closes [ADR-02]. The estimate must be accurate in relative terms across cheap and expensive homes, which is why the model is trained and judged in log space [ADR-02, ADR-12].

## 1.3 User Objective

A user (for example, an agent or seller in the simulated business context) submits the full set of property attributes and receives:

- an estimated sale price in dollars,
- the version of the model that produced it, and
- a warning flag if the property falls outside the population the model is certified for (living area above 4,000 sq ft) [ADR-06, ADR-15].

A technical user can also score a CSV file of many properties offline through a command-line entry point [ADR-15].

## 1.4 Expected System Outcome

At release, the project delivers:

1. A validated, reproducible training system that turns the raw dataset into one versioned artifact [ADR-04, ADR-08, ADR-14, ADR-20].
2. A documented, statistically justified model choice with a single, unbiased holdout performance estimate [ADR-09, ADR-11, ADR-12].
3. A containerized FastAPI service deployed on Render that serves that artifact [ADR-15].
4. An automated test suite and CI pipeline that guard against ML-specific failure modes such as leakage and training–serving skew [ADR-16, ADR-18].
5. Documentation that explains what was built, why, and where its limits lie [ADR-19].

---

# 2. Problem Statement

## 2.1 The Problem Being Solved

Estimating what a house will sell for is hard. Price depends on dozens of interacting attributes: size, build quality, age, location, basement and garage characteristics, and more. Human estimates are inconsistent between people, slow to produce, and hard to audit. The problem this project solves is:

> Given the recorded attributes of a residential property in Ames, Iowa (2006–2010 sales), produce a consistent, reproducible, and auditable estimate of its market sale price, with a known and honestly measured error rate.

## 2.2 Why Residential Property Valuation Matters

A house is usually the largest asset a household owns and the largest transaction it makes. Valuation errors have real consequences:

- **Sellers** who underprice lose money; those who overprice see their listing sit unsold.
- **Buyers** risk overpaying.
- **Lenders** base loan amounts on appraised value, so valuation errors become credit risk.
- **Tax authorities** assess property tax on estimated value, so errors create unfair tax burdens.

Because errors matter in proportion to the price of the home (a $20,000 error is serious on a $100,000 house and minor on a $600,000 house), the useful measure of quality is **relative** error. This drives the choice of log-space modeling and the primary metric [ADR-02, ADR-12].

## 2.3 Why Machine Learning Is Appropriate

Machine learning fits this problem for four reasons:

1. **The relationship is learnable from data.** Historical sales pair property attributes with actual prices. This is exactly the setting supervised regression is designed for [ADR-02].
2. **The relationship is complex.** Value depends on many attributes and their interactions (for example, quality matters more in larger homes). Hand-written rules cannot capture this reliably. Regularized linear models and gradient-boosted trees can [ADR-10].
3. **Consistency and auditability.** A trained model gives the same answer for the same input every time, and its behavior can be measured on held-out data [ADR-09, ADR-12].
4. **Measurable baselines exist.** Simple heuristics (the median price; a two-feature linear model) give a concrete bar the ML system must clear, which makes the value of ML explicit rather than assumed [ADR-10, ADR-12].

ML is also bounded here: the data covers one city and five years, so the system is a learning demonstration, not a real-world valuation product [ADR-01, ADR-19].

---

# 3. Project Vision

## 3.1 The Final System

The final system has two halves joined by one artifact [ADR-20].

**Offline training system.** A single command takes the raw CSV, verifies its hash, validates it against a schema, applies the scope rule, performs a stratified split, runs repeated cross-validation and hyperparameter tuning for every approved candidate, logs everything to MLflow, selects a model with the one-standard-error rule and diagnostic gates, evaluates it once on the holdout, refits it on all in-scope data, and saves it with full provenance metadata [ADR-04, ADR-06, ADR-09, ADR-11, ADR-13, ADR-14, ADR-16].

**Online serving system.** A Docker image, with the artifact baked in, runs a FastAPI service on Render. It validates requests against the same data contract used in training, calls the pipeline, returns a dollar prediction with the model version and an out-of-domain flag, and writes a structured log line for every request [ADR-15].

**Quality gates in between.** GitHub Actions runs linting, type checking, the test suite, a smoke training run, and a Docker build on every change [ADR-16, ADR-18].

## 3.2 What Success Looks Like

Success has three dimensions.

**Model quality.** The selected model clearly beats both baselines and reaches a holdout log-RMSE of at most 0.13 and a MAPE of at most 10% [ADR-12].

**Engineering quality.** Training is reproducible from a clean checkout. The same raw row gives the same prediction whether scored through the pipeline directly or through the API. Every artifact can be traced to its code, data, and environment [ADR-08, ADR-14, ADR-18].

**Learning quality.** Every design choice is documented with its reasoning, and the documentation is good enough for a reviewer or interviewer to understand why the system was built this way [ADR-03, ADR-19].

---

# 4. Scope Definition

This section reproduces the scope fixed by ADR-03 exactly.

## In Scope

| # | Item | Source |
|---|---|---|
| S-1 | Data validation (ingestion schema and API contract) | ADR-03, ADR-04 |
| S-2 | Exploratory data analysis | ADR-03, ADR-19 |
| S-3 | Leakage-safe preprocessing pipeline | ADR-03, ADR-05, ADR-08 |
| S-4 | Domain feature engineering | ADR-03, ADR-07 |
| S-5 | Cross-validated model comparison | ADR-03, ADR-09, ADR-10 |
| S-6 | Hyperparameter tuning | ADR-03, ADR-13 |
| S-7 | Locked holdout evaluation | ADR-03, ADR-09, ADR-11 |
| S-8 | Model persistence with metadata | ADR-03, ADR-14 |
| S-9 | FastAPI prediction service | ADR-03, ADR-15 |
| S-10 | Docker packaging | ADR-03, ADR-15 |
| S-11 | Cloud deployment (Render) | ADR-03, ADR-15 |
| S-12 | Experiment tracking (MLflow) | ADR-03, ADR-16 |
| S-13 | Automated tests | ADR-03, ADR-18 |
| S-14 | Continuous integration | ADR-03, ADR-16 |
| S-15 | Documentation | ADR-03, ADR-19 |

The optional Kaggle submission using `test.csv` [ADR-01] no longer applies: the full 2,930-row file already contains the Kaggle `test.csv` properties with their prices, so a submission would not be an independent check. Kaggle files are not used in any role.

## Out of Scope

| # | Item | Reason (from ADR) |
|---|---|---|
| O-1 | Deep learning | Poor fit for about 2.9k tabular rows [ADR-03, ADR-10] |
| O-2 | AutoML | Hides the learning the project exists for [ADR-03] |
| O-3 | Stacking meta-learners | Overfitting risk and opacity; a fixed-weight blend covers ensembling [ADR-03, ADR-10] |
| O-4 | Kubernetes | Infrastructure overhead with no benefit at this size [ADR-03, ADR-15] |
| O-5 | Feature stores | No online features or multi-team reuse [ADR-03] |
| O-6 | Orchestration frameworks (Airflow, Prefect, Kubeflow) | The pipeline is one command [ADR-03, ADR-16] |
| O-7 | Automated retraining | Static dataset [ADR-03] |
| O-8 | Live drift monitoring | No live traffic; structured logs are the groundwork [ADR-03, ADR-16] |
| O-9 | Frontend UI | The API and its OpenAPI docs are the interface [ADR-03, ADR-15] |
| O-10 | Reduced-input "quick estimate" model | Explicitly excluded [ADR-15] |
| O-11 | Load or performance testing | No traffic worth simulating [ADR-18] |
| O-12 | Real-world valuation use | Data is one city, 2006–2010 [ADR-01, ADR-19] |

---

# 5. Stakeholders

| Stakeholder | Who they are | What they need from the system | How the project serves them |
|---|---|---|---|
| **End users** | Agents or sellers in the simulated AVM context; technical users calling the API or CLI | An accurate price estimate, clear errors for bad input, and a warning when the property is outside the model's domain | `/predict`, `/predict/batch`, 422 validation errors, `out_of_domain` flag, CLI batch scoring [ADR-06, ADR-15] |
| **Developers** | The project author (a 2nd-year AIML student) and any contributor | A clear structure, reproducible environment, fast feedback on mistakes | `src/` layout, uv lockfile, Make targets, pre-commit, CI, tests [ADR-16, ADR-17, ADR-18] |
| **Project reviewers** | Technical reviewers auditing correctness | Evidence that there is no leakage, that validation is honest, and that decisions are justified | Leakage tests, locked holdout, ADRs, MLflow history, model card [ADR-09, ADR-18, ADR-19] |
| **Evaluators** | Internship evaluators and interviewers | A quick, credible picture of skills and results | README results summary, architecture diagram, model card, live deployment [ADR-15, ADR-19] |
| **Future maintainers** | Anyone extending or re-running the project later | To know why things are as they are and how to change them safely | MADR ADR log, data card, config-driven feature groups, pinned versions, metadata [ADR-08, ADR-14, ADR-19] |

---

# 6. Functional Requirements

Requirements describe **what** the system must do, not how. Tool names appear only where the ADR itself fixes the tool.

## 6.1 Data Ingestion

**FR-001 — Raw data loading.** The system must load the raw dataset from the designated raw-data location and must never modify, overwrite, or re-save that file. [ADR-01, ADR-04]

**FR-002 — Data integrity verification.** On every training run, the system must compute the SHA-256 hash of the raw dataset, compare it with the hash committed to the repository, and stop with an explicit error if they differ. [ADR-01, ADR-04]

**FR-003 — Consistent missing-value parsing.** The ingestion step must parse the raw file with an explicit, documented definition of which text tokens count as missing, so that the observed missingness of every column is the same regardless of library defaults. [ADR-04, ADR-05]

*Interpretation note (IN-01):* the raw file encodes missing values both as the text `NA` and as empty cells, and also contains the literal category `None` (in `MasVnrType`). Some versions of common CSV parsers treat `None` as missing by default, which silently changes missing-value counts. The ADR requires data to be treated as a contract [ADR-04]; this requirement makes that contract independent of parser behavior. Because the semantic filler maps both cases to `"None"` [ADR-05], model behavior is unaffected, but reported counts and schema checks must be stable.

## 6.2 Data Validation

**FR-004 — Ingestion schema validation.** The system must validate the raw dataset against a Pandera schema that checks: presence of all expected columns; column data types; allowed category values taken from the data dictionary; numeric ranges; uniqueness of the identifiers `Id` and `PID`; and `SalePrice > 0`. Any violation must stop the run and report every failing check. [ADR-04]

**FR-005 — Single source of allowed values.** The allowed category values and ranges used by the ingestion schema and by the API request schema must be derived from one shared configuration, so the two contracts cannot drift apart. [ADR-04, ADR-16]

**FR-006 — Scope rule application.** Before any split, the system must remove rows with `GrLivArea > 4000` and must record the number of rows removed and their `Id` values in the run's logged outputs. [ADR-06]

**FR-007 — Stratified development/holdout split.** The system must split the in-scope dataset into an 80% development set and a 20% holdout set, stratified on binned `log1p(SalePrice)`, using a fixed seed. The split must be written to disk once and reused unchanged by all later runs. [ADR-09]

## 6.3 Exploratory Data Analysis

**FR-008 — EDA coverage.** The project must produce the EDA deliverables defined in DOC-02, Section 13: target analysis, missing-value analysis, numerical and categorical feature analysis, outlier investigation, leakage assessment, and feature-engineering assessment. [ADR-07, ADR-19]

**FR-009 — Holdout protection during EDA.** Any analysis that relates features to `SalePrice` for the purpose of informing modeling choices must use the development set only. Data-quality profiling that does not use the target (schema, types, missingness) may use the full raw dataset. [ADR-09]

*Interpretation note (IN-02):* ADR-09 locks the holdout and permits exactly one evaluation. Studying holdout prices during EDA would let holdout information shape feature choices, which quietly defeats that lock. The ADR-06 scope rule is the single exception, because it is a pre-split scope definition already fixed by the ADR from the dataset author's documentation. Checking the split's target balance (AC-009) is not a feature–target analysis and informs no modeling choice, so it is also permitted.

**FR-010 — Notebook discipline.** EDA notebooks must import any reusable logic from the project package and must not contain logic that training or serving depend on. [ADR-17]

## 6.4 Feature Engineering

**FR-011 — Semantic missing-value filling.** The system must fill values that the data dictionary defines as "feature absent" with the category `"None"` for categorical columns (`PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `MiscFeature`, the `Garage*` categoricals, the `Bsmt*` categoricals, `MasVnrType`) and with `0` for the related numeric columns (`GarageArea`, `GarageCars`, basement square-footage and basement bathroom columns, `MasVnrArea`). This step must be stateless: it learns nothing from data. [ADR-05]

**FR-012 — Approved engineered features.** The system must compute the approved engineered features: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `IsRemodeled`, `TotalPorchSF`, `HasPool`, `HasGarage`, `HasBsmt`, `HasFireplace`, `Has2ndFlr`, and `GarageAge`. `GarageYrBlt` must not be imputed; it is replaced by `HasGarage` and `GarageAge`, with `GarageAge = 0` when there is no garage. The step must be stateless. [ADR-05, ADR-07]

**FR-013 — Ordinal and categorical recoding.** The system must map quality and condition scales to ordered integers (None = 0, Po = 1, Fa = 2, TA = 3, Gd = 4, Ex = 5) and must treat `MSSubClass` as a categorical feature rather than a number. [ADR-07]

**FR-014 — Leakage exclusions.** The system must exclude `SaleType` and `SaleCondition` from model features, while keeping the rows that contain them. `Id` and `PID` must also be excluded, as they are record identifiers, not property attributes *(Interpretation IN-03)*. `YrSold` must be kept as the valuation-year input. [ADR-02]

**FR-015 — Feature ablation.** Each approved engineered feature must be assessed with a cross-validation ablation (the model scored with and without it). A feature is retained only if its removal does not improve (lower) the mean CV log-RMSE *(Interpretation IN-04)*. The ablation results must be recorded. [ADR-07]

**FR-016 — Excluded techniques.** The feature pipeline must not use target encoding, polynomial feature expansion, PCA, or automated feature generation. [ADR-07]

## 6.5 Preprocessing

**FR-017 — Single end-to-end pipeline.** Each trained model must be a single object containing, in order: the semantic missing-value filler, the feature engineer, a model-family-specific column transformer, and the estimator, all wrapped in a target transformer that applies `log1p` when fitting and `expm1` when predicting. The object must accept raw feature rows and return predictions in dollars. [ADR-08]

**FR-018 — Linear-family preprocessing.** For linear models, the column transformer must apply statistical imputation, a Yeo-Johnson power transform and standard scaling to numeric columns, and one-hot encoding that ignores unseen categories to nominal columns. [ADR-08]

**FR-019 — Tree-family preprocessing.** For tree-based models, the column transformer must apply statistical imputation and ordinal encoding that maps unseen categories to −1, with no scaling or power transform. [ADR-08]

**FR-020 — Fitted imputation inside the pipeline.** Values that are genuinely unknown (for example `LotFrontage`, `Electrical`, and any unexpected missing value at serving time) must be imputed inside the pipeline: median plus a missing-indicator column for numeric features, most frequent value for categorical features. Imputation statistics must be learned only from the data the pipeline is fitted on. Missing values in the columns listed in FR-011 are handled by FR-011 under the approved rule, whatever their cause; the documented basement anomalies are recorded as described in DOC-02 Section 6.6, without changing this rule. [ADR-05, ADR-08]

**FR-021 — Configuration-driven feature groups.** Feature groups (numeric, nominal, ordinal, dropped) must be declared in YAML configuration, not hard-coded. [ADR-08, ADR-16]

## 6.6 Model Training

**FR-022 — Candidate set.** The system must train and evaluate exactly the approved candidates: `DummyRegressor` (median); a linear regression on `OverallQual` and `GrLivArea`; Ridge; Lasso; Random Forest; LightGBM; and a fixed-weight average of the best linear model and LightGBM. [ADR-10]

**FR-023 — Training entry point.** Training, evaluation, and prediction must be runnable through command-line entry points and corresponding Make targets. [ADR-16, ADR-17]

## 6.7 Cross-Validation

**FR-024 — Repeated stratified CV.** Model comparison must use 5-fold cross-validation repeated 3 times (15 fits) on the development set, stratified by price-quantile bins, with fixed seeds. [ADR-09]

**FR-025 — CV reporting.** For each candidate, the system must report the mean log-RMSE and its standard error across the 15 folds. [ADR-09, ADR-11]

## 6.8 Hyperparameter Tuning

**FR-026 — Linear tuning.** Ridge and Lasso must be tuned over a log-spaced grid of `alpha` values using the standard CV scheme. [ADR-13]

**FR-027 — Random Forest tuning.** Random Forest must be tuned with Optuna using about 30 trials. [ADR-13]

**FR-028 — LightGBM tuning.** LightGBM must be tuned with Optuna's TPE sampler for 100 trials over learning rate, number of leaves, minimum child samples, number of estimators, subsample, column sample, and L1/L2 regularization, optimizing mean repeated-CV log-RMSE. [ADR-13]

**FR-029 — Tuning isolation.** All tuning must use only the development set, with fixed seeds. The holdout set must never be read during tuning. [ADR-09, ADR-13]

## 6.9 Model Selection

**FR-030 — One-standard-error rule.** The system must rank candidates by mean CV log-RMSE and select the simplest candidate whose mean is within one standard error of the best, using the simplicity order: linear < Random Forest < LightGBM < blend. [ADR-11]

**FR-031 — Blend admission rule.** The blend may be selected only if its mean CV log-RMSE is lower than the best single model's by more than one standard error. [ADR-11]

**FR-032 — Diagnostic gates.** Before approval, the selected model must pass three diagnostic reviews on out-of-fold development predictions: (a) residuals show no strong pattern against predicted value; (b) no price decile shows systematic bias; (c) no neighborhood shows extreme error. Results must be recorded as the evidence artifacts and written judgements specified in AC-037 *(Interpretation IN-10)*. [ADR-11]

## 6.10 Holdout Evaluation

**FR-033 — Single holdout evaluation.** The selected model, fitted on the full development set, must be evaluated on the holdout set exactly once (the final holdout evaluation). The resulting log-RMSE, MAE in dollars, MAPE, and R² are the reported performance and the basis for release gating. This evaluation may take place only after the selection record (FR-030 to FR-032) is final. [ADR-09, ADR-11, ADR-12]

Baseline reference scoring, as defined in Section 8.5, is not a second evaluation of any selectable model: it is computed once, after selection is final, and is never used for selection, tuning, or any iterative decision *(Interpretation IN-12)*.

**FR-034 — Production refit.** After holdout evaluation, the production artifact must be refit on all 2,925 in-scope rows using the selected hyperparameters. [ADR-11]

## 6.11 Temporal Diagnostic Evaluation

**FR-035 — Temporal robustness check.** The system must train the selected model configuration on 2006–2009 sales and evaluate it on 2010 sales, and report the result as a robustness diagnostic. This result must not influence model selection. [ADR-09]

## 6.12 Experiment Tracking

**FR-036 — Run logging.** Every CV evaluation and every tuning trial must be logged to MLflow (local file backend) with hyperparameters, metrics, fixed seed, data hash, and git commit SHA. [ADR-13, ADR-14, ADR-16]

**FR-037 — Artifact logging.** The final production artifact and its metadata must be logged to MLflow. [ADR-14]

## 6.13 Artifact Persistence

**FR-038 — Artifact and metadata.** The system must save the full fitted pipeline as one joblib artifact together with a `metadata.json` file containing: model version (semantic versioning), git commit SHA, training data SHA-256 hash, Python and library versions, hyperparameters, CV and holdout metrics, input schema, and training timestamp. [ADR-14]

**FR-039 — Faithful reload.** A reloaded artifact must produce predictions identical to the in-memory pipeline that was saved. [ADR-14, ADR-18]

## 6.14 FastAPI Serving

**FR-040 — Health endpoint.** `GET /health` must report service liveness. [ADR-15]

**FR-041 — Model information endpoint.** `GET /model-info` must return the loaded model's metadata. [ADR-15]

**FR-042 — Single prediction.** `POST /predict` must accept one property in the full raw feature schema and return the predicted price in dollars, the model version, and the `out_of_domain` flag. [ADR-15]

**FR-043 — Batch prediction endpoint.** `POST /predict/batch` must accept between 1 and 20 properties *(lower bound: Interpretation IN-05; upper bound: project change, M11: supersedes the previous 100-record limit)* and return one prediction per property in input order, with the same fields as the single endpoint. Requests with more than 20 properties must be rejected. [ADR-15]

**FR-044 — Request validation.** Requests must be validated against the full raw schema: fields that are legitimately NA in the dataset are nullable; all others are required; categorical fields are restricted to allowed values; numeric fields are range-checked. Invalid requests must receive HTTP 422 with a description of the problem. [ADR-04, ADR-15]

**FR-045 — Out-of-domain flag.** Properties with `GrLivArea > 4000` must still receive a prediction, with `out_of_domain` set to `true`. All other properties must have `out_of_domain` set to `false`. [ADR-06, ADR-15]

**FR-046 — Model loading.** The service must load the model once at startup, not per request. [ADR-15]

**FR-047 — Structured request logging.** Each prediction request must produce one structured JSON log line containing an inputs hash, the prediction, the latency, and the model version. [ADR-15]

## 6.15 Docker Deployment

**FR-048 — Container image.** The service must be packaged as one Docker image based on Python 3.12 slim, running as a non-root user, with the model artifact baked into the image. [ADR-15]

**FR-049 — Cloud deployment.** The image must be deployed to Render as a web service reachable over HTTPS. [ADR-15]

## 6.16 Batch Prediction

**FR-050 — Offline batch scoring.** A command-line entry point must read a CSV of properties in the raw schema, validate it against the same contract as the API, and write predictions (with `out_of_domain` flags) to an output file. A file may contain 1 to 20 rows, the same limit as the batch endpoint *(project change, M11: supersedes the previous 100-record limit)*. [ADR-04, ADR-15]

## 6.17 Documentation

**FR-051 — README.** The repository must include a README with the problem statement, a results summary, an architecture diagram, and a quickstart covering setup, training, serving, and calling the API. [ADR-19]

**FR-052 — ADR log.** `docs/adr/` must contain one MADR-format file per ADR decision, with ADR-000 as the foundation. Superseded ADRs must be marked, not deleted. [ADR-19]

**FR-053 — Data card.** A data card must document the source, license, known issues, the scope rule, and missing-value semantics. [ADR-19]

**FR-054 — Model card.** A model card in the Mitchell et al. format must document intended use, out-of-scope uses, metrics, limitations, and the fact that the production model was refit on all data after holdout evaluation. [ADR-11, ADR-19]

**FR-055 — API documentation.** The service must expose auto-generated OpenAPI documentation that includes a complete example request payload. [ADR-15, ADR-19]

**FR-056 — Code documentation.** Public functions and classes in the package must have Google-style docstrings. [ADR-19]

**FR-057 — EDA narrative.** An EDA report notebook must state conclusions that motivate each feature and preprocessing choice, as specified in DOC-02. [ADR-19]

## 6.18 Testing and CI

**FR-058 — Test suite.** The project must include unit, schema, pipeline, leakage, model quality gate, API, training–serving consistency, and smoke-training tests. [ADR-18]

**FR-059 — Continuous integration.** GitHub Actions must run, on every push and pull request: linting, type checking, the test suite, a smoke training run, and a Docker image build. [ADR-16, ADR-18]

**FR-060 — Local quality hooks.** pre-commit hooks must run Ruff, mypy, and file-hygiene checks before commits. [ADR-16]

---

# 7. Non-Functional Requirements

## 7.1 Reproducibility

**NFR-001 — Reproducible training.** Given the same code commit, raw data hash, configuration, declared environment (the same uv lockfile and Python version), random seeds, candidate set, and tuning configuration, a full training run must produce: identical development and holdout splits; the same selected model; the same selected hyperparameters; and CV metrics that agree with the reference run within the documented numerical tolerance. [ADR-09, ADR-13, ADR-14, ADR-16]

*Interpretation note (IN-09):* fixed seeds and pinned versions make training reproducible in intent, but numerical libraries can still produce tiny floating-point differences between runs (for example from thread scheduling or low-level linear-algebra routines). Demanding bit-identical scores would make the criterion fail for reasons unrelated to project correctness. DOC-01 therefore requires exact agreement for everything that is discrete (splits, selected model, selected hyperparameters) and agreement within an **absolute tolerance of 1e-6 on log-RMSE** for every per-fold CV score and every candidate's mean CV score. The tolerance is recorded in configuration (NFR-020) and applied by the reproducibility check. This does not relax reproducibility: any difference larger than the tolerance, or any change in a discrete outcome, is a failure.

**NFR-002 — Locked environment.** All Python dependencies must be pinned exactly in a uv lockfile under Python 3.12, and training and serving must both use that lockfile. [ADR-14, ADR-16]

**NFR-003 — Clean-checkout reproducibility.** A new contributor must be able to go from a clean clone to a trained artifact using only the README quickstart and documented Make targets, given the raw CSV. [ADR-16, ADR-19]

## 7.2 Reliability

**NFR-004 — Fail-fast data handling.** Invalid or altered data must stop training with an explicit error rather than producing a model. [ADR-04]

**NFR-005 — Graceful request handling.** Invalid API input must yield HTTP 422 and must never crash the service or produce a prediction. [ADR-15]

**NFR-006 — Unseen categories.** A category value that passes API validation but never appeared during training must not cause an error; the pipeline must handle it through its unseen-category settings. [ADR-08]

**NFR-007 — Valid output.** Every prediction must be a finite, positive dollar amount. [ADR-08, ADR-18]

## 7.3 Maintainability

**NFR-008 — Package structure.** All training and serving logic must live in the `src/house_price` package following the module layout in ADR-17. [ADR-17]

**NFR-009 — Code quality.** Code in `src/` must pass Ruff linting and formatting and mypy type checking with zero errors. [ADR-16]

**NFR-010 — Estimator compatibility.** Custom transformers must follow the scikit-learn estimator API, so that they can be cloned, fitted, and serialized like built-in transformers. [ADR-08]

## 7.4 Versioning

**NFR-011 — Semantic model versions.** Every production artifact must carry a semantic version. The same version must appear in its metadata, in `/model-info`, and in every prediction response. [ADR-14, ADR-15]

**NFR-012 — Immutable releases.** Each Docker image must contain exactly one model artifact, so an image version identifies a model release. [ADR-15]

**NFR-013 — Decision versioning.** Changes to any ADR decision must be recorded as a new ADR that marks the old one superseded. [ADR-19]

## 7.5 Testing

**NFR-014 — Coverage.** Test coverage of `src/` must be at least 80%. Notebooks are excluded. [ADR-18]

**NFR-015 — CI gating.** A change must not be merged to the main branch unless every CI stage passes. [ADR-16, ADR-18]

## 7.6 Performance

**NFR-016 — Laptop-scale training.** The complete training workflow, including all tuning budgets, must run on a standard laptop without specialized hardware, with each tuning study completing in minutes. [ADR-13]

**NFR-017 — Serving efficiency.** The model must be loaded once at startup, batch requests are capped at 20 properties *(project change, M11: supersedes the previous 100-record limit)*, and per-request latency must be measured and logged. [ADR-15]

*Interpretation note (IN-13):* the ADR sets no latency target and excludes load testing [ADR-15, ADR-18]. This document therefore requires latency to be measured and recorded, but does not impose a latency SLO. Adding one would be a new architectural decision.

**NFR-018 — Fast CI feedback.** The CI smoke training run must use a small data sample so that CI checks integration without running full tuning. [ADR-18]

## 7.7 Configuration Management

**NFR-019 — Validated configuration.** Feature groups, schema rules, model settings, and tuning settings must be defined in YAML and loaded into validated Pydantic models. Invalid configuration must fail at startup with a clear error. [ADR-16]

**NFR-020 — No hidden constants.** Values that define behavior (seeds, split ratio, fold counts, trial budgets, the scope threshold, the batch limit, the reproducibility tolerance) must come from configuration, not be scattered as literals in code. [ADR-08, ADR-16]

## 7.8 Documentation Quality

**NFR-021 — Decision rationale.** Every documented decision must state its reasoning, tradeoffs, and rejected alternatives. [ADR-19]

**NFR-022 — Honest limitations.** The model card and data card must explicitly state out-of-scope uses and known limitations, including geographic and temporal limits and the domain boundary of 4,000 sq ft. [ADR-06, ADR-19]

**NFR-023 — Documentation freshness.** API documentation must be generated from the service code so it cannot drift from the implementation. [ADR-19]

## 7.9 Deployment Consistency

**NFR-024 — No training–serving skew.** The service must produce predictions only by calling the persisted pipeline on raw input. No preprocessing may be reimplemented in serving code. [ADR-08, ADR-15]

**NFR-025 — Container security baseline.** The container must run as a non-root user, and the service must only load artifacts produced by this project's training system. [ADR-14, ADR-15]

**NFR-026 — Portable image.** The deployed image must be a standard container that runs identically locally and on Render. [ADR-15]

## 7.10 Traceability

**NFR-027 — Artifact provenance.** Every artifact must be traceable to its git commit, data hash, library versions, and hyperparameters through its metadata and MLflow record. [ADR-14]

**NFR-028 — Requirement traceability.** Every functional requirement must map to at least one acceptance criterion (Section 10), and every requirement must cite its ADR source. [ADR-19]

**NFR-029 — Prediction traceability.** Every served prediction must be linkable to a model version through its response and its log line. [ADR-15]

---

# 8. Success Metrics

All metrics follow ADR-12.

## 8.1 Primary Metric

**Log-RMSE**: the root mean squared error between `log1p(predicted price)` and `log1p(actual price)`. It approximates the RMSLE used by the Kaggle competition.

Why it is primary: it measures relative error, so a 10% miss counts about the same on a cheap house as on an expensive one. This matches how pricing errors actually matter (Section 2.2). It is also what every candidate is optimized for, so the metric used for selection and the metric used for training agree [ADR-02, ADR-12].

Rough intuition: for small values, a log-RMSE of 0.13 corresponds to a typical relative error of roughly 13%.

## 8.2 Secondary Metrics

Reported, never used for selection [ADR-12]:

| Metric | Purpose |
|---|---|
| MAE (dollars) | Translates error into money: "typically off by $X" |
| MAPE | Average percentage error, easy for non-technical readers |
| R² | Share of price variance explained; familiar to reviewers |

## 8.3 Diagnostics

Residual-vs-predicted plots, error by price decile, error by neighborhood, and predicted-vs-actual plots [ADR-11, ADR-12].

## 8.4 Target Values

| Criterion | Target | Measured on |
|---|---|---|
| Holdout log-RMSE | ≤ 0.13 | Final holdout evaluation of the selected model [ADR-12] |
| Holdout MAPE | ≤ 10% | Final holdout evaluation of the selected model [ADR-12] |
| Improvement over baselines | Clear margin over both baselines (operational definition below, IN-11) | Development-set CV, plus final holdout evaluation compared with baseline reference scoring [ADR-12] |

## 8.5 Baseline Expectations

Two baselines define the floor [ADR-10]:

1. **Median baseline** (`DummyRegressor`): predicts the development-set median price for every house. This is the "no model" reference.
2. **Heuristic baseline**: linear regression on `OverallQual` and `GrLivArea`, the two most intuitive price drivers. This represents what a simple rule of thumb achieves.

A useful ML system must beat the heuristic baseline, not just the median. If the tuned models could not clearly beat a two-feature regression, the added complexity would not be justified.

**Interpretation note (IN-11) — "clear margin."** ADR-12 requires beating both baselines "by a clear margin" but gives no number. To make this testable, this document defines it using the statistical framework ADR-11 already uses: the selected model's mean CV log-RMSE must be lower than each baseline's mean CV log-RMSE by more than one standard error of the baseline's estimate, **and** the selected model's holdout log-RMSE must be lower than each baseline's holdout log-RMSE.

**Interpretation note (IN-12) — how the holdout is used.** The second half of the IN-11 definition needs baseline scores on the holdout. To keep this consistent with ADR-09's rule that the holdout is evaluated exactly once (for the selected model), the two uses are separated:

- **A. Final holdout evaluation (selected model).** The selected model is evaluated on the holdout exactly once. This result is the reported performance and the release gate (AC-038 to AC-041). ADR-direct.
- **B. Baseline reference scoring.** The two baselines' holdout scores are computed once, in the same release step, after the selection record is final. They exist only to make the IN-11 comparison. They:
  - are not used for model selection;
  - are not used to tune any hyperparameter;
  - do not trigger any iterative decision (no model, feature, or setting is changed in response to them);
  - do not constitute repeated holdout selection, because the baselines are not candidates for deployment and selection has already closed.

If the selected model failed a holdout gate, the response would be to record the failure and follow the ADR supersession process, not to re-select using holdout results. This keeps the holdout's single-use guarantee intact.

---

# 9. Acceptance Criteria

Each criterion states an observable result and how it is verified. Criteria are grouped in the order they will naturally be met during implementation, so each group can serve as a milestone exit check.

## 9.1 Milestone A — Data Foundation

**AC-001.** Running the ingestion step leaves the raw CSV byte-identical (its SHA-256 before and after the run is unchanged). *Verify:* automated test comparing hashes.

**AC-002.** If the raw CSV's SHA-256 differs from the committed hash, training stops with a non-zero exit code and an error message naming the mismatch; no artifact or split file is written. *Verify:* test using a modified copy of the file.

**AC-003** *(Interpretation IN-01)*. Ingestion reports the same missing-value count for every column on every run and every supported environment, and the documented count for `MasVnrType` matches the raw file's missing tokens (`NA` and empty cells) only, never the literal text `None`. *Verify:* schema test asserting per-column missing counts.

**AC-004.** The ingestion schema rejects each of the following, each in a separate test: a missing expected column; a non-numeric value in a numeric column; a category value not in the allowed list; a duplicate `Id`; a `SalePrice` ≤ 0. Each rejection names the failing column and check. *Verify:* schema tests (5 cases).

**AC-005.** The ingestion schema and the API request schema read allowed categories and ranges from the same configuration source; changing an allowed value in that configuration changes both. *Verify:* test asserting both schemas expose identical allowed values.

**AC-006.** After the scope rule, the in-scope dataset has exactly 2,925 rows, no row has `GrLivArea > 4000`, and the 5 removed `Id` values are recorded in the run output. *Verify:* automated test plus logged run output.

**AC-007.** The development set has 2,340 rows and the holdout set has 585 rows (totaling 2,925), with no shared `Id`. *Verify:* automated test.

*Interpretation note (IN-06):* ADR-09 fixes a 20% holdout but not exact row counts. In general the split sizes depend on rounding in the stratified splitter, and either rounding is acceptable provided the total and disjointness hold; for 2,925 in-scope rows, 20% is exactly 585, so no rounding variant arises.

**AC-008.** Running the split step twice produces identical development and holdout `Id` sets, and later runs read the persisted split rather than regenerating it. *Verify:* test comparing `Id` sets across runs.

**AC-009** *(Interpretation IN-07)*. The distribution of binned `log1p(SalePrice)` in the holdout set matches the development set: each bin's share differs by no more than 2 percentage points. *Verify:* automated check recorded in the EDA outputs (DOC-02 deliverable E-08). ADR-09 requires stratification but sets no tolerance; 2 percentage points is the DOC-01 operational check that stratification worked. It is a split-integrity check, not a modeling analysis (see FR-009).

## 9.2 Milestone B — EDA

**AC-010.** Every EDA deliverable listed in DOC-02 Section 13 exists in the repository (notebook sections, saved figures in `reports/figures/`, or saved tables). *Verify:* checklist review against DOC-02 Section 13.

**AC-011** *(Interpretation IN-02)*. No EDA notebook reads the holdout file in any analysis that relates features to `SalePrice`, other than the split-balance check (AC-009); and no analysis relating features to `SalePrice` runs on the raw file other than the ADR-06 scope review (DOC-02 E-24 to E-26). *Verify:* code review of notebooks, and a search of notebook sources for the holdout and raw file paths.

**AC-012.** EDA notebooks contain no function or class definitions used by training or serving; all such logic is imported from `src/house_price`. *Verify:* code review.

**AC-013.** The EDA report states a written conclusion for each approved engineered feature and for each preprocessing branch, linking it to the evidence shown. *Verify:* checklist review.

## 9.3 Milestone C — Features and Preprocessing

**AC-014.** On a fixture containing every "feature absent" NA case listed in FR-011, the semantic filler outputs `"None"` in every listed categorical column and `0` in every listed numeric column, and leaves no missing value in those columns. *Verify:* unit test.

**AC-015.** The semantic filler and the feature engineer do not modify their input data frame, produce identical output for identical input regardless of prior calls, and have no fitted state. *Verify:* unit tests (input unchanged after call; fitting on different data does not change output).

**AC-016.** For a hand-computed fixture row, each of the 12 engineered features in FR-012 equals its expected value exactly, including `GarageAge = 0` and `HasGarage = 0` for a property with no garage. *Verify:* unit test.

**AC-017.** Quality/condition columns are mapped as None = 0, Po = 1, Fa = 2, TA = 3, Gd = 4, Ex = 5, and `MSSubClass` is treated as categorical by the column transformer. *Verify:* unit tests.

**AC-018** *(`Id`/`PID` exclusion: Interpretation IN-03)*. The fitted pipeline's input features do not include `SaleType`, `SaleCondition`, `Id`, or `PID`; rows with any value of `SaleType` or `SaleCondition` remain in the training data; `YrSold` is used. *Verify:* pipeline test inspecting consumed columns and row counts.

**AC-019** *(Interpretation IN-04)*. A recorded ablation table lists, for each engineered feature, the mean CV log-RMSE with and without it, and every retained feature satisfies the FR-015 retention rule. *Verify:* review of the logged ablation results in MLflow.

**AC-020.** The pipeline contains no target encoder, polynomial feature generator, PCA step, or automated feature generator. *Verify:* test inspecting pipeline steps; code review.

**AC-021.** Calling `predict` on raw feature rows (with `SalePrice` absent) returns values in dollars, and every prediction on the development fixture is finite and positive. *Verify:* pipeline test.

**AC-022** *(tolerance: Interpretation IN-08)*. The linear branch output contains scaled numeric columns (mean approximately 0 and standard deviation approximately 1 on the fitting data, within 1e-6) and one-hot columns; the tree branch output contains no scaled columns and encodes an unseen category as −1. *Verify:* pipeline tests.

**AC-023.** A row containing a category never seen during fitting, but allowed by the schema, is scored without error by both branches. *Verify:* pipeline test.

**AC-024.** For a numeric column with genuinely unknown values, the fitted pipeline outputs a missing-indicator column, and the imputed value equals the median of the fitting data only. *Verify:* unit test comparing to a manually computed median.

**AC-025 (leakage).** Fitting the pipeline on data A and then transforming data B leaves every fitted statistic (imputation values, scaling parameters, power-transform parameters, encoder categories) unchanged, and those statistics equal the ones obtained by fitting on A alone. *Verify:* leakage test.

**AC-026.** A fitted pipeline can be cloned with scikit-learn's `clone()` and the clone can be fitted and used to predict. *Verify:* pipeline test.

**AC-027.** Feature group membership (numeric, nominal, ordinal, dropped) is read from YAML; moving a column between groups in the YAML changes the pipeline's treatment of that column without code changes. *Verify:* configuration test.

## 9.4 Milestone D — Training, Validation, and Tuning

**AC-028.** A training run produces CV results for all 7 approved candidates and for no others. *Verify:* MLflow run listing.

**AC-029.** Each candidate's CV evaluation consists of exactly 15 fold scores (5 folds × 3 repeats) on the development set, and the run records their mean and standard error. *Verify:* MLflow metrics review; automated test on the CV runner.

**AC-030.** No tuning or CV code path opens the holdout file. *Verify:* test that runs tuning with the holdout file made unreadable and confirms success.

**AC-031.** The Ridge and Lasso tuning records show a log-spaced `alpha` grid; the Random Forest study shows about 30 Optuna trials; the LightGBM study shows exactly 100 TPE trials over the parameters listed in FR-028. *Verify:* MLflow review.

**AC-032.** Every tuning trial and CV run in MLflow records its hyperparameters, metrics, seed, data hash, and git commit SHA. *Verify:* automated check over the MLflow records of one full run.

**AC-033** *(tolerance: Interpretation IN-09)*. Two full training runs with the same code commit, raw data hash, configuration, declared environment (same uv lockfile and Python version), random seeds, candidate set, and tuning configuration produce:
1. identical development and holdout `Id` sets;
2. the same selected model;
3. the same selected hyperparameters (exact equality of every value);
4. per-fold and mean CV log-RMSE for every candidate that agree within an absolute tolerance of 1e-6.

Any discrete difference (items 1–3), or any CV metric difference larger than 1e-6, fails the criterion. *Verify:* reproducibility check that runs training twice and compares the two runs' split files, selection records, and MLflow metrics, reporting the largest observed metric difference.

**AC-034.** Training, evaluation, and prediction can each be started with a single documented Make target. *Verify:* running each target from a clean checkout.

## 9.5 Milestone E — Selection and Evaluation

**AC-035.** The selection record lists every candidate's mean CV log-RMSE and standard error, identifies the best candidate, states the one-standard-error threshold, and shows that the chosen candidate is the simplest one within it under the order linear < Random Forest < LightGBM < blend. *Verify:* review of the selection record; automated test of the selection function on synthetic scores.

**AC-036.** If the blend is selected, the record shows its mean CV log-RMSE is lower than the best single model's by more than one standard error; otherwise the blend is not selected. *Verify:* automated test of the selection function on synthetic scores covering both cases.

**AC-037** *(Interpretation IN-10)*. The diagnostic gate record contains: a residual-vs-predicted plot; mean signed error per price decile; and log-RMSE per neighborhood, all on out-of-fold development predictions; plus a written pass/fail judgement for each of the three gates, with the reasoning stated. ADR-11 describes these gates qualitatively, so DOC-01 requires the evidence and a reviewable judgement rather than inventing numeric thresholds. *Verify:* review of the saved report.

**AC-038.** The selected model receives exactly one final holdout evaluation, recorded as a single MLflow run tagged as the final holdout evaluation, reporting log-RMSE, MAE, MAPE, and R². Baseline reference scoring *(Interpretation IN-12)* is recorded in a separate run tagged as baseline reference, computed once per baseline. Both runs are created after the selection record is final, and no selection, tuning, or configuration change is recorded after them for that release. *Verify:* MLflow review showing exactly one final-evaluation run and at most one baseline-reference run per release, with timestamps later than the selection record.

**AC-039.** The selected model's holdout log-RMSE is ≤ 0.13. *Verify:* holdout run metric; automated quality gate test on stored metrics.

**AC-040.** The selected model's holdout MAPE is ≤ 10%. *Verify:* holdout run metric; automated quality gate test.

**AC-041** *(Interpretation IN-11, IN-12)*. The selected model beats both baselines under the Section 8.5 definition: its mean CV log-RMSE is lower than each baseline's by more than one standard error, and its final-holdout log-RMSE is lower than each baseline's holdout log-RMSE from baseline reference scoring. The baseline reference scores are used only for this comparison. *Verify:* selection record, final holdout evaluation run, and baseline reference run; automated quality gate test.

**AC-042.** The production artifact's metadata records a training row count of 2,925 and the hyperparameters of the selected configuration. *Verify:* metadata inspection; automated test.

**AC-043.** A temporal diagnostic record exists that shows log-RMSE for the selected configuration trained on 2006–2009 and evaluated on 2010, and it is labelled as not used for selection. *Verify:* review of the saved report and MLflow run.

## 9.6 Milestone F — Persistence and Tracking

**AC-044.** The artifact directory contains one joblib model file and one `metadata.json` containing all fields listed in FR-038, each non-empty. *Verify:* automated test on the metadata keys and values.

**AC-045.** The data hash in `metadata.json` equals the committed raw-data hash, and the git SHA equals the commit that produced the artifact. *Verify:* automated test.

**AC-046.** Saving and reloading the artifact gives predictions identical to the pre-save pipeline on the full development fixture (exact equality). *Verify:* round-trip test.

**AC-047.** The final artifact and metadata are present as MLflow artifacts of the release run. *Verify:* MLflow review.

## 9.7 Milestone G — API Service

**AC-048.** `GET /health` returns HTTP 200 with a body indicating the service is alive. *Verify:* API test.

**AC-049.** `GET /model-info` returns HTTP 200 with a body equal to the contents of the loaded artifact's `metadata.json`. *Verify:* API test.

**AC-050.** `POST /predict` with the documented example payload returns HTTP 200 and a body containing a positive predicted price in dollars, the model version from metadata, and `out_of_domain: false`. *Verify:* API test.

**AC-051** *(0-property case: Interpretation IN-05)*. `POST /predict/batch` with 20 valid properties returns HTTP 200 and 20 predictions in input order; with 21 properties it returns HTTP 422 *(project change, M11: supersedes the previous 100-record limit)*; with 0 properties it returns HTTP 422. *Verify:* API tests.

**AC-052.** Each of the following requests returns HTTP 422 and no prediction: a required field missing; a categorical field with a disallowed value; a numeric field outside its allowed range; a non-numeric value in a numeric field; `null` in a non-nullable field. *Verify:* API tests (5 cases).

**AC-053.** A valid request with `null` in a field that is legitimately NA in the dataset (for example `PoolQC`) returns HTTP 200. *Verify:* API test.

**AC-054.** A valid request with `GrLivArea = 4001` returns HTTP 200 with a prediction and `out_of_domain: true`; with `GrLivArea = 4000` it returns `out_of_domain: false`. *Verify:* API tests (boundary cases).

**AC-055.** The artifact is loaded once per service process: a test counting load calls across 10 consecutive requests observes exactly one load. *Verify:* API test.

**AC-056.** Each prediction request produces exactly one log line that parses as JSON and contains an inputs hash, the prediction, a latency value, and the model version. *Verify:* API test capturing log output.

**AC-057 (training–serving consistency).** For every row of a fixture drawn from the development set, the prediction returned by `POST /predict` equals the prediction from calling the loaded pipeline directly on the same raw row. *Verify:* consistency test.

**AC-058.** The OpenAPI documentation page is served, lists all four endpoints, and includes a complete example request payload that passes validation. *Verify:* API test fetching the OpenAPI schema and validating the example.

## 9.8 Milestone H — Packaging, Deployment, and Batch

**AC-059.** The Docker image builds from the repository Dockerfile using a Python 3.12 slim base, contains the model artifact, and its process runs as a non-root user. *Verify:* CI build; inspection of the running container's user.

**AC-060.** A container started from the image locally answers `GET /health` with HTTP 200 and returns the same prediction as AC-050 for the example payload. *Verify:* container smoke test.

**AC-061.** The Render deployment answers `GET /health` with HTTP 200 over HTTPS, and `GET /model-info` reports the same model version as the released artifact. *Verify:* manual check recorded in the release notes.

**AC-062** *(invalid-file behavior: Interpretation IN-14)*. The batch CLI, given a valid CSV in the raw schema, writes an output file with one prediction and one `out_of_domain` flag per input row, in input order; given a CSV with an invalid row, it exits with a non-zero code and reports the invalid rows. *Verify:* integration tests.

**AC-063.** For the same input rows, the batch CLI's predictions equal those returned by the API. *Verify:* integration test.

## 9.9 Milestone I — Quality Automation

**AC-064.** The test suite contains at least one test in each of the eight categories listed in FR-058, and all tests pass. *Verify:* test report grouped by category.

**AC-065.** Test coverage of `src/` is at least 80%. *Verify:* pytest-cov report in CI.

**AC-066.** The CI workflow runs, in order, lint, type check, tests, smoke training, and Docker build on every push and pull request; a failure in any stage fails the workflow. *Verify:* workflow file review and a deliberately failing test run.

**AC-067.** Ruff and mypy report zero errors on `src/`. *Verify:* CI logs.

**AC-068.** The pre-commit configuration includes Ruff, mypy, and file-hygiene hooks, and running all hooks on the repository passes. *Verify:* running pre-commit on all files.

## 9.10 Milestone J — Documentation

**AC-069.** The README contains a problem statement, a results table with the holdout metrics from AC-038, an architecture diagram, and a quickstart; following the quickstart on a clean machine reaches a successful prediction call. *Verify:* review plus a clean-machine walkthrough.

**AC-070.** `docs/adr/` contains ADR-000 and one MADR-format file for each of ADR-01 through ADR-20, each containing status, context, decision, and consequences. *Verify:* file review.

**AC-071.** The data card contains sections for source, license, known issues, scope rule, and missing-value semantics. *Verify:* review.

**AC-072.** The model card contains intended use; out-of-scope uses (real-world valuation, homes above 4,000 sq ft, markets outside Ames 2006–2010); the holdout metrics; the temporal diagnostic result; limitations; and a statement that the production model was refit on all 2,925 rows after holdout evaluation. *Verify:* review.

**AC-073.** Every public function and class in `src/house_price` has a Google-style docstring. *Verify:* automated docstring check or code review.

---

# 10. Traceability Matrix

Every functional requirement maps to at least one acceptance criterion. The ADR column shows each requirement's source.

| FR | Requirement (short) | ADR source | Verified by |
|---|---|---|---|
| FR-001 | Raw data loading, never modified | ADR-01, ADR-04 | AC-001 |
| FR-002 | SHA-256 verification | ADR-01, ADR-04 | AC-002, AC-045 |
| FR-003 | Consistent missing-value parsing | ADR-04, ADR-05 | AC-003 |
| FR-004 | Ingestion schema validation | ADR-04 | AC-004 |
| FR-005 | Shared allowed values | ADR-04, ADR-16 | AC-005 |
| FR-006 | Scope rule before split | ADR-06 | AC-006 |
| FR-007 | Stratified persisted split | ADR-09 | AC-007, AC-008, AC-009 |
| FR-008 | EDA coverage | ADR-07, ADR-19 | AC-010 |
| FR-009 | Holdout protection in EDA | ADR-09 | AC-011 |
| FR-010 | Notebook discipline | ADR-17 | AC-012 |
| FR-011 | Semantic NA filling | ADR-05 | AC-014, AC-015 |
| FR-012 | Approved engineered features | ADR-05, ADR-07 | AC-015, AC-016 |
| FR-013 | Ordinal mapping, MSSubClass categorical | ADR-07 | AC-017 |
| FR-014 | Leakage exclusions | ADR-02 | AC-018 |
| FR-015 | Feature ablation | ADR-07 | AC-019 |
| FR-016 | Excluded techniques | ADR-07 | AC-020 |
| FR-017 | Single end-to-end pipeline | ADR-08 | AC-021, AC-026 |
| FR-018 | Linear-family preprocessing | ADR-08 | AC-022, AC-023 |
| FR-019 | Tree-family preprocessing | ADR-08 | AC-022, AC-023 |
| FR-020 | Fitted imputation in pipeline | ADR-05, ADR-08 | AC-024, AC-025 |
| FR-021 | Config-driven feature groups | ADR-08, ADR-16 | AC-027 |
| FR-022 | Candidate set | ADR-10 | AC-028 |
| FR-023 | Training entry points | ADR-16, ADR-17 | AC-034 |
| FR-024 | Repeated stratified CV | ADR-09 | AC-029, AC-033 |
| FR-025 | CV reporting | ADR-09, ADR-11 | AC-029, AC-035 |
| FR-026 | Linear tuning | ADR-13 | AC-031 |
| FR-027 | Random Forest tuning | ADR-13 | AC-031 |
| FR-028 | LightGBM tuning | ADR-13 | AC-031 |
| FR-029 | Tuning isolation | ADR-09, ADR-13 | AC-030, AC-033 |
| FR-030 | One-SE rule | ADR-11 | AC-035 |
| FR-031 | Blend admission rule | ADR-11 | AC-036 |
| FR-032 | Diagnostic gates | ADR-11 | AC-037 |
| FR-033 | Single holdout evaluation | ADR-09, ADR-11, ADR-12 | AC-038, AC-039, AC-040, AC-041 |
| FR-034 | Production refit | ADR-11 | AC-042 |
| FR-035 | Temporal diagnostic | ADR-09 | AC-043 |
| FR-036 | Run logging | ADR-13, ADR-14, ADR-16 | AC-032 |
| FR-037 | Artifact logging | ADR-14 | AC-047 |
| FR-038 | Artifact and metadata | ADR-14 | AC-044, AC-045 |
| FR-039 | Faithful reload | ADR-14, ADR-18 | AC-046 |
| FR-040 | Health endpoint | ADR-15 | AC-048 |
| FR-041 | Model-info endpoint | ADR-15 | AC-049 |
| FR-042 | Single prediction | ADR-15 | AC-050, AC-057 |
| FR-043 | Batch endpoint | ADR-15 | AC-051 |
| FR-044 | Request validation | ADR-04, ADR-15 | AC-052, AC-053 |
| FR-045 | Out-of-domain flag | ADR-06, ADR-15 | AC-054 |
| FR-046 | Load once at startup | ADR-15 | AC-055 |
| FR-047 | Structured request logging | ADR-15 | AC-056 |
| FR-048 | Container image | ADR-15 | AC-059, AC-060 |
| FR-049 | Render deployment | ADR-15 | AC-061 |
| FR-050 | Offline batch scoring | ADR-04, ADR-15 | AC-062, AC-063 |
| FR-051 | README | ADR-19 | AC-069 |
| FR-052 | ADR log | ADR-19 | AC-070 |
| FR-053 | Data card | ADR-19 | AC-071 |
| FR-054 | Model card | ADR-11, ADR-19 | AC-072 |
| FR-055 | API documentation | ADR-15, ADR-19 | AC-058 |
| FR-056 | Code docstrings | ADR-19 | AC-073 |
| FR-057 | EDA narrative | ADR-19 | AC-013 |
| FR-058 | Test suite categories | ADR-18 | AC-025, AC-057, AC-064, AC-065 |
| FR-059 | Continuous integration | ADR-16, ADR-18 | AC-066, AC-067 |
| FR-060 | Local quality hooks | ADR-16 | AC-068 |

**Coverage check:** all 60 functional requirements map to at least one acceptance criterion, and all 73 acceptance criteria are referenced by at least one requirement.

**NFR verification summary.** Non-functional requirements are verified through the same criteria:

| NFR | Verified by |
|---|---|
| NFR-001 | AC-033, AC-046 |
| NFR-002 | AC-059 (image built from lockfile), clean-checkout walkthrough in AC-069 |
| NFR-003 | AC-069 |
| NFR-004 | AC-002, AC-004 |
| NFR-005 | AC-052 |
| NFR-006 | AC-023 |
| NFR-007 | AC-021 |
| NFR-008 | AC-012, code review |
| NFR-009 | AC-067 |
| NFR-010 | AC-026 |
| NFR-011 | AC-049, AC-050 |
| NFR-012 | AC-059 |
| NFR-013 | AC-070 |
| NFR-014 | AC-065 |
| NFR-015 | AC-066 |
| NFR-016 | AC-031 (budgets), run duration recorded in MLflow |
| NFR-017 | AC-051, AC-055, AC-056 |
| NFR-018 | AC-066 |
| NFR-019 | AC-027 |
| NFR-020 | AC-027, code review |
| NFR-021 | AC-070 |
| NFR-022 | AC-071, AC-072 |
| NFR-023 | AC-058 |
| NFR-024 | AC-057, AC-063 |
| NFR-025 | AC-059 |
| NFR-026 | AC-060, AC-061 |
| NFR-027 | AC-032, AC-044, AC-045, AC-047 |
| NFR-028 | This section |
| NFR-029 | AC-050, AC-056 |

---

# 11. Project-Level Definition of Done

The release is complete only when every item below is checked. This list is the final audit checklist.

## 11.1 Data

- [ ] Raw CSV unchanged; SHA-256 committed and verified on every run (AC-001, AC-002).
- [ ] Ingestion schema passes on the raw file and rejects all tested invalid cases (AC-003, AC-004).
- [ ] Ingestion and API schemas share one configuration source (AC-005).
- [ ] Scope rule leaves 2,925 rows; removed IDs recorded (AC-006).
- [ ] Split persisted, reproducible, stratified, disjoint (AC-007, AC-008, AC-009).

## 11.2 Analysis

- [ ] All DOC-02 Section 13 deliverables present (AC-010).
- [ ] Holdout untouched by EDA target analysis (AC-011).
- [ ] Notebooks contain no core logic (AC-012).
- [ ] EDA conclusions link each feature and preprocessing choice to evidence (AC-013).

## 11.3 Pipeline

- [ ] Semantic filler and feature engineer tested and stateless (AC-014, AC-015, AC-016).
- [ ] Ordinal mapping and `MSSubClass` handling correct (AC-017).
- [ ] `SaleType`, `SaleCondition`, `Id`, `PID` excluded as features (AC-018).
- [ ] Ablation recorded; every retained feature justified (AC-019).
- [ ] No excluded techniques present (AC-020).
- [ ] Pipeline predicts dollars from raw rows; both branches behave as specified (AC-021 to AC-024, AC-026).
- [ ] Leakage test passes (AC-025).
- [ ] Feature groups driven by YAML (AC-027).

## 11.4 Modeling

- [ ] All 7 candidates evaluated with 5×3 repeated stratified CV (AC-028, AC-029).
- [ ] Holdout never read during tuning (AC-030).
- [ ] Tuning budgets match ADR-13 (AC-031).
- [ ] All runs logged with full provenance (AC-032).
- [ ] Training reproducible: identical splits, same selected model and hyperparameters, CV metrics within the documented 1e-6 tolerance (AC-033).
- [ ] Selection follows the one-SE rule and blend rule, with a written record (AC-035, AC-036).
- [ ] Diagnostic gates passed and recorded (AC-037).
- [ ] Selected model received exactly one final holdout evaluation; baseline reference scoring computed once, after selection, and used for no decision (AC-038).
- [ ] Holdout log-RMSE ≤ 0.13 (AC-039).
- [ ] Holdout MAPE ≤ 10% (AC-040).
- [ ] Both baselines beaten by a clear margin as defined in IN-11 (AC-041).
- [ ] Production model refit on 2,925 rows (AC-042).
- [ ] Temporal diagnostic reported (AC-043).

## 11.5 Artifact

- [ ] Joblib artifact and complete metadata saved (AC-044, AC-045).
- [ ] Reload round-trip exact (AC-046).
- [ ] Artifact logged to MLflow (AC-047).

## 11.6 Service and Deployment

- [ ] All four endpoints behave as specified (AC-048 to AC-051).
- [ ] Validation, nullable fields, and domain flag correct (AC-052, AC-053, AC-054).
- [ ] Model loads once; every request logged (AC-055, AC-056).
- [ ] API and pipeline predictions identical (AC-057).
- [ ] OpenAPI docs with valid example (AC-058).
- [ ] Docker image: 3.12-slim, non-root, artifact baked in, works locally (AC-059, AC-060).
- [ ] Render deployment live and reporting the released version (AC-061).
- [ ] Batch CLI correct and consistent with the API (AC-062, AC-063).

## 11.7 Quality Automation

- [ ] All eight test categories present and passing (AC-064).
- [ ] Coverage of `src/` ≥ 80% (AC-065).
- [ ] CI runs all five stages and gates merges (AC-066).
- [ ] Ruff and mypy clean (AC-067).
- [ ] pre-commit configured and passing (AC-068).

## 11.8 Documentation

- [ ] README complete; quickstart verified on a clean machine (AC-069).
- [ ] ADR-000 and ADR-01 to ADR-20 in MADR format (AC-070).
- [ ] Data card complete (AC-071).
- [ ] Model card complete, including limitations and refit statement (AC-072).
- [ ] Docstrings on all public APIs (AC-073).

## 11.9 Release Sign-off

- [ ] Release version tagged in git; the tag's commit SHA matches the artifact metadata.
- [ ] The deployed model version, the artifact metadata, and the README results table all report the same version and metrics.
- [ ] No open ADR deviations: any change from ADR-000 has its own superseding ADR.
- [ ] Every interpretation note (IN-01 to IN-14) is still consistent with the ADR; any interpretation that had to change is recorded in this document's version history.
