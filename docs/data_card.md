# Data Card: Ames Housing (Draft)

**Status:** Draft (M2), updated in M3 with the EDA evidence. Every known issue below was verified by the M3 deliverables (DOC-02 §13); the E-35 confirmation register records 59 of 60 documented properties confirmed and 1 not confirmed (K-17). Finalized in M13.
**Requirement:** FR-053 / AC-071 (source, license, known issues, scope rule, missing-value semantics).
**Evidence:** `reports/data_validation/` (M2, `python -m house_price.data.validate`), `data/processed/split_manifest.json`, and `reports/eda/` with `reports/figures/eda/` (M3, `python -m house_price.eda`; register `E-35_confirmation_register.csv`, inputs `E-36_data_card_inputs.csv`).

---

## 1. Source

| Property | Value |
|---|---|
| Dataset | Ames Housing dataset, compiled by Dean De Cock from the Ames, Iowa Assessor's Office |
| Reference | De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.* Journal of Statistics Education, 19(3) |
| File | `data/raw/train.csv`, the full 2,930-row file (82 columns) |
| SHA-256 | `65a1cbb89c2b58b11674097135dd04e710df5db726bcfe6ec551de38b44e4911` (committed in `configs/data.yaml`, verified on every load) |
| Size | 963,738 bytes, CRLF line endings |
| Sale period | 2006 to 2010 (`YrSold`) |
| Target | `SalePrice` (US dollars) |

### Dataset version

ADR-01 and DOC-01/DOC-02 describe the 1,460-row Kaggle `train.csv`. By project-owner decision (2026-09-28), the project instead uses the full 2,930-row De Cock file present in the repository. The design (hash check, explicit parsing, schema contract, scope rule, stratified split) is unchanged. The numbers that follow from the data are different:

| Quantity | Documents (Kaggle file) | This dataset |
|---|---|---|
| Raw rows x columns | 1,460 x 81 | 2,930 x 82 |
| Identifier(s) | `Id` | `Order` (loaded as `Id`) and `PID` |
| Rows removed by the scope rule | 4 | 5 |
| In-scope rows | 1,456 | 2,925 |
| Development / holdout | 1,164-1,165 / 291-292 | 2,340 / 585 |
| Nullable columns | 19 | 27 |
| Model-input columns (DN-11) | 77 | 77 |

These numbers now live in configuration (`configs/data.yaml: scope.expected_rows_after`) rather than in code. DOC-01 to DOC-05 are aligned with this dataset; the ADR text (ADR-01, ADR-06) still contains Kaggle-file assumptions.

### Column names

Raw headers use the original dictionary spelling with spaces (`Gr Liv Area`, `Year Remod/Add`). On load, each header is renamed to the canonical name used throughout the code and the design documents (spaces and `/` removed, e.g. `GrLivArea`, `YearRemodAdd`). `Order`, the source row number, becomes `Id`. The mapping is declared per column in `configs/schema.yaml` (`source_name` to `name`). Only names change on load; values are never changed.

## 2. License

The dataset was published with the article above for educational use. This project does not redistribute it: `data/raw/` is gitignored and only its SHA-256 is committed. Anyone reproducing the project obtains the file themselves and must confirm the terms of the copy they download.

## 3. Scope Rule

- **Rule:** the model covers homes with `GrLivArea <= 4000` sq ft [ADR-06]. Rows above the threshold are removed once, on the validated raw dataset, **before** the split (`house_price.data.scope.apply_scope_rule`).
- **Effect on this dataset:** 5 rows removed, 2,925 in scope. The count is asserted against `configs/data.yaml`, and any other count stops the pipeline.

| Id | GrLivArea | SaleCondition | Note |
|---|---|---|---|
| 1499 | 5,642 | Partial | Low price for its size (unfinished-home sale) |
| 1761 | 4,476 | Abnorml | Very expensive home at the edge of the data |
| 1768 | 4,316 | Normal | Very expensive home at the edge of the data |
| 2181 | 5,095 | Partial | Low price for its size; built after the recorded sale year |
| 2182 | 4,676 | Partial | Low price for its size (unfinished-home sale) |

*Confirmed in M3 (E-24 to E-26; register P-04, P-55 to P-57).* The three partial sales sit 8.8, 7.1, and 6.2 standard deviations below the in-scope size trend on the log scale (Ids 1499, 2181, 2182); the other two are at the 99.97th and 100th price percentiles of the raw file. Removing all five steepens the fitted log-price slope on `GrLivArea` by 5.9%, so they do flatten the size effect. DOC-02 §9.2 documents 4 such homes in the Kaggle subset; this file has 5, recorded as a discrepancy against the ADR-06 text (K-15).

- **Why it is a scope decision, not cleaning:** the same rule defines the training population, the holdout population, and the serving domain. The API still scores larger homes but flags them `out_of_domain` (DOC-02 §9.4).
- **Split:** stratified 80/20 on deciles of `log1p(SalePrice)` with seed 42. The result is 2,340 development / 585 holdout rows, and the largest per-bin share difference is 0.13 pp (tolerance 2 pp). The `Id` lists and file hashes are in `data/processed/split_manifest.json`.

## 4. Missing-Value Semantics

**Parsing (DN-18).** Only the token `NA` and the empty cell are parsed as missing, and library default token lists are disabled. This file uses **both** tokens. Most "feature absent" values are `NA`, while most genuinely unrecorded numeric values (`LotFrontage`, `GarageYrBlt`, `MasVnrArea`) are empty cells. Both are counted as missing.

**Two kinds of missing value** [ADR-05, DOC-02 §6]:

| Kind | Meaning | Columns (missing count in the raw file) |
|---|---|---|
| Feature absent | The house does not have the feature; `NA` is information, not ignorance | `PoolQC` (2,917), `MiscFeature` (2,824), `Alley` (2,732), `Fence` (2,358), `FireplaceQu` (1,422), `GarageType` (157), `GarageFinish` / `GarageQual` / `GarageCond` / `GarageYrBlt` (159), basement categoricals (80-83), `MasVnrType` / `MasVnrArea` (23) |
| Unknown | The attribute exists but its value was not recorded | `LotFrontage` (490), `Electrical` (1), and the unrecorded fields of the anomaly rows in K-02 to K-05 |

*Confirmed in M3 (E-09, E-10; register P-39, P-60).* 96.3% of the 13,997 missing cells mean "feature absent", established from companion columns where they exist (for example `GarageArea = 0`, `PoolArea = 0`, `Fireplaces = 0`) and from the data dictionary otherwise. `LotFrontage` missingness carries some price signal: development rows without it have a median log price 0.091 higher (about 10%) than rows with it (E-12), which supports the missing-indicator column.

In M4 and M5, every missing value in a Layer 1 column (the absent-feature categoricals and their related numerics) is filled by the stateless semantic filler (`"None"` / `0`) *whatever its cause*, including the unknown cells of the anomaly rows (FR-020, DOC-02 §6.6). Only `LotFrontage`, `Electrical`, `GarageAge` for a garage without a recorded year, and any unexpected missing value at serving time go to fitted imputation inside the pipeline. No missing value is filled outside the pipeline. The per-column counts are in `reports/eda/E-09_missing_values.csv`, and every column with missing values is declared `nullable` in `configs/schema.yaml`.

## 5. Known Issues

Every item was verified in M3; the **M3** column gives the status and evidence (E-xx deliverable, P-xx register entry). No value is changed in response to any of them (DOC-03 §5.4). Row identifiers refer to `Id` (= source `Order`).

| # | Issue | Rows | Handling | M3 |
|---|---|---|---|---|
| K-01 | **`MasVnrType` parsing hazard.** The column holds 23 missing values and 1,752 literal `None` ("no veneer") values. Default CSV parsing would report 1,775 missing. | 23 / 1,752 | Explicit tokens (DN-18); tested (AC-003) | Confirmed (E-04, P-40) |
| K-02 | **Basement anomalies.** A basement exists (`BsmtQual` recorded) but `BsmtExposure` is missing. | 67, 1797, 2780 | Semantic filler maps to `"None"` by column rule, a known approximation (DOC-02 §6.6) | Confirmed (E-11, P-41) |
| K-03 | **Basement anomaly.** `BsmtFinSF2 = 479` but `BsmtFinType2` is missing. | 445 | As K-02 | Confirmed (E-11, P-41) |
| K-04 | **Basement fields unrecorded.** All basement categoricals are `NA` and all basement square-footage and bath fields are empty: unknown rather than zero. | 1342 | Semantic filler fills `"None"` / `0`, i.e. treats the house as having no basement; a known approximation (DOC-02 §6.6, DOC-03 §6.2) | Confirmed (E-11, P-42) |
| K-05 | **Garage partially recorded.** `GarageType = Detchd` but finish/quality/condition/year are missing; Id 2237 also lacks `GarageCars` and `GarageArea`. | 1357, 2237 | Semantic filler fills the categoricals with `"None"` and Id 2237's `GarageCars`/`GarageArea` with `0`; `GarageAge` stays missing and is imputed in-pipeline (DOC-03 §6.4) | Confirmed (E-11, P-42) |
| K-06 | **Impossible garage year.** `GarageYrBlt = 2207` (house built 2006, sold 2007; likely a typo for 2007). The `GarageYrBlt` range in `schema.yaml` was widened to 1800-2210 so the raw file validates. | 2261 | Kept as recorded; `GarageYrBlt` is replaced by `GarageAge` in M4, where this row yields a negative age that is passed through | Confirmed (E-11, `garage_built_after_sale`) |
| K-07 | **Remodel-date floor.** `YearRemodAdd = 1950` for houses built before 1950; 1950 is a recording floor, not a remodel (DOC-02 §10.5). | 339 rows | Kept; affects `RemodAge` / `IsRemodeled` (M4, E-29) | Confirmed (E-11, E-13, P-25); effect quantified in M4 (E-29) |
| K-08 | **Remodel before construction.** `YearRemodAdd < YearBuilt`. | 851 | Kept | Confirmed (E-11) |
| K-09 | **Built after the sale.** `YearBuilt > YrSold` (pre-completion sale). | 2181 (out of scope) | Removed by the scope rule, not by cleaning | Confirmed (E-11) |
| K-10 | **`MasVnrType = None` with `MasVnrArea > 0`.** | 364, 404, 442, 1862, 1914, 2004, 2529 | Kept | Confirmed (E-11) |
| K-11 | **Dictionary vs file spellings.** `MSZoning`: `A (agr)`, `C (all)`, `I (all)`; `Neighborhood`: `NAmes` (dictionary `Names`); `BldgType`: `2fmCon`, `Duplex`, `Twnhs`; `Exterior2nd`: `Brk Cmn`, `CmentBd`, `Wd Shng`; `SaleType`: `WD ` (trailing space). | n/a | Both spellings allowed in `schema.yaml` (DOC-03 §5.3); values not normalised | Confirmed (E-02: no value outside the allowed lists) |
| K-12 | **Identical features, different prices.** Two pairs of adjacent parcels (townhouse units) sold in the same month match in every feature but differ in price. | 2508/2509, 2829/2830 | Kept: distinct properties (different `PID`) | Confirmed in M2 (`validation_report.md`) |
| K-13 | **Encoded identifiers.** `MSSubClass` is zero-padded in the file (`020`) and is parsed to the integer code. `PID` is kept as text to preserve its leading zero. | All rows | Declared in `schema.yaml` | Confirmed (E-01: all 82 columns load with the declared dtype) |
| K-14 | **Unknown electrical system.** | 1578 | Imputed in-pipeline (most frequent) | Confirmed (E-09, P-34) |
| K-15 | **Scope count differs from the ADR-06 text.** 5 homes exceed 4,000 sq ft instead of the 4 in the Kaggle subset (Section 3). | 1499, 1761, 1768, 2181, 2182 | Recorded; count set in `configs/data.yaml` and in DOC-01 AC-006; the ADR text awaits supersession | Confirmed (E-25, P-04); the aligned documents and the configured count agree, so M3 does not stop |
| K-16 | **Rare categories.** 104 category levels in 34 features cover under 1% of development rows each (for example the smallest neighborhood has 1 sale in the raw file). | n/a | Unseen-category handling in both branches (M5); estimates for these levels are unreliable | Confirmed (E-19, E-20, P-51) |
| K-17 | **Register discrepancy P-44.** DOC-02 §7.1 lists `GrLivArea` among the strongly right-skewed features, but on the in-scope development set its skewness is 0.9, below the register's threshold of 1; the other six listed features exceed it (2.4 to 21.4). Removing the five largest homes shortens the tail. | n/a | No design change: the linear branch applies Yeo-Johnson to every numeric feature regardless (ADR-08); not an ADR-fixed number | **Not confirmed** (E-13, P-44) |

No duplicate `Id` or `PID` values and no exact duplicate records exist (`reports/data_validation/validation_report.md`).
