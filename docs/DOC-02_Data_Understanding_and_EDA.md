# DOC-02: Data Understanding and EDA

**Project:** House Price Prediction — End-to-End ML Regression System
**Document ID:** DOC-02
**Version:** 1.2
**Status:** Approved baseline — revision 1.2 (dataset alignment: the project uses the full 2,930-row Ames file; see `docs/data_card.md`)
**Date:** 2026-09-28
**Authoritative source:** `house-price-prediction-adr.md` (ADR-000, ADR-01 to ADR-20)
**Related documents:** DOC-01 Product Requirements and Acceptance Criteria

---

## How to Read This Document

This document has two jobs.

1. **Understand the data.** It explains what the Ames dataset contains, what each part means in real-estate terms, and which of its properties matter for modeling.
2. **Connect EDA to architecture.** Several ADR decisions (log target, semantic missing-value handling, the 4,000 sq ft scope rule, the engineered features, the two preprocessing branches) rest on facts about the data. This document records those facts and shows how each one leads to a decision.

**On the status of the facts in this document.** This document is an EDA **design** and data-understanding baseline. The project's EDA has not yet been executed. Two kinds of statement must therefore be kept apart:

| Term | Meaning | Where it lives |
|---|---|---|
| **Documented property** | A characteristic of the project's raw file (the full 2,930-row De Cock Ames file) that is already known from the dataset author's documentation (De Cock, 2011), the data dictionary, or the M2 ingestion and validation profile of this file (`reports/data_validation/`). Every number in this document (row counts, target statistics, missing-value counts, category counts, the 5 homes above 4,000 sq ft, known quirks) is of this kind. | This document (version 1.2) |
| **Project EDA finding** | A result actually produced by this project's executed EDA notebooks or scripts, with its evidence artifact. | EDA deliverables E-01 to E-36 (Section 13), summarized in E-35 |

**Status in version 1.2: this document contains no project EDA findings.** Every data characteristic stated here is a documented property **to be verified by project EDA**. Where a number or pattern is stated, the text either says "documented" or carries a tag of the form *(documented; verify: E-xx)* naming the deliverable that will provide the evidence.

Each documented property is handled with the same chain:

> **Documented property → question to verify (Section 4) → evidence to collect (Section 13) → architectural consequence if confirmed (Section 12)**

If project EDA confirms a property, E-35 records it as confirmed with the observed value, and it becomes a project EDA finding. If project EDA contradicts a property, the discrepancy is recorded in E-35 and the data card; any resulting change to a decision follows the ADR supersession process [ADR-19]. The ADR decisions themselves are final and are not reopened here: they do not wait on EDA, and a discrepancy never silently changes them.

**Holdout protection.** Following DOC-01 FR-009, analyses that relate features to `SalePrice` in order to inform modeling are run on the development set only. Target-free data-quality profiling may use the full raw file. Two documented exceptions apply: the split-balance check (E-08, DOC-01 AC-009), which inspects only the target distribution to confirm the split worked; and the ADR-06 scope review (E-24 to E-26), which must use the raw file because the 5 affected rows are removed before the split. Figures quoted below for the full raw file are the dataset's documented properties, used to explain the decisions already fixed in the ADR.

---

# 1. Dataset Overview

## 1.1 Source

The dataset is the **Ames Housing dataset**, compiled by Dean De Cock of Truman State University from the Ames, Iowa Assessor's Office and published in the *Journal of Statistics Education* in 2011. It was built specifically as a modern, richer teaching alternative to the Boston Housing dataset.

The project uses the full De Cock file (2,930 sales), stored as `data/raw/train.csv`. ADR-01 named the 1,460-row `train.csv` of the Kaggle competition *"House Prices: Advanced Regression Techniques"*, which is a subset of this file; the project uses the full file instead by project-owner decision (recorded in the data card). The full file also contains the properties of the Kaggle `test.csv`, with their prices, so no Kaggle file is used in any role, including the optional Kaggle submission [ADR-01].

## 1.2 Size and Shape

| Property | Value |
|---|---|
| Rows (properties) | 2,930 |
| Columns | 82 |
| Identifiers | `Id` (the source file's `Order` column) and `PID` (parcel ID) (2 columns) |
| Features | 79 |
| Target | `SalePrice` (1 column) |
| Sale period | January 2006 – July 2010 |
| Geography | Ames, Iowa, USA |
| Rows after scope rule | 2,925 [ADR-06] |
| Development set | 2,340 rows (80%) [ADR-09] (DOC-01 AC-007) |
| Holdout set | 585 rows (20%) [ADR-09] (DOC-01 AC-007) |

*Status:* the raw-file shape, sale period, and the 5-row reduction to 2,925 are documented properties (verify: E-01, E-24, E-25, E-32). The scope rule and the 80/20 split are fixed by the ADR [ADR-06, ADR-09]; the 2,925 count is the verified count for this file, set in `configs/data.yaml`.

*Column names.* The raw file's headers use the data dictionary's spelling with spaces (for example `Gr Liv Area`, `Year Remod/Add`). Ingestion renames them to the canonical names used in this document and throughout the project (`GrLivArea`, `YearRemodAdd`; `Order` becomes `Id`), as declared in `configs/schema.yaml`. Only names change; values are not modified.

## 1.3 Feature Types

As stored in the raw file, the 79 features split into:

- **36 numeric columns** (integers and floats). One of these, `MSSubClass`, is a numeric *code* for a dwelling type, not a quantity, and is treated as categorical [ADR-07].
- **43 text (categorical) columns.** Some are **nominal** (no natural order, for example `Neighborhood`); many are **ordinal** (a natural order, for example quality ratings from Poor to Excellent).

*(Documented; verify: E-01, E-13, E-19.)*

This means the dataset has three kinds of information that must be handled differently: continuous quantities, unordered categories, and ordered categories. Section 3 groups them by meaning; Sections 7 and 8 describe how each kind is analyzed.

## 1.4 Target Variable

`SalePrice`: the recorded sale price of the property in US dollars. Documented property, to be verified by project EDA (E-05) — range in the raw file: $12,789 to $755,000, with a median of $160,000 and a mean of about $180,800.

## 1.5 Why This Dataset Was Selected

The selection reasons from ADR-01, explained in terms of what they teach:

1. **Realistic complexity.** 79 mixed-type features force real preprocessing decisions. A dataset with 8 clean numeric columns (California Housing) teaches almost nothing about preprocessing.
2. **Meaningful missing values.** Many `NA` values mean "this house has no pool/garage/basement," not "we don't know." Learning to tell these apart is a core tabular-data skill (Section 6).
3. **A skewed target.** House prices are right-skewed, which makes target transformation a real, measurable decision (Section 5).
4. **Known outliers.** The dataset author documented a small set of unusual sales, which lets the project practice outlier reasoning on a known case (Section 9).
5. **A public data dictionary.** Every column and code is defined, so decisions can be grounded in documentation rather than guesswork.
6. **Comparability.** Many practitioners have worked with this dataset (most often its 1,460-row Kaggle subset), so results can be sanity-checked against community reference points, bearing in mind that scores on the full file are not directly comparable with Kaggle leaderboard scores.

Alternatives were rejected because they are too simple (California Housing), ethically problematic and removed from scikit-learn (Boston Housing), or too clean (King County) [ADR-01]. ADR-01 also preferred the Kaggle subset over the full 2,930-row De Cock file for benchmark comparability; the project now uses the full file (project-owner decision, recorded in the data card), trading direct leaderboard comparability for twice as many labeled rows.

---

# 2. Business Understanding

## 2.1 What SalePrice Represents

`SalePrice` is the price at which a property actually changed hands, as recorded by the assessor. It is not an appraisal, a listing price, or an assessed tax value. It reflects what a buyer was willing to pay in a real transaction, which makes it the most direct available measure of market value.

It is also shaped by the **circumstances of the sale**, not only by the house. A foreclosure, a sale between family members, or a sale of a house that was not yet finished can all produce prices that differ from the house's typical market value. This is why the circumstances of the sale are handled carefully (Section 11) [ADR-02].

## 2.2 What Influences Residential Property Value

Real-estate valuation practice commonly groups value drivers into a few families. The Ames features cover each of them:

| Value driver | Intuition | Example Ames features |
|---|---|---|
| **Size** | More usable space costs more | `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `LotArea` |
| **Quality** | Better materials and finish cost more to build and are valued by buyers | `OverallQual`, `ExterQual`, `KitchenQual` |
| **Condition** | Worn or damaged components reduce value | `OverallCond`, `ExterCond`, `Functional` |
| **Age and renovation** | Newer or recently renovated homes need less investment | `YearBuilt`, `YearRemodAdd` |
| **Location** | Neighborhood, nearby roads, rail lines, and parks affect desirability | `Neighborhood`, `Condition1`, `MSZoning` |
| **Amenities** | Extras such as fireplaces, garages, porches, and pools add value | `Fireplaces`, `GarageCars`, `PoolArea` |
| **Configuration** | Layout, dwelling type, and number of rooms affect usability | `MSSubClass`, `HouseStyle`, `TotRmsAbvGrd` |
| **Market timing** | Prices move with the overall market | `YrSold`, `MoSold` |

A central expectation, which EDA must confirm, is that **size and quality dominate**: in this dataset `OverallQual` and `GrLivArea` are documented as the two features most strongly correlated with price. That is exactly why they form the heuristic baseline [ADR-10].

## 2.3 Why This Is a Regression Problem

The target is a continuous dollar amount. The business needs a specific estimate ("about $210,000"), not a category ("medium price"). Converting prices into bands would throw away information and give users a less useful answer, which is why classification was rejected [ADR-02]. The task is therefore **supervised regression**: learn a function from property attributes to price using past sales with known prices.

## 2.4 How Predictions Will Be Used

The system is framed as a pricing-guidance tool: a simple Automated Valuation Model (AVM) [ADR-02]. In that setting:

- Predictions are made **before a sale closes**, typically at listing time. So the model may only use information that exists at that point (Section 11).
- The user cares about **relative accuracy**. Being 10% off matters similarly whether the house is cheap or expensive. This motivates the log target (Section 5) and the primary metric [ADR-12].
- The model is **only trusted inside its domain**. Homes over 4,000 sq ft of living area are flagged as out of domain [ADR-06, ADR-15].
- The system is a **learning demonstration**. Its data covers one city from 2006–2010, so it must not be used for real-world valuation [ADR-19].

---

# 3. Dataset Structure

The 79 features are grouped below by what they describe. Each group lists its business meaning, its columns, and what matters for modeling. Every feature appears in exactly one group.

*Status:* column membership comes from the data dictionary. Counts, shares, and characterizations in the "Modeling relevance" paragraphs (for example, how many houses lack a garage, or which feature is the strongest predictor) are documented properties to be verified by project EDA; the relevant evidence is E-09 to E-11 (missing values), E-13 to E-18 (numeric), and E-19 to E-23 (categorical).

## 3.1 Location (4 features)

**Columns:** `Neighborhood`, `MSZoning`, `Condition1`, `Condition2`

**Business meaning.** Location is widely considered the single most important non-physical driver of value. `Neighborhood` identifies one of 28 areas within Ames. `MSZoning` gives the zoning class (for example residential low density, residential medium density, commercial). `Condition1` and `Condition2` record proximity to features that raise or lower desirability, such as busy arterial roads, railroads, or positive off-site features such as parks.

**Modeling relevance.** `Neighborhood` captures a large bundle of unmeasured factors (school quality, reputation, typical lot and home style). It has 28 levels, which one-hot encoding handles well at this data size; target encoding was rejected because of leakage risk [ADR-07]. `Condition2` is almost always "Norm" (normal), so it carries very little information (Section 8).

## 3.2 Lot Characteristics (8 features)

**Columns:** `LotArea`, `LotFrontage`, `LotShape`, `LandContour`, `LotConfig`, `LandSlope`, `Street`, `Alley`

**Business meaning.** The land the house sits on. `LotArea` is total lot size in square feet; `LotFrontage` is the length of street touching the property. Shape, contour, configuration (inside lot, corner lot, cul-de-sac), and slope affect usability and appeal. `Street` and `Alley` describe the type of road and alley access.

**Modeling relevance.** `LotArea` is heavily right-skewed, with a few very large rural lots. This matters for linear models and is handled by the power transform in the linear branch [ADR-08]. `LotFrontage` is the main example of a **genuinely unknown** value (documented at 490 rows, about 17% missing; verify: E-09) and is imputed statistically [ADR-05]. `Alley` is mostly `NA`, meaning "no alley access" (a "feature absent" value). `Street` is almost always paved.

## 3.3 Building Type and Style (3 features)

**Columns:** `MSSubClass`, `BldgType`, `HouseStyle`

**Business meaning.** What kind of dwelling it is. `MSSubClass` is a coded dwelling class combining age and style (for example "1-story 1946 and newer"). `BldgType` distinguishes single-family homes, townhouses, and duplexes. `HouseStyle` describes the number of stories and finish level.

**Modeling relevance.** `MSSubClass` is stored as a number but its values are codes, not quantities. Class 60 is not "twice" class 30. Treating it as numeric would impose a false ordering, so it is treated as categorical [ADR-07].

## 3.4 Property Dimensions and Room Counts (9 features)

**Columns:** `GrLivArea`, `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `TotRmsAbvGrd`, `BedroomAbvGr`, `KitchenAbvGr`, `FullBath`, `HalfBath`

**Business meaning.** How much above-ground living space the house has and how it is divided. `GrLivArea` (above-grade living area) is essentially `1stFlrSF + 2ndFlrSF + LowQualFinSF`. The room counts describe layout.

**Modeling relevance.** This is the heart of the size signal. `GrLivArea` is one of the two strongest single predictors, and it is the variable used for the scope rule [ADR-06]. Several columns overlap heavily (for example `GrLivArea` and `TotRmsAbvGrd` are strongly correlated), which matters for linear models and motivates regularization [ADR-10]. Above-ground area misses the basement, which motivates `TotalSF` [ADR-07].

## 3.5 Overall Quality and Condition (3 features)

**Columns:** `OverallQual`, `OverallCond`, `Functional`

**Business meaning.** `OverallQual` rates the overall material and finish on a 1–10 scale. `OverallCond` rates the overall state of repair on a 1–10 scale. `Functional` records deductions for functional problems (for example major damage).

**Modeling relevance.** `OverallQual` is documented as the single strongest predictor of price in this dataset. Its relationship with price is increasing and convex: each step up the scale adds more dollars than the last, which becomes closer to linear in log-price space [ADR-02]. `OverallCond` has a weaker and less tidy relationship with price, because well-maintained old houses and new houses can share similar condition scores.

## 3.6 Age and Renovation (2 features)

**Columns:** `YearBuilt`, `YearRemodAdd`

**Business meaning.** When the house was built and when it was last remodeled. If no remodel happened, `YearRemodAdd` equals `YearBuilt`.

**Modeling relevance.** Raw years are less meaningful than **ages at the time of sale**. A house built in 1990 is 16 years old if sold in 2006 and 20 years old if sold in 2010. This motivates `HouseAge`, `RemodAge`, and `IsRemodeled` [ADR-07]. EDA must also check a documented recording quirk: `YearRemodAdd` is documented to have a floor of 1950, so for older houses a value of 1950 may reflect the recording convention rather than an actual 1950 remodel (Section 10.5; verify: E-29).

## 3.7 Basement (11 features)

**Columns:** `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinSF1`, `BsmtFinType2`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath`

**Business meaning.** Iowa homes commonly have basements, which add storage and, when finished, living space. These columns describe basement height (`BsmtQual` rates ceiling height), condition, walkout or garden-level exposure, how much is finished and to what standard, and basement bathrooms.

**Modeling relevance.** Houses without a basement have `NA` in all the basement categorical columns. This is a textbook "feature absent" case [ADR-05]. `TotalBsmtSF` is a large component of total usable space and feeds `TotalSF`; basement bathrooms feed `TotalBath` [ADR-07].

## 3.8 Garage (8 features)

**Columns:** `GarageType`, `GarageYrBlt`, `GarageFinish`, `GarageCars`, `GarageArea`, `GarageQual`, `GarageCond`, `PavedDrive`

**Business meaning.** Whether there is a garage, its type (attached, detached, built-in), when it was built, its interior finish, its capacity in cars, its size, and its quality and condition. `PavedDrive` describes the driveway.

**Modeling relevance.** The dataset is documented to contain 157 houses with no garage (verify: E-09, E-11); for these, the garage categorical columns and `GarageYrBlt` are `NA`. Two further houses have a garage type but other garage fields unrecorded (Section 6.6). `GarageYrBlt` cannot be meaningfully "imputed" for a garage that does not exist, which is why it is replaced by `HasGarage` and `GarageAge` [ADR-05]. `GarageCars` and `GarageArea` measure nearly the same thing and are strongly correlated, another multicollinearity case for linear models.

## 3.9 Exterior and Structure (9 features)

**Columns:** `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`, `MasVnrArea`, `Foundation`, `ExterQual`, `ExterCond`

**Business meaning.** The outside of the house: roof style and material, primary and secondary siding materials, masonry veneer (for example brick facing) type and area, foundation type, and the quality and condition of exterior materials.

**Modeling relevance.** `ExterQual` is a quality scale that maps to ordered integers [ADR-07]. `Foundation` often acts as an age proxy (poured concrete is common in newer homes). `RoofMatl` is dominated by one value (standard composite shingle), so its rare categories carry little reliable information. `Exterior1st` and `Exterior2nd` have 16 and 17 levels respectively in the raw file, the highest cardinality after `Neighborhood`. `MasVnrType` requires careful parsing (Section 6.5).

## 3.10 Utilities and Systems (5 features)

**Columns:** `Utilities`, `Heating`, `HeatingQC`, `CentralAir`, `Electrical`

**Business meaning.** Services and mechanical systems: available utilities, heating type and quality, central air conditioning, and electrical system type.

**Modeling relevance.** `Utilities` is effectively constant (all but three houses have all public utilities) and so contributes almost no information. `Heating` is dominated by gas forced-air. `CentralAir` is a meaningful yes/no signal, since houses without central air tend to be older and cheaper. `HeatingQC` is a quality scale. `Electrical` has a single genuinely unknown value, imputed with the most frequent category [ADR-05].

## 3.11 Interior Amenities (3 features)

**Columns:** `Fireplaces`, `FireplaceQu`, `KitchenQual`

**Business meaning.** The number and quality of fireplaces, and kitchen quality. Kitchens are widely considered one of the most value-relevant rooms in a home.

**Modeling relevance.** `KitchenQual` is a quality scale and is expected to be among the stronger categorical predictors. `FireplaceQu` is `NA` for houses without a fireplace ("feature absent"; documented count 1,422, verify: E-09) [ADR-05]. `HasFireplace` captures the presence signal directly [ADR-07].

## 3.12 Outdoor Features (8 features)

**Columns:** `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `PoolArea`, `PoolQC`, `Fence`

**Business meaning.** Outdoor living space: decks, four types of porches, a pool, and fencing.

**Modeling relevance.** Most houses have zero area in most of these columns, which produces heavily zero-inflated distributions. Individually, each porch type is sparse and noisy; together they describe a household's outdoor living space, which motivates `TotalPorchSF` [ADR-07]. The dataset is documented to contain only 13 houses with a pool, so `PoolQC` is about 99.6% `NA` ("no pool"; verify: E-09, E-13). Pools are too rare to learn much about, but `HasPool` preserves the presence signal [ADR-05, ADR-07].

## 3.13 Miscellaneous (2 features)

**Columns:** `MiscFeature`, `MiscVal`

**Business meaning.** Other features not covered elsewhere (for example a shed, a second garage, or a tennis court) and their dollar value.

**Modeling relevance.** `MiscFeature` is `NA` ("none") for a documented ~96% of houses (verify: E-09). `MiscVal` is almost always zero and has a few large values. Both are low-information and are kept only because the ADR does not exclude them; their usefulness is assessed during EDA and expressed through regularization and tree splits rather than manual removal [ADR-07, ADR-10].

## 3.14 Sale Information (4 features)

**Columns:** `MoSold`, `YrSold`, `SaleType`, `SaleCondition`

**Business meaning.** When the sale happened (month and year), the type of sale (for example conventional warranty deed, new construction, court officer deed), and the condition of the sale (normal, abnormal such as foreclosure or short sale, partial/new home, family sale, and others).

**Modeling relevance.** This group needs the most care, because it mixes information available at listing time with information that only exists once the deal closes.

- `YrSold` is kept. It serves as the **valuation year**: the year for which the estimate is made. It is also needed to compute ages [ADR-02, ADR-07].
- `MoSold` is kept as provided. EDA investigates whether any seasonal pattern exists.
- `SaleType` and `SaleCondition` are **excluded as features** because they describe the transaction's outcome, not the property (Section 11) [ADR-02].

## 3.15 Structure Summary

| Group | Features | Dominant modeling concern |
|---|---|---|
| Location | 4 | Neighborhood as a bundled location signal |
| Lot | 8 | Skewed `LotArea`; truly unknown `LotFrontage` |
| Building type | 3 | `MSSubClass` is a code, not a number |
| Dimensions and rooms | 9 | Strongest size signal; scope variable; collinearity |
| Overall quality/condition | 3 | Strongest single predictor (`OverallQual`) |
| Age and renovation | 2 | Convert years to ages; remodel-date quirk |
| Basement | 11 | Absent-basement `NA`s; part of total size |
| Garage | 8 | Absent-garage `NA`s; `GarageYrBlt` replaced |
| Exterior | 9 | Quality scales; rare roof materials; parsing of `MasVnrType` |
| Utilities and systems | 5 | Near-constant columns; one unknown `Electrical` |
| Interior amenities | 3 | Kitchen quality; absent-fireplace `NA`s |
| Outdoor | 8 | Zero-inflated areas; rare pools |
| Miscellaneous | 2 | Very sparse |
| Sale information | 4 | Leakage risk; valuation year |
| **Total** | **79** | |

---

# 4. EDA Objectives

EDA exists to answer specific questions whose answers shape modeling. Each question below is tied to the documented property it verifies (Sections 5 to 11), the evidence deliverable that will answer it (Section 13), and the decision it informs. EDA is complete when every question has a recorded answer supported by evidence in E-35 (DOC-01 AC-010, AC-013).

## 4.1 Data Integrity

| # | Question | Evidence | Informs |
|---|---|---|---|
| Q1 | Does the raw file match the expected shape (2,930 × 82), column names, and types? | E-01 | Ingestion schema [ADR-04] |
| Q2 | Are all category values within the data dictionary's allowed codes? | E-02 | Allowed-value config [ADR-04] |
| Q3 | Are `Id` and `PID` unique and is every `SalePrice` positive? | E-03 | Schema checks [ADR-04] |
| Q4 | Does the CSV parser change missing-value counts depending on its defaults? | E-04 | Parsing contract (DOC-01 FR-003) |

## 4.2 Target

| # | Question | Evidence | Informs |
|---|---|---|---|
| Q5 | How skewed and heavy-tailed is `SalePrice`, and how much does `log1p` reduce this? | E-05, E-06 | Log target [ADR-02] |
| Q6 | Is residual spread larger for expensive homes on the raw scale than on the log scale? | E-07 | Metric choice [ADR-12] |

## 4.3 Missing Values

| # | Question | Evidence | Informs |
|---|---|---|---|
| Q7 | For each column with `NA`, does it mean "absent" or "unknown"? | E-09, E-10 | Two-layer strategy [ADR-05] |
| Q8 | Are "absent" patterns internally consistent (for example, all garage columns `NA` together)? | E-11 | Semantic filler rules [ADR-05] |
| Q9 | Does missingness in `LotFrontage` relate to price? | E-12 | Missing-indicator value [ADR-05] |

## 4.4 Features

| # | Question | Evidence | Informs |
|---|---|---|---|
| Q10 | Which numeric features are strongly skewed or zero-inflated? | E-13, E-14 | Power transform in the linear branch [ADR-08] |
| Q11 | Which features correlate most with log price, and which are strongly correlated with each other? | E-15, E-16, E-17 | Regularized linear candidates; feature engineering [ADR-07, ADR-10] |
| Q12 | Which categorical features are near-constant or have rare categories? | E-19, E-20 | Unseen-category handling; expectations about learnability [ADR-08] |
| Q13 | Do quality scales relate monotonically to price? | E-23 | Ordinal mapping [ADR-07] |
| Q14 | Are relationships mostly linear in log space, or are there thresholds and interactions? | E-17, E-21, E-28 | Choice of both linear and tree candidates [ADR-10] |

## 4.5 Outliers and Scope

| # | Question | Evidence | Informs |
|---|---|---|---|
| Q15 | Which properties fall above 4,000 sq ft, and how do their prices compare with the size trend? | E-24, E-25, E-26 | Scope rule [ADR-06] |
| Q16 | Are there other extreme values, and are they legitimate? | E-18 | Confirming no statistical outlier removal [ADR-06] |

## 4.6 Feature Engineering and Leakage

| # | Question | Evidence | Informs |
|---|---|---|---|
| Q17 | Does each approved engineered feature show the expected relationship with log price? | E-27, E-28, E-30 | Feature approval and ablation [ADR-07] |
| Q18 | Which columns would not be known at listing time? | E-31 | Leakage exclusions [ADR-02] |

## 4.7 Time

| # | Question | Evidence | Informs |
|---|---|---|---|
| Q19 | How are sales distributed across 2006–2010, and does price level shift across years? | E-32, E-33, E-34 | Temporal diagnostic [ADR-09] |

---

# 5. Target Variable Analysis

## 5.1 Distribution of SalePrice

**Documented property — to be verified by project EDA (E-05, E-06).** Properties of `SalePrice` in the raw file:

| Statistic | Documented value (approximate) |
|---|---|
| Minimum | $12,789 |
| Median | $160,000 |
| Mean | $180,800 |
| Maximum | $755,000 |
| Skewness | about 1.7 |
| Excess kurtosis | about 5.1 |

**What these numbers mean (assuming EDA confirms them).**

- **The mean is above the median.** A small number of expensive homes pull the average up. This is the signature of **right skew**: a long tail on the high-price side.
- **Skewness of about 1.7.** A symmetric distribution has skewness 0. Values above 1 are generally considered highly skewed.
- **Excess kurtosis of about 5.1.** A normal distribution has excess kurtosis 0. A high value means **heavy tails**: extreme prices occur more often than a bell curve would predict.

**Why house prices look like this.** Prices cannot go below zero but can rise a long way. Most homes cluster around a typical price, while a small number of large, high-quality homes sit far above. Price also tends to behave **multiplicatively**: a feature like better quality raises price by a percentage rather than by a fixed dollar amount. Multiplicative effects naturally produce right-skewed, roughly log-normal distributions.

## 5.2 Heavy-Tail Behavior

The heavy right tail has three practical consequences.

1. **Squared-error models are dominated by the tail.** A model trained to minimize squared dollar error treats a $50,000 miss on a $600,000 house as 25 times worse than a $10,000 miss on a $100,000 house, even though both are similar percentage errors. The model would spend its effort fitting a few expensive homes at the expense of the typical home.
2. **Error grows with price (heteroscedasticity).** On the raw scale, errors for expensive homes are naturally larger in dollars. Linear models assume a roughly constant error spread; this assumption breaks on raw prices.
3. **Evaluation becomes misleading.** Dollar RMSE would mostly report how well the model does on expensive homes [ADR-12].

## 5.3 The Effect of log1p

Applying `log1p(x) = log(1 + x)` to `SalePrice` is documented to reduce skewness from about 1.7 to about 0, close to symmetric (verify: E-05, E-06). On the log scale:

- A **fixed log difference means a fixed percentage difference.** A log error of 0.1 is roughly a 10% error, whether the house costs $100,000 or $600,000.
- Multiplicative effects become **additive**, which suits linear models.
- Error spread becomes much more **constant** across the price range.

**Why `log1p` rather than `log`.** For house prices (all well above zero) the two are practically identical. `log1p` is the numerically safe convention that is also defined at zero, and it matches the Kaggle evaluation metric's definition, keeping results comparable [ADR-02, ADR-12].

## 5.4 Modeling Implications

| Implication | How the architecture handles it |
|---|---|
| Train on the log scale | `TransformedTargetRegressor(func=log1p, inverse_func=expm1)` wraps every candidate [ADR-08] |
| Users need dollars | The wrapper converts predictions back to dollars, so the API and CLI never deal with logs [ADR-08, ADR-15] |
| Evaluate relative error | Primary metric is log-RMSE [ADR-12] |
| Communicate in business terms | MAE in dollars and MAPE reported as secondary metrics [ADR-12] |
| Keep price ranges balanced in validation | Stratify splits and CV folds on binned log price [ADR-09] |

## 5.5 Retransformation Bias

Converting a log-space prediction back with `expm1` gives an estimate of the **median** price for houses like the input, not the **mean**. Because the distribution is right-skewed, the median is slightly lower than the mean, so dollar predictions will be very slightly low on average. ADR-02 accepts this and requires it to be **documented, not corrected**. The model card records it [ADR-19].

## 5.6 Why ADR-02 Selected log1p(SalePrice)

In summary:

1. The business cares about percentage accuracy; log-space error measures exactly that.
2. It removes most of the skew and stabilizes error variance, which helps linear models and makes errors comparable across the price range.
3. It aligns training, selection, and the Kaggle metric, so the project optimizes and reports the same quantity.
4. The alternative, predicting raw dollars, lets a few expensive homes dominate training and evaluation [ADR-02].

**EDA must confirm (Q5, Q6):** skewness and kurtosis before and after the transform; histogram and Q–Q plot of both versions; and, from a simple model, residual spread by price on both scales (E-05, E-06, E-07).

**Consequence if confirmed:** the evidence documents the rationale for the log target exactly as designed [ADR-02]. **If not confirmed** (for example, much weaker skew than documented), the discrepancy is recorded in E-35; the log target remains the approved decision unless a superseding ADR is written.

---

# 6. Missing Value Analysis Strategy

## 6.1 Why Missing Values Need Thought

A missing value is not a single kind of thing. In Ames there are two very different reasons a cell can be empty, and treating them the same way corrupts the data. ADR-05 therefore uses two layers of handling.

## 6.2 Missing Because the Feature Is Absent

The data dictionary uses `NA` to mean **"this house does not have this feature."** A house without a pool has `NA` in `PoolQC` because there is no pool to rate. Nothing is actually unknown.

| Column(s) | Documented missing count (raw file; verify: E-09) | Meaning of NA |
|---|---|---|
| `PoolQC` | 2,917 | No pool |
| `MiscFeature` | 2,824 | No miscellaneous feature |
| `Alley` | 2,732 | No alley access |
| `Fence` | 2,358 | No fence |
| `FireplaceQu` | 1,422 | No fireplace |
| `GarageType` | 157 | No garage |
| `GarageFinish`, `GarageQual`, `GarageCond` | 159 each | No garage for 157 rows; 2 documented rows where a garage exists but these fields are unrecorded (Section 6.6) |
| `GarageYrBlt` | 159 | No garage (no build year exists) for 157 rows; unrecorded for the same 2 rows |
| `BsmtQual`, `BsmtCond`, `BsmtFinType1` | 80 each | No basement for 79 rows; 1 row with every basement field unrecorded (Section 6.6) |
| `BsmtExposure` | 83 | As above, plus 3 documented anomalous rows where a basement exists (Section 6.6) |
| `BsmtFinType2` | 81 | As above, plus 1 documented anomalous row where a basement exists (Section 6.6) |
| `MasVnrType`, `MasVnrArea` | 23 each | No masonry veneer recorded |

**Handling (Layer 1, semantic, stateless) [ADR-05]:**

- Categorical columns become the explicit category `"None"`.
- Related numeric columns (`GarageArea`, `GarageCars`, basement square footage and basement bath counts, `MasVnrArea`) become `0`.
- `GarageYrBlt` is not filled. It is replaced by `HasGarage` and `GarageAge`, with `GarageAge = 0` when there is no garage.

**Why this is correct.** "No pool" is real information, and it is informative for price: houses without garages or basements tend to be cheaper. Turning `NA` into an explicit "None" category preserves that information. Because this step only applies a fixed rule from the data dictionary, it learns nothing from data and cannot leak information, so it is safe to apply anywhere in the pipeline [ADR-05, ADR-08].

## 6.3 Missing Because the Value Is Unknown

Some values are genuinely unrecorded: the house has the attribute, but its value was not captured. Layer 2 handles missing values in columns that are **not** covered by the Layer 1 rule, plus any missing value that remains after Layer 1 (for example, unexpected missing values at serving time). Section 6.6 explains the one subtle case where an "unknown" cell falls inside a Layer 1 column.

| Column | Documented missing count (raw file; verify: E-09) | Why it is "unknown" |
|---|---|---|
| `LotFrontage` | 490 (about 17%) | Every lot touches a street somewhere; the length simply was not recorded |
| `Electrical` | 1 | Every house has an electrical system; its type is unrecorded |
| Any column at serving time | — | A future input may contain a missing value the training data never had |

**Handling (Layer 2, statistical, fitted inside the pipeline) [ADR-05, ADR-08]:**

- Numeric: median imputation **plus a missing-indicator column**.
- Categorical: most-frequent-value imputation.
- Statistics are learned only from the data the pipeline is being fitted on (the training folds during CV, the development set for holdout evaluation, all in-scope data for production).

**Why the missing indicator.** If missingness itself relates to price (for example, if frontage went unrecorded more often for certain kinds of lots), the indicator lets the model learn that pattern instead of treating imputed values as real. EDA question Q9 checks whether such a relationship exists.

**Why not neighborhood-level medians for LotFrontage.** Frontage varies by neighborhood, so a per-neighborhood median would be slightly more accurate. ADR-05 rejected it to avoid a custom group-aware fitted transformer, noting the indicator recovers much of the signal.

## 6.4 Missing Categories at Serving Time

A distinct risk is a category value that is allowed by the data dictionary but never appeared in the training data (for example, a rare roof material). This is not a missing value, but it has the same effect: the model has never seen it. It is handled by unseen-category settings: one-hot encoding ignores it in the linear branch, and ordinal encoding maps it to −1 in the tree branch [ADR-08]. Category values outside the data dictionary are rejected by validation before reaching the model [ADR-04, ADR-15].

## 6.5 A Parsing Hazard: MasVnrType

`MasVnrType` is documented to store both missing values (23 rows, stored as empty cells) and the literal text `None` (no veneer, 1,752 rows) (verify: E-04). Some versions of common CSV libraries treat the text `None` as missing by default. Depending on the library version, the reported missing count for this column can therefore jump from 23 to 1,775.

This matters for **reporting and schema checks**, not for the model: the semantic filler maps both to `"None"`, so model inputs are identical either way. The ingestion step parses the file with an explicit missing-value definition so counts are stable (DOC-01 FR-003). This is a good illustration of why data must be treated as a contract [ADR-04].

## 6.6 Internal Consistency Checks

The "absent" patterns should be internally consistent. EDA must check this (Q8):

- All four garage categoricals and `GarageYrBlt` should be `NA` for the same rows (157 documented), with `GarageArea = 0` and `GarageCars = 0` in those rows.
- The basement categoricals should be `NA` together, with `TotalBsmtSF = 0`.
- `MasVnrType` and `MasVnrArea` should be missing together.

**The basement anomalies (documented; verify: E-11).** Across the basement categorical columns, the `NA` values are documented to follow two different patterns:

1. **The majority pattern: feature absent.** 79 rows have `NA` in every basement categorical column and `TotalBsmtSF = 0`. These houses have no basement, and the `NA` is a true "feature absent" value.
2. **Four exceptional rows: a semantic inconsistency.** Three houses (`Id` 67, 1797, 2780) have a basement but no recorded `BsmtExposure`. Another (`Id` 445) has finished area in `BsmtFinSF2` but no recorded `BsmtFinType2`. In each, the basement clearly exists, so the missing cell is not "feature absent." Semantically it is an **unknown** value that happens to sit in a column whose `NA` is otherwise defined as "absent." This is a data-quality inconsistency in the source data.

**Related unrecorded rows (documented; verify: E-11).** The same situation occurs in three more rows: `Id` 1342 has every basement field unrecorded, including the square-footage and bathroom counts; `Id` 1357 and `Id` 2237 have a garage type (`Detchd`) but no recorded finish, quality, condition, or build year, and `Id` 2237 also lacks `GarageCars` and `GarageArea`. These cells also fall in Layer 1 columns and are handled exactly as described below.

**How this fits the two-layer strategy.** This could look like a contradiction: unknown values are supposed to go to Layer 2, yet these two cells are handled by the Layer 1 semantic filler. It is resolved by how ADR-05 defines the layers:

- The Layer 1 semantic filler is defined **by column**, and it is **stateless**. It applies its fixed rule to every `NA` in its listed columns, including `BsmtExposure` and `BsmtFinType2`. It does not, and by design cannot, inspect other columns to decide whether a basement exists.
- Therefore, **the semantic filler applies according to the approved project rule**, and these two cells become `"None"`. They never reach Layer 2, because Layer 2 only sees what remains missing after Layer 1.
- **EDA separately records the anomalous rows and their interpretation.** E-11 must identify each of these rows by `Id`, show the related basement or garage columns that reveal the inconsistency, and state that the filled value (`"None"` or `0`) is a known approximation for these cells. The data card records them all [ADR-19].
- The same rule applies at serving time: if a request supplies a basement area but a `null` basement exposure, the pipeline treats that field the same way.

**This clarification does not change ADR-05.** No new imputation strategy is introduced, and the rule for these columns is exactly the approved one. The expected effect is negligible (a few cells in 7 of 2,930 rows). Any future change to how such anomalies are handled would require a superseding ADR.

This is a useful lesson: real datasets contain small contradictions, and a documented, deliberate handling is better than a silent one.

## 6.7 Risks of Improper Handling

| Improper approach | What goes wrong |
|---|---|
| Fill `PoolQC` with the most common value | Nearly every house becomes "has a pool of that quality": false data |
| Fill `GarageYrBlt` with the median year | Garage-less houses appear to have a garage built in a typical year |
| Drop rows with any missing value | Almost every row has at least one `NA` (because of `PoolQC`, `MiscFeature`, `Alley`), so nearly the entire dataset would be lost |
| Drop columns with many `NA`s | The "has pool/fence/fireplace" signal is lost |
| Compute imputation statistics before splitting | Holdout information leaks into training; validation scores become optimistic |
| Impute in a notebook, separately from serving | Serving computes a different value from training: training–serving skew |

The two-layer, in-pipeline strategy avoids every row of this table [ADR-05, ADR-08].

---

# 7. Numerical Feature Analysis Strategy

## 7.1 Distribution Analysis

**What to do.** For each numeric feature (on the development set): histogram, summary statistics (min, quartiles, max, mean), skewness, and the share of values equal to zero.

**What to look for.**

- **Strong right skew** (for example `LotArea`, `GrLivArea`, `MasVnrArea`, `MiscVal`, `PoolArea`, `LowQualFinSF`, `3SsnPorch`). Linear models are sensitive to long tails because extreme values have high leverage.
- **Zero inflation**: many exact zeros and a spread of positive values (porch areas, `2ndFlrSF`, `MasVnrArea`, `PoolArea`). The zero means "feature absent," and the positive values describe its size.
- **Discrete counts** (bathrooms, bedrooms, fireplaces, `GarageCars`), which take few values.
- **Bounded ratings** (`OverallQual`, `OverallCond`, 1–10).

**Why it matters.** This analysis justifies the Yeo-Johnson power transform in the linear branch [ADR-08]. Yeo-Johnson (unlike Box-Cox) works with zeros, so it can reduce skew even in zero-inflated columns. It also explains why the tree branch skips the transform: trees split on value order, which a monotonic transform does not change. Zero-inflated columns also motivate the presence flags (`HasPool`, `Has2ndFlr`, and others) [ADR-07].

## 7.2 Variability Analysis

**What to do.** Count distinct values per feature and the share of the most common value.

**What to look for.** Near-constant numeric features, where one value covers almost every row (for example `PoolArea`, `3SsnPorch`, `LowQualFinSF`, `MiscVal`, `KitchenAbvGr`).

**Why it matters.** A near-constant feature has little information to offer and can produce unstable estimates from its few non-typical rows. The ADR's approach is not to hand-delete such features but to rely on Lasso's built-in selection, Ridge's shrinkage, and tree split selection [ADR-10], while the presence flags capture any real signal (for example `HasPool`) [ADR-07]. EDA records which features fall into this category, so their low importance is expected, not surprising.

## 7.3 Correlation Analysis

**What to do (development set only).**

1. Pearson and Spearman correlation of each numeric feature with `log1p(SalePrice)`. Spearman measures monotonic relationships and is less sensitive to outliers and non-linearity.
2. A feature-to-feature correlation matrix for the strongest predictors.
3. Scatter plots of the top predictors against log price.

**What to expect (documented; verify: E-15).** Documented as the strongest numeric correlates of price: `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`, `YearRemodAdd`.

Documented strongly inter-correlated pairs (verify: E-16):

| Pair | Why they overlap |
|---|---|
| `GarageCars` – `GarageArea` | Capacity and area measure garage size |
| `GrLivArea` – `TotRmsAbvGrd` | More space holds more rooms |
| `TotalBsmtSF` – `1stFlrSF` | The basement usually sits under the first floor |
| `YearBuilt` – `GarageYrBlt` | Garages are usually built with the house |

**Why it matters.**

- **Multicollinearity** makes unregularized linear coefficients unstable: the model cannot tell which of two near-duplicate features deserves credit. This is why the linear candidates are Ridge (which shares weight among correlated features) and Lasso (which tends to pick one) [ADR-10]. Comparing their coefficients on correlated pairs is itself a learning exercise.
- Strong correlates motivate **aggregate features**: size signals spread across several columns are combined into `TotalSF` and `TotalBath` [ADR-07].
- `GarageYrBlt`'s overlap with `YearBuilt`, plus its missing values, supports replacing it with `GarageAge` and `HasGarage` [ADR-05].

**Caution.** Correlation measures only linear (Pearson) or monotonic (Spearman) association with one feature at a time. A feature with low correlation can still matter through interactions, which is one reason tree models are also candidates [ADR-10].

## 7.4 Outlier Investigation (Numeric)

**What to do.** For key features, flag extreme values (for example beyond the 99th percentile) and inspect them against the data dictionary and against price.

**What to look for.** Whether each extreme is a **legitimate but rare property** (a very large lot, a big basement) or a **data error**. And whether any extreme values distort the size–price trend (Section 9).

**Why it matters.** ADR-06 decides that there is **no statistical outlier removal**. IQR or z-score filters would remove legitimate expensive or large homes and hide real variance. EDA confirms that extremes other than the scope rule's are legitimate and records them, so the "no removal" decision rests on evidence. Skewed tails are addressed by the log target and the linear-branch power transform, not by deletion [ADR-02, ADR-08].

## 7.5 Feature Usefulness Assessment

**What to do.** Combine the evidence into a per-feature view: correlation with log price, variability, missingness, and relationship shape (linear, threshold, or none).

**Why it matters.** EDA does not delete features (the ADR does not call for manual feature selection). Its job is to set **expectations** that later results can be checked against. If a feature expected to be weak ends up among the most important in a model, that is a warning sign worth investigating, possibly a sign of leakage or a data problem. This expectation-setting also feeds the ablation study for engineered features [ADR-07].

---

# 8. Categorical Feature Analysis Strategy

## 8.1 Category Distributions

**What to do.** For each categorical feature: a frequency table and bar chart of category counts; and, on the development set, a box plot of log price by category.

**Why it matters.** Box plots show whether categories separate prices. For example, if `KitchenQual = Ex` houses sit clearly higher than `TA` houses, the feature is informative. They also reveal whether ordinal scales behave in order (Section 8.6).

## 8.2 Cardinality

**What to do.** Count distinct values per categorical feature.

**What to expect (documented; verify: E-19).**

| Feature | Approximate number of categories |
|---|---|
| `Neighborhood` | 28 |
| `Exterior2nd` | 17 |
| `MSSubClass` (as categorical) | 16 |
| `Exterior1st` | 16 |
| `Condition1` | 9 |
| `SaleType` | 10 (excluded as a feature) |
| `Condition2`, `HouseStyle`, `RoofMatl` | 8 each |

Most other categoricals have between 2 and 7 levels.

**Why it matters.** Cardinality determines the width of one-hot encoding in the linear branch. The highest cardinality here is 28, which produces a manageable number of columns for about 2,340 development rows. This supports the ADR decision to one-hot encode rather than use target encoding, which would add leakage risk on small data [ADR-07, ADR-08]. In the tree branch, ordinal encoding produces one column per feature regardless of cardinality [ADR-08].

## 8.3 Rare Categories

**What to do.** List categories that appear in fewer than about 1% of rows (roughly 29 rows).

**What to expect (documented; verify: E-19, E-20).** Many rare levels, for example:

- `Utilities`: all but three houses have all public utilities.
- `Street`: only 12 gravel streets.
- `Condition2`, `RoofMatl`, `Heating`: dominated by one value, with several levels appearing only a handful of times.
- `Neighborhood`: some neighborhoods have only a few sales (the smallest is documented at 1 sale).

**Why it matters.**

- **Unreliable estimates.** A category seen in 2 rows gives the model almost nothing to learn from. Regularization (linear) and minimum-samples settings (trees) limit the damage [ADR-10, ADR-13].
- **Split fragility.** A rare level may appear in the holdout or in a CV fold but not in the training portion. The unseen-category settings are what keep this from crashing the pipeline or the service (NFR-006) [ADR-08].
- **Stratification limits.** Splits are stratified on price, not on categories, so rare categories can land unevenly across folds. This is an accepted source of noise in small data.

## 8.4 Frequency Imbalance

**What to do.** For each feature, record the share of its most common category.

**Why it matters.** A feature whose top category covers 99% of rows is effectively constant. Beyond rare-level issues, heavily imbalanced features mostly add noise to one-hot matrices. They are kept (the ADR does not call for manual removal), but EDA records them as low-information so that their low model importance is expected.

## 8.5 Potential Modeling Impact

| Observation | Impact | Architectural response |
|---|---|---|
| `Neighborhood` has 28 levels with strongly different price levels | Strong location signal | One-hot in linear branch; ordinal encoding in tree branch [ADR-08] |
| `MSSubClass` looks numeric but is a code | Numeric treatment would impose a false order | Treated as categorical [ADR-07] |
| Rare and unseen levels | Fitting and serving must not fail on them | `handle_unknown="ignore"`; `unknown_value=-1` [ADR-08] |
| Near-constant columns | Little information | Expected low importance; regularization and tree selection [ADR-10] |
| Many quality scales | Ordered information that one-hot would discard | Ordinal mapping [ADR-07] |

## 8.6 Ordinal Features: Checking the Order

**What to do.** For each quality/condition scale (for example `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`), plot median log price by level in the order None, Po, Fa, TA, Gd, Ex.

**What to look for.** Whether median price rises through the levels, and whether the steps are roughly even.

**Why it matters.** The ADR maps these scales to integers (None = 0, Po = 1, Fa = 2, TA = 3, Gd = 4, Ex = 5) [ADR-07]. This keeps the real order and uses one column instead of six. It assumes the order matters and treats the steps as equally spaced. EDA checks how well this assumption holds:

- For **quality** scales (`ExterQual`, `KitchenQual`, `BsmtQual`), a clear upward pattern is expected.
- For **condition** scales (`ExterCond`, `GarageCond`), the pattern is often weaker, because "typical" condition dominates and a few "excellent" condition houses are old houses that have been restored.

If a scale is not monotonic, this is recorded as a known limitation of the linear branch. Tree models can still split non-monotonically on ordinal integers, which is one of the reasons both families are candidates [ADR-10].

---

# 9. Outlier Investigation

## 9.1 Why GrLivArea Is Important

`GrLivArea` (above-grade living area in square feet) is one of the two strongest predictors of price, along with `OverallQual`. Physically it is the main measure of "how much house" a buyer gets, and its relationship with log price is close to linear across most of the data. Because it is so central, any unusual behavior at its extremes has an outsized effect on the fitted model, especially for linear models.

## 9.2 The Documented Anomaly

The dataset's author noted in the original documentation that the data contains a small number of very large houses that behave unusually, and recommended removing houses with more than 4,000 sq ft of living area. In the project's raw file, **5 houses** exceed 4,000 sq ft (`Id` 1499, 1761, 1768, 2181, 2182; the Kaggle subset contains 4 of them). The following is a documented property, to be verified by project EDA (E-24, E-25):

- **3 of them** (`Id` 1499, 2181, 2182) sold for far less than their size and quality predict. All three are recorded as **partial sales**: homes that were not finished when assessed. Their prices do not reflect a completed house of that size.
- **The other 2** (`Id` 1761, 1768) are very expensive homes, among the most expensive in the file. Their prices are plausible, but they sit at the far edge of the data, where there are too few examples to learn from.

**Why the partial sales are especially harmful.** In a scatter plot of `GrLivArea` against price, these three points sit at the extreme right with low prices. Points far from the average of a feature have high **leverage**: a linear model will tilt its fitted line toward them to reduce their large squared errors. Three points out of 2,930 can noticeably flatten the estimated size–price slope for everyone else (verify: E-26).

**If EDA finds a different count.** The scope rule itself (`GrLivArea ≤ 4000`) is fixed by ADR-06 and applies regardless. The in-scope row count of 2,925 is stated in DOC-01 (AC-006) and enforced from `configs/data.yaml`, so if project EDA found a number of affected rows other than 5, that discrepancy would be recorded in E-35 and handled through the ADR supersession process, not silently absorbed.

## 9.3 Why the Rule Excludes Properties Above 4,000 sq ft

ADR-06 applies one documented rule: **the model covers homes with `GrLivArea ≤ 4000` sq ft.** Rows above this are removed before splitting (leaving 2,925 rows), and the API still accepts such homes but flags them with `out_of_domain: true` [ADR-06, ADR-15].

Reasons:

1. **It follows the dataset author's own recommendation**, which is documented and external to this project, so it is not a result-driven choice.
2. **The high-end region is too sparse to learn.** Five houses cannot define how prices behave above 4,000 sq ft. Any prediction there is extrapolation.
3. **It protects the fit for everyone else.** Removing the high-leverage partial sales keeps them from distorting the size effect for the other 2,925 homes.
4. **It is simple, transparent, and fixed in advance.** One threshold, stated in the data card, is easy to audit.

## 9.4 Why This Is a Scope Decision, Not a Cleaning Decision

This distinction is the most important idea in this section.

A **cleaning** decision says "these rows are bad data; delete them to get better scores." If applied only during training, it produces a model evaluated on a population it was not meant to serve, or, if applied only to evaluation, flattering metrics that hide real errors.

A **scope** decision says "this model is designed and certified for homes up to 4,000 sq ft." It then applies consistently everywhere:

| Stage | How the scope rule applies |
|---|---|
| Training | Rows above 4,000 sq ft are not used |
| Evaluation | The holdout contains only in-scope homes, so metrics describe the population the model serves |
| Serving | Out-of-scope homes are still scored but explicitly flagged as `out_of_domain` |
| Documentation | The model card lists homes above 4,000 sq ft as an out-of-scope use |

Because the same rule applies at every stage, the reported metrics are honest: they describe exactly the population the model is meant for, and users are warned when they step outside it. That is why the rule is applied **before** the split, not after: it defines the population, and the split then divides that population [ADR-06, ADR-09].

## 9.5 What Is Not Done

- **No IQR or z-score filtering.** Such filters would remove legitimate expensive homes, making the model worse exactly where accuracy matters most and making reported metrics look better than they are.
- **No removal from training only.** That would evaluate on a population the model was not trained for, giving inconsistent metrics.
- **No robust losses (for example Huber).** They are an extra concept that is unnecessary when a clear scope rule solves the problem [ADR-06].

## 9.6 Modeling Implications

- Linear models will fit the size–price relationship more faithfully across the in-scope range.
- The log target and linear-branch power transform handle the remaining long tails without deleting data [ADR-02, ADR-08].
- The model makes **no reliable claim** for homes over 4,000 sq ft. This is accepted and documented [ADR-06, ADR-19].

**EDA must produce (Q15, Q16):** a scatter plot of `GrLivArea` against price (on both scales) with the 5 houses highlighted; a table of those 5 houses showing `Id`, size, price, quality, and sale condition; and a before-and-after comparison of a simple linear fit's slope with and without them (E-24, E-25, E-26). These deliverables use the raw dataset, because the 5 rows are removed before the split and so appear in neither the development set nor the holdout set. This is the ADR-06 scope-review exception in DOC-01 FR-009: the evidence documents a decision already fixed by the ADR and informs no other modeling choice.

---

# 10. Feature Engineering Discovery Process

## 10.1 Approach

ADR-07 requires every engineered feature to be:

1. motivated by an EDA finding or real-estate logic,
2. computed statelessly inside the pipeline, and
3. kept only if a cross-validation ablation shows it does not hurt.

This section documents the reasoning for each approved feature: its business meaning, the intuition, why it may help prediction, and the expected relationship with price. EDA then checks each expectation on the development set (Q17), and the ablation study provides the final evidence (DOC-01 FR-015).

*Status:* every "Expected relationship" below is a **hypothesis** grounded in real-estate logic and documented dataset properties. None is an observed project result yet. Evidence: E-27 and E-28 (development set), with the retention decision in E-30. **Consequence if confirmed:** the feature enters the ablation study with its motivation documented. **If not confirmed:** the mismatch is recorded in E-35; retention is still decided only by the ADR-07 ablation rule, not by the hypothesis.

A useful general principle: **trees and linear models benefit from engineered features in different ways.** Linear models cannot add columns together or detect "zero versus non-zero" on their own, so aggregates and flags give them information they could not otherwise use. Trees can in principle discover such patterns, but with about 2,340 rows they learn more reliably when a meaningful quantity is given directly.

## 10.2 TotalSF

**Definition.** `TotalBsmtSF + 1stFlrSF + 2ndFlrSF`.

**Business meaning.** Total usable interior floor area, above and below ground.

**Intuition.** Buyers value space, and basement space (especially finished basement) is usable space. Two houses with the same above-ground area but very different basements are not the same size. `GrLivArea` alone misses this.

**Why it may help.** Size is the strongest price driver, but it is split across several correlated columns. Combining them into one number gives linear models a single, cleaner size signal, and reduces the burden of sharing weight among collinear columns. It also directly represents the quantity buyers think in.

**Expected relationship.** Strongly positive and close to linear with log price. It is expected to correlate at least as strongly with log price as `GrLivArea` does.

## 10.3 TotalBath

**Definition.** `FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath`.

**Business meaning.** The total number of bathrooms, counting half-baths (a toilet and sink without a shower or bath) as half, and including basement bathrooms.

**Intuition.** This is how real-estate listings describe bathrooms ("2.5 baths"). Bathroom count reflects both size and convenience.

**Why it may help.** Four sparse count columns become one meaningful quantity. Basement bathrooms in particular are often zero, so on their own they are weak signals.

**Expected relationship.** Positive and roughly stepwise; each additional bathroom is associated with higher price, with diminishing gains at high counts.

## 10.4 HouseAge

**Definition.** `YrSold − YearBuilt`.

**Business meaning.** How old the house was when it sold.

**Intuition.** Buyers care about age, not about the calendar year of construction. A house built in 1990 is a 16-year-old house in 2006 and a 20-year-old house in 2010.

**Why it may help.** It expresses construction date in the terms that drive value (wear, style, systems needing replacement), and correctly accounts for the sale year. At serving time, `YrSold` is the valuation year, so age is computed for the moment of the estimate [ADR-02].

**Expected relationship.** Negative: older houses tend to sell for less. It is expected to be non-linear, with a steep drop over the first few decades and a flattening afterward, and some historic homes breaking the pattern.

## 10.5 RemodAge

**Definition.** `YrSold − YearRemodAdd`.

**Business meaning.** How many years since the house was last remodeled (or since it was built, if never remodeled).

**Intuition.** A recent renovation means updated kitchens, bathrooms, and systems, which buyers pay for.

**Why it may help.** It separates the effect of recent updating from original construction age. An old house remodeled last year may be worth more than its `HouseAge` alone suggests.

**Expected relationship.** Negative: longer since remodel means lower price.

**EDA caution.** The dataset is documented to record `YearRemodAdd` with a floor of 1950 (verify: E-29). Older houses frequently show 1950 regardless of their true history. EDA must quantify how many houses sit at this floor and interpret `RemodAge` for them carefully. This is a data-quality observation to record in the data card, not a change to the feature definition.

## 10.6 IsRemodeled

**Definition.** 1 if `YearRemodAdd` differs from `YearBuilt`, otherwise 0.

**Business meaning.** Whether the house has ever been remodeled.

**Intuition.** Remodeling signals investment and updating. It also marks houses where `RemodAge` and `HouseAge` differ meaningfully.

**Why it may help.** It gives linear models a simple switch to adjust for renovated houses, which they could not otherwise derive from two year columns.

**Expected relationship.** Weak on its own and possibly mixed, because older houses are both more likely to be remodeled and more likely to be cheaper. Its value comes mainly in combination with the age features, which is precisely the kind of question the ablation study settles [ADR-07]. The 1950 floor (Section 10.5) affects this flag too, and EDA records how often.

## 10.7 TotalPorchSF

**Definition.** `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`.

**Business meaning.** Total porch area across all porch types. (`WoodDeckSF` remains its own feature: a deck is a distinct structure from a porch.)

*Definition note:* ADR-07 names `TotalPorchSF` but does not list its components. The definition above follows the literal meaning of "porch" and the data dictionary's four porch columns. It operationalizes the approved feature name; it is not a new feature decision.

**Intuition.** Buyers value outdoor living space as a whole, and whether a porch is open, enclosed, three-season, or screened is a secondary detail.

**Why it may help.** Each porch column is individually sparse and zero-inflated. Summing them gives a denser, more stable signal.

**Expected relationship.** Mildly positive.

## 10.8 Presence Flags: HasPool, HasGarage, HasBsmt, HasFireplace, Has2ndFlr

**Definitions.** 1 if the house has the feature (pool area > 0; a garage exists; basement area > 0; fireplace count > 0; second-floor area > 0), otherwise 0.

**Business meaning.** Whether the house has each amenity at all.

**Intuition.** Having a feature is often a different kind of value from how large it is. A house with no garage is in a different market segment than one with a small garage.

**Why they help.** In zero-inflated columns, "zero" means absent and positive values mean size. A linear model given only the area column must use one slope for both, which blurs the effect. A flag lets it represent a jump for presence separately from a slope for size. `HasPool` in particular preserves the only reliable signal from pools, since only 13 houses have one [ADR-05, ADR-07].

**Expected relationships.**

| Flag | Expected relationship with price |
|---|---|
| `HasGarage` | Clearly positive; garage-less houses are markedly cheaper |
| `HasBsmt` | Positive |
| `HasFireplace` | Positive |
| `Has2ndFlr` | Mixed; two-story homes are not uniformly more expensive than one-story homes of the same total size |
| `HasPool` | Uncertain; too rare for a reliable estimate |

## 10.9 GarageAge

**Definition.** `YrSold − GarageYrBlt` when the house has a garage; 0 when it does not [ADR-05].

**Business meaning.** How old the garage was at sale.

**Intuition.** It replaces a year column that has no meaningful value for garage-less houses.

**Why it may help.** It removes the need to invent a build year for garages that do not exist. The combination with `HasGarage` matters: `GarageAge = 0` could mean either "no garage" or "brand-new garage," and `HasGarage` tells these apart.

**Expected relationship.** Negative but weak once `HouseAge` is known, because most garages are built with the house.

## 10.10 Ordinal Mapping of Quality and Condition Scales

**Definition.** For quality/condition scales on the Po–Ex scale (Section 8.6): None = 0, Po = 1, Fa = 2, TA = 3, Gd = 4, Ex = 5.

**Business meaning.** The quality or condition of a component, from absent to excellent.

**Intuition.** These ratings have a real order, and better ratings should mean higher prices.

**Why it may help.** One-hot encoding would treat "Excellent" and "Poor" as unrelated categories and split a small dataset's information across six columns. An ordered integer keeps the order and lets a linear model learn one slope per scale.

**Expected relationship.** Positive and roughly monotonic for quality scales; weaker for condition scales (Section 8.6).

## 10.11 MSSubClass as Categorical

**Definition.** `MSSubClass` is treated as a nominal category rather than a number.

**Business meaning.** A coded dwelling type (combining story count, age group, and style).

**Intuition and reason.** Its codes are labels. Treating them as numbers would tell the model that class 60 is "more" than class 30 in some meaningful quantity, which is false [ADR-07].

**Expected relationship.** No monotonic pattern; different classes have different typical price levels.

## 10.12 Techniques Deliberately Not Used

- **Automated feature generation:** produces features that cannot be explained.
- **Polynomial expansion:** creates far too many features for about 2,340 development rows.
- **Target encoding:** real leakage risk on small data; one-hot handles 28 neighborhoods well.
- **PCA:** removes interpretability without a clear benefit here [ADR-07].

The resulting feature set leaves some leaderboard score behind compared with aggressive feature engineering. That is an accepted tradeoff for explainability and learning value [ADR-07].

---

# 11. Leakage Assessment

## 11.1 What Data Leakage Is

**Data leakage** happens when a model is trained using information that would not be available at the moment it has to make a real prediction.

A simple analogy: a student practices for an exam using an answer sheet that happens to contain hints about the real exam. Their practice scores look excellent. On the real exam, without the hints, they do much worse. The practice scores were not measuring what they claimed to measure.

In ML, leakage makes validation scores look better than real-world performance. There are two common forms:

1. **Feature leakage (framing leakage):** a feature carries information about the outcome that would not exist at prediction time.
2. **Preprocessing leakage:** statistics used to transform data (such as imputation medians or scaling parameters) are computed using data from the evaluation set.

This project addresses preprocessing leakage through its pipeline design [ADR-05, ADR-08] and the leakage test (DOC-01 AC-025). This section focuses on feature leakage, which comes from the problem framing [ADR-02].

## 11.2 Why It Is Dangerous

- **Scores lie.** The model appears excellent in validation but underperforms once deployed, when the leaked information is no longer available.
- **It is silent.** A leaky model runs without errors and produces sensible-looking numbers. Nothing crashes.
- **It misleads decisions.** Model selection, deployment choices, and stakeholder trust all rest on inflated numbers.
- **It can break serving.** If a feature only exists after the sale, the API either cannot receive it or receives a guess, which breaks the relationship the model learned.

## 11.3 The Prediction Moment

The key question for any feature is: **"Will this value be known when the prediction is made?"**

The system is framed as a pricing tool used **before the sale closes**, typically when a property is listed [ADR-02]. So a feature is allowed only if it describes the property or the valuation date, not how the eventual transaction turned out.

## 11.4 Why SaleType Is Excluded

`SaleType` records the legal and financial form the completed transaction took: for example, a conventional warranty deed, a new-construction sale, a court officer deed, or various contract arrangements.

- These details are determined as the deal is negotiated and closed. When a house is first listed and priced, the eventual form of the transaction is not settled.
- It describes the **transaction**, not the **property**. The same house could sell under different sale types.
- Some sale types are documented to be associated with unusually high or low prices (for example, new-construction sales); project EDA describes this on the development set for context only (E-31). A model allowed to use `SaleType` would learn "this kind of deal ended at this price," which is not a signal it could use honestly at listing time.

Excluded as a feature [ADR-02].

## 11.5 Why SaleCondition Is Excluded

`SaleCondition` records the circumstances of the completed sale: `Normal`, `Abnorml` (for example foreclosure or short sale), `Partial` (the home was not completed when last assessed), `Family` (sale between relatives), `Alloca` (allocation), and `AdjLand` (adjoining land purchase).

- This is documented as a real price signal: abnormal and family sales are known to tend to sell below market value, and partial sales can behave unusually (context evidence, development set only: E-31).
- But it describes **how this particular transaction concluded**. The model's job is to estimate market value for the property, not to predict what a distressed or related-party deal would close at.
- Using it would train the model to rely on outcome information, which is exactly the framing leakage ADR-02 rules out.

Excluded as a feature [ADR-02].

**Important detail: the rows are kept.** Only the columns are dropped. Removing all non-normal sales would shrink the training data and bias it (for example, removing all newly built homes sold as partial sales). The model learns from all in-scope sales but is not told how each sale concluded [ADR-02].

**The honest cost.** ADR-02 acknowledges that dropping `SaleCondition` removes a real predictive signal, so the score will be slightly worse than leaderboard-optimized solutions. This is the price of a realistic deployment contract.

## 11.6 Features Checked and Kept

| Feature | Known at listing time? | Decision |
|---|---|---|
| All physical, quality, location, and amenity features | Yes: they describe the property | Kept |
| `YrSold` | Yes, as the valuation year | Kept; also used to compute ages [ADR-02] |
| `MoSold` | Yes, as the valuation month | Kept as provided |
| `SaleType` | No: transaction outcome | Excluded [ADR-02] |
| `SaleCondition` | No: transaction outcome | Excluded [ADR-02] |
| `Id`, `PID` | Not property attributes | Excluded as record identifiers |

## 11.7 Consequences If Leakage Were Allowed

- Validation scores would overstate real-world accuracy.
- Model comparisons might favor models that exploit the leaked features most effectively, rather than those that best understand the property.
- The API contract would require users to supply information they cannot know when asking for a price, which is a broken product.

## 11.8 Beyond Framing Leakage

For completeness, the project's other leakage defenses:

| Leakage path | Defense |
|---|---|
| Imputation or scaling statistics computed on evaluation data | All fitted steps live inside the pipeline and are fitted on training folds only [ADR-05, ADR-08] |
| Holdout information shaping choices | Holdout locked and evaluated once; EDA target analysis uses the development set only [ADR-09] |
| Tuning on the holdout | Tuning uses the development set only [ADR-13] |
| Target encoding | Not used [ADR-07] |

---

# 12. EDA Findings → Architectural Decisions

**Status of this section.** The rows below are **documented or expected relationships, or hypotheses, that project EDA must confirm**. They are not observed project findings. Each row shows the chain from Section 4:

> documented / expected property → evidence that will verify it → the ADR decision it supports → the consequence if confirmed

Some rows are **design principles** rather than data properties (for example, "preprocessing must be identical in training and serving"). These are marked "Design principle" in the Evidence column: they hold by construction and need no EDA evidence.

**If a row is not confirmed.** The ADR decision it supports remains in force; the discrepancy is recorded in E-35 and, where relevant, in the data card, and any change to the decision requires a superseding ADR [ADR-19]. EDA provides evidence for the approved architecture; it does not reopen it.

## 12.1 Target

| Documented / expected property (to confirm) | Evidence | Architectural decision | Consequence if confirmed |
|---|---|---|---|
| `SalePrice` is right-skewed (documented skew about 1.7) and heavy-tailed | E-05, E-06 | Train on `log1p(SalePrice)` via a target wrapper [ADR-02, ADR-08] | Near-symmetric target; stable error variance; errors comparable across price levels |
| Pricing errors matter in percentage terms | Design principle (business framing, Section 2) | Primary metric log-RMSE; MAE and MAPE secondary [ADR-12] | Selection reflects business value; results remain explainable in dollars |
| Expensive homes are rare | E-05, E-08 | Stratify splits and folds on binned log price [ADR-09] | Every fold sees the full price range; more stable CV estimates |

## 12.2 Missing Values

| Documented / expected property (to confirm) | Evidence | Architectural decision | Consequence if confirmed |
|---|---|---|---|
| Most `NA`s mean "feature absent" (pool, garage, basement, fence, fireplace, alley, misc) | E-09, E-10, E-11 | Stateless semantic filling with `"None"` / `0` [ADR-05] | Absence becomes usable information; no false data |
| A few basement and garage rows (7 documented) contain `NA`s that are semantically "unknown" inside Layer 1 columns | E-11 | Unchanged: the semantic filler applies by column under ADR-05; the anomalies are recorded, not re-routed (Section 6.6) [ADR-05, ADR-19] | Known, documented approximation with negligible effect |
| `GarageYrBlt` is undefined when there is no garage | E-09, E-11 | Replace with `HasGarage` and `GarageAge` [ADR-05] | No invented years; age expressed meaningfully |
| `LotFrontage` (documented about 17%) and `Electrical` (1) are truly unknown | E-09, E-12 | Fitted median + indicator / most frequent, inside the pipeline [ADR-05, ADR-08] | Leakage-free imputation; missingness itself available as a signal |
| `MasVnrType` missing counts depend on parser defaults | E-04 | Explicit parsing contract (DOC-01 FR-003) under the data-as-contract principle [ADR-04] | Stable counts and schema checks across environments |

## 12.3 Outliers

| Documented / expected property (to confirm) | Evidence | Architectural decision | Consequence if confirmed |
|---|---|---|---|
| 5 homes above 4,000 sq ft; 3 are partial sales priced far below trend; the region is too sparse to learn | E-24, E-25, E-26 | Scope rule `GrLivArea ≤ 4000` before split; API `out_of_domain` flag [ADR-06, ADR-15] | Undistorted fit; honest metrics for the served population; users warned outside it |
| Other extremes (large lots, large basements) are legitimate homes | E-18 | No statistical outlier removal [ADR-06] | Real variance preserved; model stays accurate for expensive homes |

## 12.4 Feature Engineering

| Documented / expected property (to confirm) | Evidence | Architectural decision | Consequence if confirmed |
|---|---|---|---|
| Size is split across basement, first-floor, and second-floor columns | E-15, E-27 | `TotalSF` [ADR-07] | Single, cleaner size signal |
| Bathroom counts are split across four sparse columns | E-13, E-27 | `TotalBath` [ADR-07] | Meaningful, listing-style quantity |
| Raw years ignore the sale year | Design principle (definition of age); E-27, E-29 for behavior | `HouseAge`, `RemodAge`, `IsRemodeled` [ADR-07] | Age expressed as buyers perceive it |
| Porch columns are individually sparse | E-13, E-27 | `TotalPorchSF` [ADR-07] | Denser outdoor-space signal |
| Amenity areas are zero-inflated | E-13, E-28 | `Has*` presence flags [ADR-07] | Presence and size modeled separately |
| Quality scales are ordered and relate monotonically to price | E-23 | Ordinal mapping Po=1 … Ex=5, None=0 [ADR-07] | Order preserved in one column per scale |
| `MSSubClass` is a code stored as a number | Data dictionary; E-19 | Treat as categorical [ADR-07] | No false numeric order |
| `SaleType` and `SaleCondition` describe the transaction outcome | Data dictionary; E-31 | Exclude as features, keep rows [ADR-02] | No framing leakage; realistic API contract |

## 12.5 Preprocessing Design

| Documented / expected property (to confirm) | Evidence | Architectural decision | Consequence if confirmed |
|---|---|---|---|
| Numeric features are skewed and zero-inflated | E-13, E-14 | Yeo-Johnson power transform + scaling in the linear branch [ADR-08] | Better-behaved inputs for linear models; zeros handled |
| Trees are unaffected by monotonic transforms and scale | Design principle (how tree splits work) | Tree branch uses no scaling or power transform [ADR-08] | Simpler tree inputs; no wasted computation |
| Categorical cardinality is at most 28 | E-19 | One-hot (linear) / ordinal (tree) encoding [ADR-08] | Manageable feature width; no target-encoding leakage |
| Rare categories may be absent from a training fold | E-20 | Unseen-category handling in both branches [ADR-08] | No crashes in CV or at serving time |
| Preprocessing must be identical in training and serving | Design principle | Single pipeline from raw input to dollars [ADR-08] | No training–serving skew |

## 12.6 Candidate Models

| Documented / expected property (to confirm) | Evidence | Architectural decision | Consequence if confirmed |
|---|---|---|---|
| Strong, roughly linear relationships in log space for the main drivers | E-15, E-17 | Ridge and Lasso candidates [ADR-10] | Strong, interpretable baselines |
| Strong multicollinearity among size and garage features | E-16 | Regularized rather than plain linear models [ADR-10] | Stable coefficients |
| Many low-information and near-constant features | E-13, E-19 | Lasso's built-in selection; tree split selection [ADR-10] | Noise suppressed without manual deletion |
| Non-linear effects (age, condition scales) and interactions (quality × size) | E-17, E-23, E-28 | Random Forest and LightGBM candidates [ADR-10] | Captures thresholds and interactions |
| Linear and tree models make different kinds of errors | Modeling phase (CV results), not EDA | Fixed-weight blend candidate, admitted only by the 1-SE rule [ADR-10, ADR-11] | Potential gain from diversity without meta-learner overfitting |
| Small data makes CV estimates noisy | Design principle (dataset size); CV standard errors in the modeling phase | Repeated 5×3 CV and the one-standard-error rule [ADR-09, ADR-11] | Selection based on real differences, not noise |
| Two dominant drivers (`OverallQual`, `GrLivArea`) | E-15 | Two-feature heuristic baseline [ADR-10] | A meaningful bar the ML system must clear |

---

# 13. EDA Deliverables

All deliverables are produced during implementation. Figures are saved to `reports/figures/`; tables are saved alongside the EDA notebooks or as MLflow artifacts; narrative conclusions appear in the EDA report notebook [ADR-17, ADR-19]. Deliverables that analyze relationships with `SalePrice` use the development set only (DOC-01 FR-009), with two documented exceptions: E-08 (split-balance check, target distribution only) and E-24 to E-26 (ADR-06 scope review, which needs the 5 rows removed before the split).

**These deliverables are where project EDA findings will live.** As of version 1.2, none has been produced. When they are, E-35 records, for every documented property stated in this document, whether it was confirmed (with the observed value) or not confirmed (with the discrepancy).

## 13.1 Data Integrity Report (full raw file; no target relationships)

| ID | Deliverable | Content |
|---|---|---|
| E-01 | Shape and schema table | Row and column counts; each column's type; comparison with the expected schema |
| E-02 | Allowed-values audit | For each categorical column, the observed values versus the data dictionary's allowed codes |
| E-03 | Key integrity checks | `Id` and `PID` uniqueness; `SalePrice > 0`; confirmed raw-file hash |
| E-04 | Parsing comparison | Per-column missing counts under the project's explicit parsing contract, with a note on how `MasVnrType` behaves under default parsing |

## 13.2 Target Analysis

| ID | Deliverable | Content |
|---|---|---|
| E-05 | Target statistics table | Min, quartiles, median, mean, max, skewness, kurtosis, on raw and log scales |
| E-06 | Target distribution figures | Histograms and Q–Q plots of `SalePrice` and `log1p(SalePrice)` |
| E-07 | Heteroscedasticity figure | Residual spread versus predicted value for a simple model, on raw and log scales |
| E-08 | Split balance check | Share of each log-price bin in the development and holdout sets (DOC-01 AC-009) |

## 13.3 Missing-Value Report

| ID | Deliverable | Content |
|---|---|---|
| E-09 | Missing-value table | Every column with missing values: count, percentage, classification (absent / unknown), and planned handling |
| E-10 | Missingness figure | Bar chart of missing percentages, colored by classification |
| E-11 | Consistency checks | Garage, basement, and masonry co-missingness checks; identification by `Id` of the documented basement and garage anomalies, the related basement or garage columns that reveal each inconsistency, and the recorded interpretation (Section 6.6) |
| E-12 | LotFrontage missingness study | Log price for rows with and without `LotFrontage` |

## 13.4 Numerical Feature Analysis

| ID | Deliverable | Content |
|---|---|---|
| E-13 | Numeric profile table | For each numeric feature: summary statistics, skewness, share of zeros, number of distinct values |
| E-14 | Distribution grid | Histograms of all numeric features |
| E-15 | Target correlation table | Pearson and Spearman correlation of each numeric feature with log price, ranked |
| E-16 | Correlation heatmap | Feature-to-feature correlations for the top 15 predictors, with the strongly correlated pairs listed |
| E-17 | Top-predictor scatter plots | Log price against the top predictors |
| E-18 | Numeric extremes table | Values beyond the 99th percentile for key features, with an assessment of legitimacy |

## 13.5 Categorical Feature Analysis

| ID | Deliverable | Content |
|---|---|---|
| E-19 | Categorical profile table | For each categorical feature: cardinality, top category and its share, number of categories under 1% |
| E-20 | Rare-category list | All categories with fewer than about 1% of rows (about 29 in the raw file) |
| E-21 | Price-by-category plots | Box plots of log price by category for each categorical feature |
| E-22 | Neighborhood table and plot | Row count and median log price per neighborhood |
| E-23 | Ordinal monotonicity plots | Median log price per level for each quality/condition scale, in None → Ex order, with a monotonicity note per scale |

## 13.6 Outlier Analysis

| ID | Deliverable | Content |
|---|---|---|
| E-24 | Scope rule figure | `GrLivArea` versus price (raw and log) with the 5 homes above 4,000 sq ft highlighted |
| E-25 | Scope rule table | The 5 homes: `Id`, `GrLivArea`, `SalePrice`, `OverallQual`, `SaleCondition` |
| E-26 | Leverage comparison | Slope of a simple linear fit of log price on `GrLivArea`, with and without the 5 homes |

## 13.7 Feature Engineering Assessment

| ID | Deliverable | Content |
|---|---|---|
| E-27 | Engineered feature table | For each approved feature: definition, expected relationship, observed correlation with log price |
| E-28 | Engineered feature plots | Log price against each engineered feature; presence flags shown as box plots |
| E-29 | Remodel-date audit | Number of rows with `YearRemodAdd = 1950`, and the effect on `RemodAge` and `IsRemodeled` |
| E-30 | Ablation results | Mean CV log-RMSE with and without each engineered feature, with the retention decision (produced in the modeling phase, per ADR-07; referenced here) |

## 13.8 Leakage and Time

| ID | Deliverable | Content |
|---|---|---|
| E-31 | Leakage review table | For each sale-information column: known at listing time?, decision, reason; plus descriptive median log price by `SaleType` and `SaleCondition` level on the development set, for context only (these columns are never model features) |
| E-32 | Sale-year distribution | Sales count by year, noting that 2010 covers January–July only |
| E-33 | Price by year | Median log price by year of sale |
| E-34 | Seasonality plot | Sales count and median log price by month |

## 13.9 EDA Report

| ID | Deliverable | Content |
|---|---|---|
| E-35 | EDA narrative notebook | Answers to every EDA question Q1–Q19 (Section 4) with references to the evidence above; a confirmation register listing every documented property in this document as confirmed (with observed value) or not confirmed (with discrepancy); and a written conclusion for each engineered feature and preprocessing branch (DOC-01 AC-013) |
| E-36 | Data card inputs | Documented known issues (parsing hazard, basement and garage inconsistencies, remodel-date floor, rare categories) handed to the data card [ADR-19] |

---

# 14. Risks and Limitations

## 14.1 Dataset Size

The 2,925 in-scope rows, and the 2,340 rows of the development set, are a small dataset. Consequences:

- **Noisy performance estimates.** The same model can score noticeably differently on different splits. This is why the project uses repeated CV, reports standard errors, and applies the one-standard-error rule [ADR-09, ADR-11].
- **Limited room for complexity.** Many features, many parameters, and aggressive tuning risk fitting noise. This motivates the small feature set, regularization, and bounded tuning budgets [ADR-07, ADR-10, ADR-13].
- **Small holdout.** 585 rows gives an unbiased estimate, but its uncertainty is still meaningful. The holdout score should be read as an estimate with a margin, not as an exact number.

## 14.2 Geographic Limitations

All data comes from Ames, Iowa: a mid-sized US college town with its own housing stock, price levels, and neighborhood structure. Location effects learned here (the `Neighborhood` categories themselves) have no meaning anywhere else. The model must not be applied to other cities [ADR-19].

## 14.3 Temporal Limitations and Generalization

Sales run from 2006 to mid-2010, a period that includes the 2008 financial crisis. Consequences:

- Price levels and relationships may have shifted during the period.
- Random splits mix years together, which can hide time effects. The temporal diagnostic (train 2006–2009, test 2010) is included precisely to expose this, and its result is reported in the model card [ADR-09].
- 2010 is documented to contain only January–July sales (verify: E-32), so the temporal test set is smaller and seasonally incomplete.
- The model has no knowledge of any market after 2010 [ADR-19].

## 14.4 Small-Sample Risks

- **Rare categories** give unreliable estimates and may be missing from some folds (Section 8.3).
- **Rare features** such as pools (documented at 13 houses) cannot be learned reliably.
- **Selection noise:** with small data, many candidates can look different by chance. The one-standard-error rule and the preference for simpler models guard against over-reading small differences [ADR-11].
- **Tuning optimism:** tuning and comparison share the same CV, so CV scores are slightly optimistic. The locked holdout corrects for this [ADR-09].

## 14.5 Potential Biases

- **Sale-type composition.** The data includes foreclosures, family sales, and partial sales. The model is not told which is which (Section 11), so its estimates reflect the average mix of sale types in the data, not purely "normal" market value.
- **Neighborhood imbalance.** Some neighborhoods have many sales and others very few. Errors will likely be larger in sparse neighborhoods; the diagnostic gates check error by neighborhood [ADR-11].
- **Price-range imbalance.** Expensive and very cheap homes are rarer. Stratification ensures they are represented in every fold, but the model still has fewer examples to learn from at the extremes [ADR-09].
- **Retransformation bias.** Dollar predictions estimate the median, not the mean, so they are very slightly low on average (Section 5.5).
- **Recording artifacts.** The 1950 floor on `YearRemodAdd` and the basement and garage recording inconsistencies (Section 6.6) are small data-quality issues that slightly affect specific features.

## 14.6 What These Limitations Mean for Use

The model is valid only as a demonstration of ML engineering on Ames homes up to 4,000 sq ft sold between 2006 and 2010. The model card states this explicitly, along with the out-of-scope uses [ADR-19].

---

# 15. EDA Conclusions

*Status:* in version 1.2, these are the **expected** conclusions, based on documented properties and the reasoning in this document. They become project EDA findings only once E-35 records them as confirmed.

## 15.1 Most Important Dataset Characteristics (documented; to be confirmed)

1. **Size and quality dominate price.** `OverallQual` and `GrLivArea` are the strongest single predictors, with total size and several quality scales close behind.
2. **The target is right-skewed and heavy-tailed**, and becomes close to symmetric on the log scale.
3. **Most missing values mean "absent,"** not "unknown." Only a few columns (mainly `LotFrontage`) are genuinely unknown.
4. **The data mixes three feature types:** continuous quantities (many skewed or zero-inflated), nominal categories (up to 28 levels), and ordered quality scales.
5. **Many features are near-constant or have rare categories**, carrying little information.
6. **Size-related and garage-related features are strongly inter-correlated.**
7. **Five very large homes** sit outside the region the data can support, three of them partial sales with anomalous prices.

## 15.2 Key Risks

- Small sample size: noisy estimates and limited room for model complexity.
- Leakage: through transaction-outcome features, or through preprocessing done outside the pipeline.
- Rare categories: unreliable estimates and possible absence from training folds.
- Narrow geography and time: no generalization beyond Ames, 2006–2010.

## 15.3 Key Opportunities

- Strong, interpretable signals make well-regularized linear models competitive and explainable.
- Domain-driven engineered features (`TotalSF`, `TotalBath`, ages, flags) can add real signal with little complexity.
- The mix of linear and non-linear structure favors comparing, and possibly blending, linear and tree models.
- The dataset's well-documented quirks provide concrete lessons in data contracts, scope rules, and leakage.

## 15.4 Implications for Modeling

- Train on `log1p(SalePrice)` and select on log-RMSE [ADR-02, ADR-12].
- Handle missing values in two layers inside one pipeline [ADR-05, ADR-08].
- Use separate preprocessing branches for linear and tree models [ADR-08].
- Compare regularized linear models, Random Forest, LightGBM, and a blend with repeated CV and the one-standard-error rule [ADR-09, ADR-10, ADR-11].
- Keep engineered features only when ablation supports them [ADR-07].
- Evaluate once on a locked holdout and report the temporal diagnostic alongside it [ADR-09].

## 15.5 Implications for Deployment

- The API must accept the full raw feature schema, with nullability matching the data dictionary's "absent" semantics, so training and serving share one contract [ADR-04, ADR-15].
- Allowed category values must come from the same configuration as the ingestion schema, so the two cannot drift apart [ADR-04].
- The pipeline must tolerate categories that are valid but unseen in training [ADR-08].
- Homes above 4,000 sq ft must be flagged as `out_of_domain` [ADR-06, ADR-15].
- `YrSold` acts as the valuation year and must be supplied with each request, because age features depend on it [ADR-02, ADR-07].
- The model card must communicate the geographic, temporal, and size limits, the retransformation bias, and the temporal diagnostic result [ADR-19].
