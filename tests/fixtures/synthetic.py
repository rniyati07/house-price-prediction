"""Synthetic raw file for CI smoke training (DOC-05 RC-04, §5.5; DOC-03 DN-17).

CI has no dataset, so the CI smoke stage runs on a committed synthetic file that follows
``configs/schema.yaml``: raw-file headers (``source_name``), the ``NA`` / empty-cell missing
tokens, allowed codes and ranges, and plausible relationships (garage and basement fields
agree, years are ordered, sale price rises with size and quality). It is generated once by
this helper and committed as ``tests/fixtures/synthetic_raw.csv``; it contains no real rows.

    python -m tests.fixtures.synthetic generate          # (re)write the committed fixture
    python -m tests.fixtures.synthetic prepare <root>    # an isolated smoke project for CI
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from house_price.config import SchemaConfig, load_project_config
from house_price.data.load import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
FIXTURE = Path(__file__).resolve().parent / "synthetic_raw.csv"
N_ROWS, N_OUT_OF_SCOPE, SEED = 600, 2, 42
SCOPE_THRESHOLD = 4000


def _schema() -> SchemaConfig:
    return load_project_config(CONFIG_DIR, REPO_ROOT).schema


def generate(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """``n`` synthetic rows in canonical column names (the last rows are out of scope)."""
    schema, rng = _schema(), np.random.default_rng(seed)

    def choice(name: str, size: int = n) -> np.ndarray:
        return rng.choice(np.array(schema.by_name(name).allowed_values, dtype=object), size)

    f: dict[str, object] = {"Id": np.arange(1, n + 1)}
    f["PID"] = [f"{5_000_000_000 + i:010d}" for i in range(n)]
    for name in [s.name for s in schema.columns if s.allowed_values and s.dtype == "category"]:
        f[name] = choice(name)
    f["MSSubClass"] = choice("MSSubClass").astype(int)
    f["Utilities"], f["Street"], f["Electrical"] = "AllPub", "Pave", "SBrkr"
    f["Alley"], f["Fence"], f["MiscFeature"], f["PoolQC"] = None, None, None, None
    yr_sold = rng.integers(2006, 2011, n)
    built = np.minimum(rng.integers(1900, 2009, n), yr_sold)
    remod = np.minimum(np.maximum(built, 1950) + rng.integers(0, 15, n), yr_sold)
    month = np.where(yr_sold == 2010, rng.integers(1, 8, n), rng.integers(1, 13, n))
    f.update(YearBuilt=built, YearRemodAdd=remod, YrSold=yr_sold, MoSold=month)
    qual = rng.integers(3, 11, n)
    f.update(OverallQual=qual, OverallCond=rng.integers(3, 10, n))
    first = rng.integers(600, 1900, n)
    second = np.where(rng.random(n) < 0.4, rng.integers(300, 1100, n), 0)
    first[-N_OUT_OF_SCOPE:], second[-N_OUT_OF_SCOPE:] = 2300, 2000  # GrLivArea 4300 > 4000
    f.update({"1stFlrSF": first, "2ndFlrSF": second, "LowQualFinSF": 0})
    living = first + second
    f["GrLivArea"] = living
    has_bsmt = rng.random(n) < 0.95
    fin1 = np.where(has_bsmt, rng.integers(0, 800, n), 0).astype(float)
    unf = np.where(has_bsmt, rng.integers(100, 800, n), 0).astype(float)
    f.update(BsmtFinSF1=fin1, BsmtFinSF2=0.0, BsmtUnfSF=unf, TotalBsmtSF=fin1 + unf)
    f["BsmtFullBath"] = np.where(has_bsmt, rng.integers(0, 2, n), 0).astype(float)
    f["BsmtHalfBath"] = 0.0
    for name in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
        f[name] = np.where(has_bsmt, f[name], None)  # type: ignore[call-overload]
    f["BsmtFinType2"] = np.where(has_bsmt, "Unf", None)  # type: ignore[call-overload]
    has_garage = rng.random(n) < 0.94
    cars = np.where(has_garage, rng.integers(1, 4, n), 0)
    f.update(GarageCars=cars.astype(float),
             GarageArea=np.where(has_garage, cars * 250 + rng.integers(-40, 40, n), 0).astype(float),
             GarageYrBlt=np.where(has_garage, built, np.nan))  # fmt: skip
    for name in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
        f[name] = np.where(has_garage, f[name], None)  # type: ignore[call-overload]
    fireplaces = rng.integers(0, 3, n)
    f["Fireplaces"] = fireplaces
    f["FireplaceQu"] = np.where(fireplaces > 0, f["FireplaceQu"], None)  # type: ignore[call-overload]
    vnr = choice("MasVnrType")
    f["MasVnrType"] = vnr
    f["MasVnrArea"] = np.where(vnr == "None", 0, rng.integers(50, 400, n)).astype(float)
    f["LotFrontage"] = np.where(rng.random(n) < 0.15, np.nan, rng.integers(30, 120, n)).astype(
        float
    )
    f["LotArea"] = rng.integers(3000, 20000, n)
    f.update(FullBath=rng.integers(1, 4, n), HalfBath=rng.integers(0, 2, n),
             BedroomAbvGr=rng.integers(1, 6, n), KitchenAbvGr=1,
             TotRmsAbvGrd=rng.integers(4, 11, n))  # fmt: skip
    for name in ("WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"):
        f[name] = np.where(rng.random(n) < 0.4, rng.integers(20, 300, n), 0)
    f.update(PoolArea=0, MiscVal=0, SaleType="WD", SaleCondition="Normal")
    log_price = (10.2 + 0.00045 * living + 0.09 * qual + 0.0002 * (fin1 + unf)
                 - 0.003 * (yr_sold - built) + rng.normal(0, 0.12, n))  # fmt: skip
    f["SalePrice"] = np.round(np.exp(log_price)).astype(int)
    frame = pd.DataFrame({name: f[name] for name in schema.names})
    return frame


def write_raw(frame: pd.DataFrame, path: Path = FIXTURE) -> Path:
    """Write in the raw-file layout: source headers; ``NA`` for categories, empty numerics."""
    schema, out = _schema(), frame.copy()
    for spec in schema.columns:
        if spec.dtype in ("category", "string"):
            out[spec.name] = out[spec.name].where(out[spec.name].notna(), "NA")
        elif spec.dtype == "int":
            out[spec.name] = out[spec.name].astype("int64")
    out = out.rename(columns={s.name: s.source_name for s in schema.columns})
    out.to_csv(path, index=False, lineterminator="\n", na_rep="")
    return path


def prepare(root: Path, fixture: Path = FIXTURE) -> Path:
    """An isolated smoke project in ``root``: the real schema/features/models/validation
    configuration and a ``data.yaml`` that points at the synthetic raw file."""
    config_dir, raw = root / "configs", root / "data" / "raw" / "train.csv"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture, raw)
    for name in ("schema.yaml", "validation.yaml", "features.yaml", "models.yaml"):
        shutil.copyfile(CONFIG_DIR / name, config_dir / name)
    data = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
    frame = pd.read_csv(fixture)
    data["dataset_name"] = "synthetic CI fixture (tests/fixtures/synthetic_raw.csv)"
    data["raw_sha256"] = sha256_file(raw)
    data["scope"]["expected_rows_after"] = int((frame["Gr Liv Area"] <= SCOPE_THRESHOLD).sum())
    (config_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_dir


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "generate":
        print(write_raw(generate()))
    elif command == "prepare" and len(sys.argv) == 3:
        print(prepare(Path(sys.argv[2]).resolve()))
    else:
        sys.exit("usage: python -m tests.fixtures.synthetic generate | prepare <root>")
