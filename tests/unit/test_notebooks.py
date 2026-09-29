"""Review-style checks for the M3 notebooks (DOC-05 M3 "Tests"; AC-011, AC-012; M3-2, M3-3, M3-6).

These replace unit tests for the notebooks themselves: they search the notebook sources for
the holdout and for function or class definitions, and confirm each notebook was executed.
"""

from __future__ import annotations

import ast
import json
import re

import pytest

from tests.conftest import REPO_ROOT

NOTEBOOK_DIR = REPO_ROOT / "notebooks"
EXPECTED = ["01_data_integrity", "02_target", "03_missing_values", "04_numeric",
            "05_categorical", "06_outliers_scope", "07_feature_engineering", "08_leakage_time",
            "99_eda_report"]  # fmt: skip
# E-08 (split balance) is the permitted exception; 99 only reports the resulting split sizes.
HOLDOUT_MENTION_ALLOWED = {"02_target", "99_eda_report"}
HOLDOUT_PATH = re.compile(r"holdout\.csv|holdout_path")
EDA_SOURCES = sorted((REPO_ROOT / "src" / "house_price" / "eda").glob("*.py"))


def _notebook(stem: str) -> dict[str, object]:
    return json.loads((NOTEBOOK_DIR / f"{stem}.ipynb").read_text(encoding="utf-8"))


def _code(stem: str) -> str:
    cells = _notebook(stem)["cells"]
    assert isinstance(cells, list)
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_expected_notebooks_exist() -> None:
    present = sorted(p.stem for p in NOTEBOOK_DIR.glob("*.ipynb"))
    assert present == EXPECTED


@pytest.mark.parametrize("stem", EXPECTED)
def test_notebook_defines_no_functions_or_classes(stem: str) -> None:
    """AC-012 / M3-3: all logic is imported from the package (FR-010)."""
    tree = ast.parse(_code(stem))
    defined = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda)]  # fmt: skip
    assert not defined


@pytest.mark.parametrize("stem", EXPECTED)
def test_notebook_imports_the_package(stem: str) -> None:
    assert "from house_price.eda import" in _code(stem)


@pytest.mark.parametrize("stem", EXPECTED)
def test_holdout_is_not_used(stem: str) -> None:
    """AC-011 / M3-2: no notebook references the holdout path; only the permitted
    exceptions mention the holdout at all (DOC-05 M3 recommended verification)."""
    text = (NOTEBOOK_DIR / f"{stem}.ipynb").read_text(encoding="utf-8")
    assert not HOLDOUT_PATH.search(text)
    if stem not in HOLDOUT_MENTION_ALLOWED:
        assert "holdout" not in text.lower()


def test_eda_package_never_references_the_holdout_path() -> None:
    offenders = [p.name for p in EDA_SOURCES if HOLDOUT_PATH.search(p.read_text(encoding="utf-8"))]
    assert not offenders


def test_training_and_serving_code_do_not_import_eda() -> None:
    """The EDA package is analysis only: nothing else in ``src`` depends on it."""
    package = REPO_ROOT / "src" / "house_price"
    offenders = [
        str(p.relative_to(package))
        for p in package.rglob("*.py")
        if "eda" not in p.relative_to(package).parts
        and "house_price.eda" in p.read_text(encoding="utf-8")
    ]
    assert not offenders


@pytest.mark.parametrize("stem", EXPECTED)
def test_notebook_was_executed_without_errors(stem: str) -> None:
    """M3-6 evidence: committed with fresh outputs and no error output."""
    cells = [c for c in _notebook(stem)["cells"] if c["cell_type"] == "code"]  # type: ignore[union-attr, index]
    assert all(c["execution_count"] is not None for c in cells)
    outputs = [o for c in cells for o in c["outputs"]]
    assert outputs
    assert not [o for o in outputs if o["output_type"] == "error"]
