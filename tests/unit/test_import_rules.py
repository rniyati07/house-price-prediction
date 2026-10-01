"""Import rule (DOC-04 §9.1, §18.1; NFR-024; DOC-05 M11 task 10): ``house_price.api`` may
import ``house_price.config``, ``house_price.persistence`` and ``house_price.data.schema``
(plus its own modules), and nothing from ``features``, ``pipelines``, ``models`` or
``evaluation``, so serving cannot reimplement preprocessing or feature engineering."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import house_price.api

API_DIR = Path(house_price.api.__file__).parent
FORBIDDEN = ("house_price.features", "house_price.pipelines", "house_price.models",
             "house_price.evaluation")  # fmt: skip
ALLOWED = ("house_price.api", "house_price.config", "house_price.persistence",
           "house_price.data.schema")  # fmt: skip


def project_imports(path: Path) -> set[str]:
    """Every ``house_price`` module imported by ``path`` (top level or inside functions)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import inside house_price.api
                found.add("house_price.api." + (node.module or ""))
            elif node.module:
                found.add(node.module)
                found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return {name for name in found if name.split(".")[0] == "house_price"}


def _within(name: str, packages: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in packages)


def test_api_modules_exist() -> None:
    names = {p.stem for p in API_DIR.glob("*.py")}
    assert names == {"__init__", "settings", "schemas", "logging", "predict", "errors", "app"}


def test_api_imports_nothing_forbidden() -> None:
    for path in sorted(API_DIR.glob("*.py")):
        imports = project_imports(path)
        assert not [n for n in imports if _within(n, FORBIDDEN)], path.name
        assert all(_within(n, ALLOWED) for n in imports), (path.name, imports)


def test_importing_the_service_loads_no_ml_implementation_module() -> None:
    """Transitively too: importing the app (before any artifact is loaded) pulls in none of
    the forbidden packages (the unpickled model brings its own classes at load time)."""
    code = ("import sys, house_price.api.app, house_price.api.predict\n"
            f"bad = [m for m in sys.modules if m.startswith({FORBIDDEN!r})]\n"
            "print(bad); sys.exit(1 if bad else 0)")  # fmt: skip
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_rule_detects_a_violation(tmp_path: Path) -> None:
    sample = tmp_path / "bad.py"
    sample.write_text("from house_price.features.engineer import FeatureEngineer\n"
                      "def f():\n    import house_price.pipelines.build\n")  # fmt: skip
    imports = project_imports(sample)
    assert _within("house_price.features.engineer", FORBIDDEN)
    assert {"house_price.features.engineer", "house_price.pipelines.build"} <= imports
