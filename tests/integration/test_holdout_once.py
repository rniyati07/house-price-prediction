"""Holdout preconditions and once-only protection (DOC-03 §12.2; DOC-05 M9-3; AC-038):
``evaluate`` refuses without a completed passing review for the current selection record,
refuses a second final evaluation for the same record, and (real mode) a dirty tree.
Exercised in smoke mode and on copies; the real holdout is never involved."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from house_price.evaluation import holdout
from house_price.evaluation.holdout import EvaluationError, run_evaluate
from house_price.models import selection
from house_price.tracking import Lineage, Tracker
from tests.conftest import SmokeProject


def _tracker(project: SmokeProject, smoke: bool = True) -> Tracker:
    record = project.train.selection_record
    assert record is not None
    lineage = Lineage(pipeline_run_id=record["pipeline_run_id"], git_commit="a" * 40,
                      git_dirty=False, data_sha256="b" * 64, split_manifest_sha256="c" * 64,
                      config_hash="d" * 64, seed=42)  # fmt: skip
    return Tracker((project.root / "mlruns").resolve().as_uri(), lineage,
                   experiment_override="smoke" if smoke else None)  # fmt: skip


def test_first_smoke_evaluation_succeeded_and_a_second_is_refused(
    smoke_project: SmokeProject,
) -> None:
    assert smoke_project.evaluation["final_holdout"]
    with pytest.raises(EvaluationError, match="already exists .* evaluated once"):
        run_evaluate(smoke_project.root / "configs", smoke_project.root, smoke=True)


def test_a_draft_review_is_refused(smoke_project: SmokeProject, tmp_path: Path) -> None:
    record = smoke_project.train.selection_record
    assert record is not None
    draft = selection.write_review(selection.draft_review(record["selection_record_id"], {}),
                                   tmp_path / "review.yaml")  # fmt: skip
    pre = holdout.check_preconditions(record, draft, _tracker(smoke_project), smoke=True,
                                      dirty=False)  # fmt: skip
    assert any("status is 'draft'" in p for p in pre.problems)
    assert any("not signed" in p for p in pre.problems)


def test_a_review_for_another_record_is_refused(
    smoke_project: SmokeProject, tmp_path: Path
) -> None:
    record = smoke_project.train.selection_record
    assert record is not None
    other = selection.write_review(selection.smoke_review("another-record"), tmp_path / "r.yaml")
    pre = holdout.check_preconditions(record, other, _tracker(smoke_project), smoke=True,
                                      dirty=False)  # fmt: skip
    assert any("not the current" in p for p in pre.problems)


def test_clean_tree_is_enforced_in_real_mode_and_recorded_in_smoke(
    smoke_project: SmokeProject, tmp_path: Path
) -> None:
    record = smoke_project.train.selection_record
    assert record is not None
    review = tmp_path / "r.yaml"
    selection.write_review(selection.smoke_review(record["selection_record_id"]), review)
    real = holdout.check_preconditions(record, review, _tracker(smoke_project, smoke=False),
                                       smoke=False, dirty=True)  # fmt: skip
    assert "the git working tree is not clean" in real.problems
    smoke = holdout.check_preconditions(record, review, _tracker(smoke_project, smoke=False),
                                        smoke=True, dirty=True)  # fmt: skip
    assert smoke.checks["git_dirty"] is True and smoke.checks["clean_tree_enforced"] is False
    assert "the git working tree is not clean" not in smoke.problems


def test_real_evaluate_refuses_before_touching_any_holdout(
    smoke_project: SmokeProject, tmp_path: Path
) -> None:
    """A real-mode evaluate on a copy with a draft review refuses at the preconditions; the
    copy has no holdout file at all, so any attempt to read one would fail differently."""
    root = tmp_path / "project"
    shutil.copytree(smoke_project.root / "configs", root / "configs")
    shutil.copytree(smoke_project.root / "data", root / "data")
    source = smoke_project.root / "artifacts/smoke/selection"
    target = root / "reports/selection"
    target.mkdir(parents=True)
    shutil.copy2(source / selection.RECORD_NAME, target / selection.RECORD_NAME)
    record = selection.read_record(target / selection.RECORD_NAME)
    selection.write_review(selection.draft_review(record["selection_record_id"], {}),
                           target / selection.REVIEW_NAME)  # fmt: skip
    assert not (root / "data/processed/holdout.csv").exists()
    with pytest.raises(EvaluationError, match="refuses to run") as caught:
        run_evaluate(root / "configs", root, smoke=False)
    assert "draft" in str(caught.value)
    # even a signed smoke review cannot approve a real evaluation
    selection.write_review(selection.smoke_review(record["selection_record_id"]),
                           target / selection.REVIEW_NAME)  # fmt: skip
    with pytest.raises(EvaluationError, match="smoke review cannot approve"):
        run_evaluate(root / "configs", root, smoke=False)
    assert not (root / "reports/evaluation").exists()
    review = yaml.safe_load((target / selection.REVIEW_NAME).read_text(encoding="utf-8"))
    assert review["smoke"] is True
