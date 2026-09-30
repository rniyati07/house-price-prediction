"""Model selection (DOC-03 §11.4 to §11.7, DN-08, DN-19; DOC-05 M9-1; DOC-01 AC-035,
AC-036). Synthetic score tables: every expected answer is worked out from the documented
rule in the test's comment, never from the real M8 numbers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from house_price.models import selection
from house_price.models.selection import CandidateScore, ReviewError

TIERS = {"ridge": 1, "lasso": 1, "random_forest": 2, "lightgbm": 3, "blend": 4}


def table(**means_and_ses: tuple[float, float]) -> list[CandidateScore]:
    """Candidates with (mean, SE); baselines untiered, the rest with their DOC-03 tier."""
    scores = [CandidateScore("dummy_median", None, 0.40, 0.002, [], {}),
              CandidateScore("linear_2feat", None, 0.19, 0.002, [], {})]  # fmt: skip
    for name, (mean, se) in means_and_ses.items():
        scores.append(CandidateScore(name, TIERS[name], mean, se, [mean] * 15, {}))
    return scores


def test_best_is_also_simplest() -> None:
    # ridge is best (0.110); nothing simpler exists -> ridge
    r = selection.select(table(ridge=(0.110, 0.002), lasso=(0.115, 0.002),
                               random_forest=(0.130, 0.003), lightgbm=(0.120, 0.002),
                               blend=(0.112, 0.002)))  # fmt: skip
    assert r.selected.name == "ridge" and r.best.name == "ridge"
    assert r.threshold == pytest.approx(0.112) and not r.blend_admitted


def test_simpler_model_within_one_se_wins() -> None:
    # lightgbm best 0.100 (se 0.005): t = 0.105; ridge 0.104 <= t and tier 1 < 3 -> ridge
    r = selection.select(table(ridge=(0.104, 0.002), lasso=(0.130, 0.002),
                               random_forest=(0.140, 0.003), lightgbm=(0.100, 0.005),
                               blend=(0.099, 0.002)))  # fmt: skip
    assert r.best.name == "lightgbm" and r.selected.name == "ridge"
    assert [c.name for c in r.admissible] == ["ridge", "lightgbm"]


def test_simpler_model_just_outside_one_se_loses() -> None:
    # t = 0.100 + 0.005 = 0.105; ridge 0.1051 > t -> not admissible -> lightgbm
    r = selection.select(table(ridge=(0.1051, 0.002), lasso=(0.130, 0.002),
                               random_forest=(0.140, 0.003), lightgbm=(0.100, 0.005),
                               blend=(0.099, 0.002)))  # fmt: skip
    assert r.selected.name == "lightgbm"


def test_threshold_boundary_is_inclusive() -> None:
    # mean <= t (DOC-03 §11.4 step 3), with exactly representable values: t = 0.5 + 0.25
    r = selection.select(
        table(ridge=(0.75, 0.01), lasso=(0.9, 0.01), random_forest=(0.9, 0.01),
              lightgbm=(0.5, 0.25))
    )  # fmt: skip
    assert r.threshold == 0.75 and r.selected.name == "ridge"


def test_ridge_lasso_exact_tie_goes_to_ridge() -> None:
    r = selection.select(table(ridge=(0.110, 0.002), lasso=(0.110, 0.003),
                               random_forest=(0.130, 0.003), lightgbm=(0.120, 0.002)))  # fmt: skip
    assert r.selected.name == "ridge"


def test_within_a_tier_the_lower_mean_wins() -> None:
    # both linear models admissible (t = 0.102); lasso has the lower mean -> lasso
    r = selection.select(table(ridge=(0.1015, 0.002), lasso=(0.1010, 0.002),
                               random_forest=(0.130, 0.003), lightgbm=(0.100, 0.002)))  # fmt: skip
    assert r.selected.name == "lasso"


def test_tier_ordering_random_forest_before_lightgbm() -> None:
    # lightgbm best 0.100 (se 0.005) -> t 0.105; rf 0.104 admissible; linear too high -> rf
    r = selection.select(table(ridge=(0.150, 0.002), lasso=(0.160, 0.002),
                               random_forest=(0.104, 0.003), lightgbm=(0.100, 0.005)))  # fmt: skip
    assert r.selected.name == "random_forest"


def test_blend_admitted_replaces_the_one_se_result() -> None:
    # best single ridge 0.110 (se 0.002): bar 0.108; blend 0.1079 < bar -> blend selected
    r = selection.select(table(ridge=(0.110, 0.002), lasso=(0.111, 0.002),
                               random_forest=(0.130, 0.003), lightgbm=(0.112, 0.002),
                               blend=(0.1079, 0.002)))  # fmt: skip
    assert r.blend_admitted and r.selected.name == "blend"
    assert r.blend_bar == pytest.approx(0.108)
    assert "replaces" in r.reason


def test_blend_at_the_bar_is_rejected_and_excluded() -> None:
    # admission is strict (mean_blend < m - se): blend exactly at the bar is excluded
    r = selection.select(table(ridge=(0.110, 0.002), lasso=(0.111, 0.002),
                               random_forest=(0.130, 0.003), lightgbm=(0.112, 0.002),
                               blend=(0.108, 0.002)))  # fmt: skip
    assert not r.blend_admitted and r.selected.name == "ridge"
    assert "blend" not in [c.name for c in r.pool]  # excluded before §11.4


def test_blend_better_but_within_one_se_is_rejected() -> None:
    # blend 0.1085 beats ridge 0.110 but not by more than one SE (bar 0.108) -> ridge
    r = selection.select(table(ridge=(0.110, 0.002), lasso=(0.111, 0.002),
                               random_forest=(0.130, 0.003), lightgbm=(0.112, 0.002),
                               blend=(0.1085, 0.002)))  # fmt: skip
    assert not r.blend_admitted and r.selected.name == "ridge"


def test_baselines_are_never_eligible() -> None:
    scores = table(ridge=(0.110, 0.002), lasso=(0.111, 0.002), random_forest=(0.130, 0.003),
                   lightgbm=(0.112, 0.002))  # fmt: skip
    scores[0] = CandidateScore("dummy_median", None, 0.01, 0.001, [], {})  # impossibly good
    r = selection.select(scores)
    assert r.selected.name == "ridge" and "dummy_median" not in [c.name for c in r.pool]


def test_selection_is_deterministic_and_order_free() -> None:
    scores = table(ridge=(0.110, 0.002), lasso=(0.110, 0.002), random_forest=(0.130, 0.003),
                   lightgbm=(0.112, 0.002), blend=(0.109, 0.002))  # fmt: skip
    first, reversed_ = selection.select(scores), selection.select(list(reversed(scores)))
    assert first.selected.name == reversed_.selected.name == "ridge"
    assert first.threshold == reversed_.threshold


def _record() -> dict[str, object]:
    scores = table(ridge=(0.110, 0.002), lasso=(0.111, 0.002), random_forest=(0.130, 0.003),
                   lightgbm=(0.112, 0.002), blend=(0.1079, 0.002))  # fmt: skip
    result = selection.select(scores)
    hyper = {"linear": "ridge", "tree": "lightgbm", "weights": [0.5, 0.5],
             "linear_params": {"alpha": 10.0}, "tree_params": {"num_leaves": 17}}  # fmt: skip
    return selection.selection_record(result, scores, pipeline_run_id="run-1",
                                      hyperparameters=hyper, provenance={"source": "test"})  # fmt: skip


def test_selection_record_has_every_documented_field() -> None:
    record = _record()
    for key in (
        "selection_record_id",
        "pipeline_run_id",
        "candidates",
        "best_single",
        "threshold",
        "blend_admitted",
        "selected",
        "baseline_margins",
        "created_at",
    ):
        assert key in record, key  # DOC-03 §11.7
    assert len(record["candidates"]) == 7  # type: ignore[arg-type]
    assert record["selected"]["name"] == "blend"  # type: ignore[index]
    margins = record["baseline_margins"]
    assert margins["linear_2feat"]["selected_minus_baseline"] == pytest.approx(0.1079 - 0.19)  # type: ignore[index]
    assert margins["dummy_median"]["baseline_se"] == 0.002  # type: ignore[index]
    assert record["eligible"] == ["ridge", "lasso", "random_forest", "lightgbm", "blend"]


def test_scores_from_comparison_table() -> None:
    table_ = pd.DataFrame({
        "candidate": ["dummy_median", "ridge"], "tier": [float("nan"), 1.0],
        "cv_mean": [0.4, 0.11], "cv_se": [0.002, 0.003], "cv_mae": [1.0, 2.0],
        "cv_mape": [3.0, 4.0], "cv_r2": [0.1, 0.9], "fold_01": [0.4, 0.1], "fold_02": [0.4, 0.12],
    })  # fmt: skip
    scores = selection.scores_from_comparison(table_)
    assert scores[0].tier is None and not scores[0].eligible
    assert scores[1].tier == 1 and scores[1].fold_scores == [0.1, 0.12]


# ------------------------------------------------------------------- review (DN-19)


def _write(path: Path, review: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(review, sort_keys=False), encoding="utf-8")
    return path


def _signed(record_id: str) -> dict[str, object]:
    return {"selection_record_id": record_id, "status": "final", "reviewer": "A. Reviewer",
            "reviewed_at": "2026-10-01",
            "gates": {g: {"result": "pass", "reasoning": "read the plot"}
                      for g in selection.REVIEW_GATES}}  # fmt: skip


def test_a_signed_passing_review_for_the_record_is_accepted(tmp_path: Path) -> None:
    path = _write(tmp_path / "r.yaml", _signed("rec-1"))
    assert selection.review_problems(path, "rec-1", smoke=False) == []
    assert selection.check_review(path, "rec-1")["reviewer"] == "A. Reviewer"


def test_the_generated_draft_never_passes(tmp_path: Path) -> None:
    path = selection.write_review(selection.draft_review("rec-1", {"n": 1}), tmp_path / "r.yaml")
    draft = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert draft["status"] == "draft" and draft["reviewer"] is None
    problems = selection.review_problems(path, "rec-1", smoke=False)
    assert any("status is 'draft'" in p for p in problems)
    assert any("not signed" in p for p in problems)
    with pytest.raises(ReviewError):
        selection.check_review(path, "rec-1")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(selection_record_id="other"), "not the current"),
        (lambda r: r.update(reviewer=""), "not signed"),
        (lambda r: r.pop("reviewed_at"), "no reviewed_at"),
        (lambda r: r["gates"]["price_decile_bias"].update(result="fail"), "not 'pass'"),
        (lambda r: r["gates"]["residual_pattern"].update(reasoning=" "), "no reasoning"),
        (lambda r: r["gates"].pop("neighborhood_error"), "neighborhood_error"),
    ],
)  # fmt: skip
def test_review_refusals(tmp_path: Path, change: object, message: str) -> None:
    review = _signed("rec-1")
    change(review)  # type: ignore[operator]
    problems = selection.review_problems(_write(tmp_path / "r.yaml", review), "rec-1", smoke=False)
    assert any(message in p for p in problems), problems


def test_missing_review_is_refused(tmp_path: Path) -> None:
    assert "not found" in selection.review_problems(tmp_path / "none.yaml", "rec-1", smoke=False)[0]


def test_smoke_review_only_approves_smoke(tmp_path: Path) -> None:
    path = selection.write_review(selection.smoke_review("rec-1"), tmp_path / "r.yaml")
    assert selection.review_problems(path, "rec-1", smoke=True) == []
    assert any("smoke review cannot approve" in p
               for p in selection.review_problems(path, "rec-1", smoke=False))  # fmt: skip


def test_hyperparameters_for_tuned_and_blend() -> None:
    best = {n: {"best_params": {"p": i}, "estimator_params": {"p": i, "fixed": True}}
            for i, n in enumerate(("ridge", "lasso", "random_forest", "lightgbm"))}  # fmt: skip
    assert selection.hyperparameters_for("lasso", best, "ridge") == {
        "tuned_params": {"p": 1},
        "estimator_params": {"p": 1, "fixed": True},
    }
    blend = selection.hyperparameters_for("blend", best, "ridge")
    assert blend["linear"] == "ridge" and blend["linear_params"] == {"p": 0}
    assert blend["tree_params"] == {"p": 3} and blend["weights"] == [0.5, 0.5]
    with pytest.raises(selection.SelectionError):
        selection.hyperparameters_for("blend", best, None)
