"""Score-level fusion: the weight simplex, the grid search and stacking."""

from __future__ import annotations

import numpy as np
import pytest

from voice_pathology_detection.fusion.score_fusion import (
    StackingFuser,
    fuse_scores,
    grid_search_weights,
    simplex_grid,
)


def test_the_grid_covers_the_simplex_and_nothing_else() -> None:
    grid = simplex_grid(3, step=0.25)
    assert grid.shape[1] == 3
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid >= 0.0)
    # Compositions of 4 units into 3 parts: C(4 + 2, 2) = 15.
    assert grid.shape[0] == 15


def test_the_corners_of_the_simplex_are_included() -> None:
    grid = simplex_grid(3, step=0.5)
    corners = {tuple(row) for row in grid}
    assert (1.0, 0.0, 0.0) in corners
    assert (0.0, 0.0, 1.0) in corners


def test_a_single_source_needs_no_search() -> None:
    assert simplex_grid(1).tolist() == [[1.0]]


def test_the_grid_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="n_sources must be positive"):
        simplex_grid(0)
    with pytest.raises(ValueError, match="step must lie"):
        simplex_grid(2, step=0.0)
    with pytest.raises(ValueError, match="does not divide 1"):
        simplex_grid(2, step=0.3)


def test_blending_normalises_the_weights() -> None:
    scores = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    assert fuse_scores(scores, {"a": 1.0, "b": 1.0}).tolist() == [0.5, 0.5]
    assert fuse_scores(scores, {"a": 3.0, "b": 1.0}).tolist() == [0.75, 0.25]


def test_blending_rejects_invalid_weights() -> None:
    scores = {"a": [1.0], "b": [0.0]}
    with pytest.raises(ValueError, match="no weight given"):
        fuse_scores(scores, {"a": 1.0})
    with pytest.raises(ValueError, match="must not be negative"):
        fuse_scores(scores, {"a": -1.0, "b": 2.0})
    with pytest.raises(ValueError, match="sum to zero"):
        fuse_scores(scores, {"a": 0.0, "b": 0.0})


def test_sources_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="but the first source has"):
        fuse_scores({"a": [1.0, 2.0], "b": [1.0]}, {"a": 1.0, "b": 1.0})


def test_non_finite_scores_are_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        fuse_scores({"a": [np.nan]}, {"a": 1.0})


def test_the_search_prefers_the_source_that_is_right() -> None:
    y = np.array([0, 0, 1, 1])
    scores = {
        "good": np.array([0.1, 0.2, 0.8, 0.9]),
        "misleading": np.array([0.9, 0.8, 0.2, 0.1]),
    }
    best = grid_search_weights(scores, y, metric="uar", step=0.1)
    assert best.score == pytest.approx(1.0)
    assert best.weights["good"] > best.weights["misleading"]
    assert sum(best.weights.values()) == pytest.approx(1.0)


def test_two_partial_systems_fuse_better_than_either_alone() -> None:
    y = np.array([0, 0, 1, 1])
    # Each source gets one example wrong, but on different examples.
    scores = {"a": np.array([0.4, 0.9, 0.6, 0.7]), "b": np.array([0.3, 0.2, 0.4, 0.8])}
    alone_a = grid_search_weights({"a": scores["a"]}, y, step=0.5).score
    alone_b = grid_search_weights({"b": scores["b"]}, y, step=0.5).score
    both = grid_search_weights(scores, y, step=0.1).score
    assert both >= max(alone_a, alone_b)


def test_the_search_records_every_grid_point() -> None:
    y = np.array([0, 1])
    scores = {"a": np.array([0.1, 0.9]), "b": np.array([0.2, 0.8])}
    result = grid_search_weights(scores, y, step=0.25)
    assert len(result.table) == 5
    assert set(result.table.columns) == {"a", "b", "uar"}
    assert result.table["uar"].is_monotonic_decreasing
    assert result.to_dict()["metric"] == "uar"


@pytest.mark.parametrize("metric", ["uar", "accuracy", "roc_auc"])
def test_every_metric_can_be_optimised(metric: str) -> None:
    y = np.array([0, 0, 1, 1])
    scores = {"a": np.array([0.1, 0.2, 0.8, 0.9])}
    assert grid_search_weights(scores, y, metric=metric, step=0.5).score == pytest.approx(1.0)


def test_the_search_validates_its_inputs() -> None:
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="unknown metric"):
        grid_search_weights({"a": [0.1, 0.9]}, y, metric="f1")
    with pytest.raises(ValueError, match="but the scores have"):
        grid_search_weights({"a": [0.1, 0.9]}, [0, 1, 1])
    with pytest.raises(ValueError, match="both classes"):
        grid_search_weights({"a": [0.1, 0.9]}, [1, 1])
    with pytest.raises(ValueError, match="at least one score source"):
        grid_search_weights({}, y)


def test_stacking_learns_to_prefer_the_reliable_source() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 40)
    scores = {
        "sharp": y + rng.normal(0.0, 0.15, 80),
        "noisy": y + rng.normal(0.0, 1.5, 80),
    }
    fuser = StackingFuser().fit(scores, y)
    assert fuser.coefficients["sharp"] > fuser.coefficients["noisy"]
    assert fuser.predict_proba(scores).shape == (80,)
    assert (fuser.predict(scores) == y).mean() > 0.9


def test_stacking_reports_what_it_needs() -> None:
    fuser = StackingFuser()
    with pytest.raises(ValueError, match="must be fitted"):
        fuser.predict_proba({"a": [0.1]})
    with pytest.raises(ValueError, match="must be fitted"):
        _ = fuser.coefficients

    y = np.array([0, 0, 1, 1])
    fuser.fit({"a": [0.1, 0.2, 0.8, 0.9]}, y)
    with pytest.raises(ValueError, match="missing score"):
        fuser.predict_proba({"b": [0.1, 0.2, 0.8, 0.9]})


def test_stacking_checks_the_label_length() -> None:
    with pytest.raises(ValueError, match="but the scores have"):
        StackingFuser().fit({"a": [0.1, 0.9]}, [0, 1, 1])
