"""Combining the scores of several systems.

When more than one model scores the same subjects (an acoustic model, a
baseline, a model trained on a different normalisation), their scores can be
blended.  I use a convex combination: the weights are non-negative and sum to
one, so the blend stays on the scale of its inputs and each weight reads as a
share.  With the two or three systems this is used for, a grid over the
probability simplex is exhaustive, and every point is scored with the same
metric the rest of the project reports.

The other option is stacking, where a logistic regression learns the
combination instead of a grid search picking it.  It is strictly more
expressive, so it has to be fitted on held-out scores, never on the ones it is
evaluated on.

Examples
--------
>>> import numpy as np
>>> scores = {"a": np.array([0.9, 0.1, 0.8]), "b": np.array([0.2, 0.3, 0.7])}
>>> best = grid_search_weights(scores, np.array([1, 0, 1]), step=0.5)
>>> round(sum(best.weights.values()), 6)
1.0
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from speechdsp import uar

_LOGGER = logging.getLogger(__name__)

#: Metrics that :func:`grid_search_weights` can optimise.
FUSION_METRICS: tuple[str, ...] = ("uar", "accuracy", "roc_auc")


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Best weights found by a grid search.

    Attributes
    ----------
    weights : dict
        Weight of every source, summing to one.
    score : float
        Value of the optimised metric at those weights.
    metric : str
        Which metric was optimised.
    table : pandas.DataFrame
        Every grid point that was evaluated, with one column per source plus
        the metric, sorted best first.
    """

    weights: dict[str, float]
    score: float
    metric: str
    table: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the result."""
        return {
            "weights": {name: float(w) for name, w in self.weights.items()},
            "score": float(self.score),
            "metric": self.metric,
            "n_grid_points": len(self.table),
        }


def simplex_grid(n_sources: int, step: float = 0.1) -> np.ndarray:
    """Enumerate the convex weight vectors on a regular grid.

    Parameters
    ----------
    n_sources : int
        Number of systems to blend; must be at least one.
    step : float, optional
        Grid spacing, default 0.1.  ``1 / step`` must be close to an integer.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_points, n_sources)``.  Every row is non-negative
        and sums to one.

    Raises
    ------
    ValueError
        If ``n_sources`` is not positive or ``step`` does not divide one.

    Examples
    --------
    >>> simplex_grid(2, step=0.5).tolist()
    [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
    """
    if n_sources < 1:
        raise ValueError(f"n_sources must be positive, got {n_sources}")
    if not 0.0 < step <= 1.0:
        raise ValueError(f"step must lie in (0, 1], got {step}")
    divisions = round(1.0 / step)
    if abs(divisions * step - 1.0) > 1e-9:
        raise ValueError(f"step={step} does not divide 1 into a whole number of parts")

    if n_sources == 1:
        return np.ones((1, 1))

    # Stars and bars: choose n_sources - 1 bar positions among divisions + bars.
    rows: list[list[float]] = []
    for bars in combinations(range(divisions + n_sources - 1), n_sources - 1):
        previous = -1
        counts: list[int] = []
        for bar in bars:
            counts.append(bar - previous - 1)
            previous = bar
        counts.append(divisions + n_sources - 2 - previous)
        rows.append([count / divisions for count in counts])
    return np.asarray(sorted(rows), dtype=np.float64)


def _stack_scores(scores: Mapping[str, object]) -> tuple[list[str], np.ndarray]:
    """Validate a mapping of score vectors and stack it into a matrix.

    A list of score vectors is an easy thing to pass by mistake, so it gets a
    ``TypeError`` that says what is expected instead of an index error later.
    """
    if not isinstance(scores, Mapping):
        raise TypeError(
            "scores must be a mapping from a system name to its score vector, for "
            f"example {{'dnn': ..., 'svm': ...}}, got {type(scores).__name__}"
        )
    if not scores:
        raise ValueError("at least one score source is required")
    names = list(scores)
    columns = []
    length: int | None = None
    for name in names:
        column = np.asarray(scores[name], dtype=np.float64).ravel()
        if length is None:
            length = column.size
        elif column.size != length:
            raise ValueError(
                f"source {name!r} has {column.size} score(s) but the first source has {length}"
            )
        if not np.all(np.isfinite(column)):
            raise ValueError(f"source {name!r} contains non-finite scores")
        columns.append(column)
    return names, np.column_stack(columns)


def fuse_scores(scores: Mapping[str, object], weights: Mapping[str, float]) -> np.ndarray:
    """Blend several score vectors with the given weights.

    Parameters
    ----------
    scores : mapping
        One score vector per system, all of the same length.
    weights : mapping
        Weight of every system.  They are normalised to sum to one, and none
        may be negative.

    Returns
    -------
    numpy.ndarray
        The blended score vector.

    Raises
    ------
    ValueError
        If a system is missing a weight, a weight is negative, or the weights
        sum to zero.

    Examples
    --------
    >>> import numpy as np
    >>> fuse_scores({"a": [1.0, 0.0], "b": [0.0, 1.0]}, {"a": 1.0, "b": 1.0}).tolist()
    [0.5, 0.5]
    """
    names, matrix = _stack_scores(scores)
    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(f"no weight given for source(s) {missing}")
    vector = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    if np.any(vector < 0.0):
        raise ValueError(f"weights must not be negative, got {vector.tolist()}")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("the weights sum to zero")
    return matrix @ (vector / total)


def _metric_value(metric: str, y_true: np.ndarray, fused: np.ndarray, threshold: float) -> float:
    """Evaluate one fusion candidate with the requested metric."""
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, fused))
    predicted = (fused >= threshold).astype(np.int64)
    if metric == "uar":
        return float(uar(y_true, predicted))
    return float(accuracy_score(y_true, predicted))


def grid_search_weights(
    scores: Mapping[str, object],
    y_true: object,
    *,
    metric: str = "uar",
    step: float = 0.1,
    threshold: float = 0.5,
) -> FusionResult:
    """Search the weight simplex for the best blend.

    Parameters
    ----------
    scores : mapping
        One score vector per system; each entry is the score of the positive
        class, of the same length as ``y_true``.
    y_true : array_like
        True binary labels.
    metric : str, optional
        One of :data:`FUSION_METRICS`, default ``"uar"``.
    step : float, optional
        Grid spacing on the simplex, default 0.1.
    threshold : float, optional
        Decision threshold applied to the fused score for the two
        threshold-dependent metrics, default 0.5.

    Returns
    -------
    FusionResult
        The best weights, their score and the whole evaluated grid.

    Raises
    ------
    ValueError
        If the sources disagree in length, the metric is unknown, or the labels
        do not cover two classes.

    Examples
    --------
    >>> import numpy as np
    >>> scores = {"a": [0.9, 0.1], "b": [0.1, 0.9]}
    >>> grid_search_weights(scores, [1, 0], step=0.5).score
    1.0
    """
    if metric not in FUSION_METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose one of {FUSION_METRICS}")
    names, matrix = _stack_scores(scores)
    truth = np.asarray(y_true).ravel()
    if truth.size != matrix.shape[0]:
        raise ValueError(f"y_true has {truth.size} entries but the scores have {matrix.shape[0]}")
    if np.unique(truth).size < 2:
        raise ValueError(
            f"fusion needs both classes to be present, found {np.unique(truth).tolist()}"
        )

    grid = simplex_grid(len(names), step=step)
    fused = matrix @ grid.T  # (n_samples, n_grid_points), one blend per column
    values = np.asarray(
        [_metric_value(metric, truth, fused[:, i], threshold) for i in range(grid.shape[0])]
    )

    table = pd.DataFrame(grid, columns=names)
    table[metric] = values
    table = table.sort_values(metric, ascending=False, ignore_index=True)

    best = int(np.argmax(values))
    weights = {name: float(grid[best, position]) for position, name in enumerate(names)}
    _LOGGER.info(
        "best fusion over %d grid point(s): %s -> %s %.4f",
        grid.shape[0],
        weights,
        metric,
        values[best],
    )
    return FusionResult(weights=weights, score=float(values[best]), metric=metric, table=table)


class StackingFuser:
    """Learn the combination of several score vectors with logistic regression.

    Parameters
    ----------
    C : float, optional
        Inverse regularisation strength, default 1.
    random_state : int, optional
        Seed, default 0.

    Attributes
    ----------
    sources_ : list of str
        Source names, in the column order the model was fitted on.
    model_ : sklearn.linear_model.LogisticRegression
        The fitted combiner.

    Notes
    -----
    Unlike the grid search, the learned weights are unconstrained: they may be
    negative, which lets one system correct another rather than only dilute it.
    The price is that the combiner is itself a fitted model and must see scores
    that were held out from both of its inputs.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.array([0] * 20 + [1] * 20)
    >>> scores = {"a": y + rng.normal(0, 0.2, 40), "b": y + rng.normal(0, 0.5, 40)}
    >>> fuser = StackingFuser().fit(scores, y)
    >>> fuser.predict_proba(scores).shape
    (40,)
    """

    def __init__(self, C: float = 1.0, random_state: int = 0) -> None:
        self.C = C
        self.random_state = random_state

    def fit(self, scores: Mapping[str, object], y_true: object) -> StackingFuser:
        """Fit the combiner on held-out scores.

        Parameters
        ----------
        scores : mapping
            One score vector per system.
        y_true : array_like
            True binary labels.

        Returns
        -------
        StackingFuser
            ``self``, fitted.
        """
        self.sources_, matrix = _stack_scores(scores)
        truth = np.asarray(y_true).ravel()
        if truth.size != matrix.shape[0]:
            raise ValueError(
                f"y_true has {truth.size} entries but the scores have {matrix.shape[0]}"
            )
        self.model_ = LogisticRegression(
            C=self.C, random_state=self.random_state, max_iter=1000
        ).fit(matrix, truth)
        return self

    def _matrix(self, scores: Mapping[str, object]) -> np.ndarray:
        """Stack the scores in the order the combiner was fitted on."""
        if not hasattr(self, "model_"):
            raise ValueError("this fuser must be fitted before it can predict")
        missing = [name for name in self.sources_ if name not in scores]
        if missing:
            raise ValueError(f"missing score(s) for source(s) {missing}")
        return np.column_stack(
            [np.asarray(scores[name], dtype=np.float64).ravel() for name in self.sources_]
        )

    def predict_proba(self, scores: Mapping[str, object]) -> np.ndarray:
        """Return the fused probability of the positive class.

        Parameters
        ----------
        scores : mapping
            One score vector per system.

        Returns
        -------
        numpy.ndarray
            Probabilities of shape ``(n_samples,)``.
        """
        matrix = self._matrix(scores)
        return np.asarray(self.model_.predict_proba(matrix)[:, 1])

    def predict(self, scores: Mapping[str, object]) -> np.ndarray:
        """Return the fused hard decision.

        Parameters
        ----------
        scores : mapping
            One score vector per system.

        Returns
        -------
        numpy.ndarray
            Predicted labels of shape ``(n_samples,)``.
        """
        matrix = self._matrix(scores)
        return np.asarray(self.model_.predict(matrix))

    @property
    def coefficients(self) -> dict[str, float]:
        """Learned weight of every source."""
        if not hasattr(self, "model_"):
            raise ValueError("this fuser must be fitted first")
        return dict(zip(self.sources_, self.model_.coef_[0].tolist(), strict=True))
