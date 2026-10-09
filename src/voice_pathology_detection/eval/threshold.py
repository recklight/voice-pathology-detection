"""Choosing the operating point of the detection stage from data.

A screening test is defined as much by its threshold as by its model.  A
threshold carried over from another experiment is a hyper-parameter fitted to
that experiment's evaluation set, so every rule here derives the threshold
from scores the final evaluation never sees (in the routes, an inner
validation split of the training fold), and the value is saved with the
results.

Every threshold between the lowest score called positive and the highest score
called negative gives the same rates on the selection set.  The tuned rules
return the midpoint of that interval rather than its upper end, so a new
subject scoring a little below the lowest selected positive is still called
positive.

``youden``
    Maximise Youden's J, ``sensitivity + specificity - 1``: the point furthest
    from chance, and the right default when both kinds of error cost the same.
``target_sensitivity`` / ``target_specificity``
    Pick the threshold that just reaches a requested sensitivity (or
    specificity) and, among those, the one with the best value on the other
    axis.  This is how a screening operating point is usually specified.
``fixed``
    Use a value given in the configuration, for replicating a specific setting.

Examples
--------
>>> import numpy as np
>>> y = np.array([0, 0, 1, 1])
>>> score = np.array([0.1, 0.2, 0.8, 0.9])
>>> choice = select_threshold(y, score, method="youden")
>>> choice.sensitivity, choice.specificity
(1.0, 1.0)

References
----------
.. [1] W. J. Youden, "Index for rating diagnostic tests", *Cancer*, 3(1):32-35,
       1950.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

_LOGGER = logging.getLogger(__name__)

#: Threshold selection rules implemented here.
THRESHOLD_METHODS: tuple[str, ...] = (
    "fixed",
    "youden",
    "target_sensitivity",
    "target_specificity",
)


@dataclass(frozen=True, slots=True)
class ThresholdChoice:
    """A chosen operating point and the numbers that justify it.

    Attributes
    ----------
    threshold : float
        Score at or above which a subject is called positive.
    sensitivity, specificity : float
        Rates achieved at that threshold on the data it was selected from.
    youden : float
        ``sensitivity + specificity - 1``.
    auc : float
        Area under the ROC curve of the scores, independent of the threshold;
        ``nan`` when only one class is present.
    method : str
        Rule that produced the threshold.
    target : float or None
        Operating point that was requested, for the two ``target_*`` rules.
    """

    threshold: float
    sensitivity: float
    specificity: float
    youden: float
    auc: float
    method: str
    target: float | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        """JSON-serialisable view of the choice."""
        return {
            "threshold": float(self.threshold),
            "sensitivity": float(self.sensitivity),
            "specificity": float(self.specificity),
            "youden": float(self.youden),
            "auc": float(self.auc),
            "method": self.method,
            "target": None if self.target is None else float(self.target),
        }


def _validated(y_true: object, score: object) -> tuple[np.ndarray, np.ndarray]:
    """Return validated label and score vectors for the two-class task."""
    y = np.asarray(y_true).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    if y.size != s.size:
        raise ValueError(f"y_true has {y.size} entries but score has {s.size}")
    if y.size == 0:
        raise ValueError("cannot choose a threshold from empty inputs")
    if not np.all(np.isfinite(s)):
        raise ValueError("score contains non-finite values")
    present = np.unique(y)
    if present.size != 2:
        raise ValueError(
            f"threshold selection needs both classes, found labels {present.tolist()}. "
            "Increase split.val_fraction so the inner validation set covers both."
        )
    return y, s


def roc_table(y_true: object, score: object, *, positive_label: int = 1) -> pd.DataFrame:
    """Tabulate sensitivity and specificity against the threshold.

    Parameters
    ----------
    y_true : array_like
        True labels.
    score : array_like
        Continuous score of the positive class.
    positive_label : int, optional
        Label treated as positive, default 1.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold``, ``sensitivity``, ``specificity`` and ``youden``,
        one row per candidate threshold.

    Notes
    -----
    Built from :func:`sklearn.metrics.roc_curve`, whose first row is the
    degenerate "call nothing positive" point.  It is kept because it is a valid
    operating point, but its threshold (infinite in scikit-learn 1.3 and later,
    the largest score plus one before that) is replaced by the next float above
    the largest score.  For vote ratios that lands just above 1 only when some
    subject had every frame voted pathological.
    """
    y, s = _validated(y_true, score)
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=positive_label)
    thresholds = np.asarray(thresholds, dtype=np.float64).copy()
    top = float(s.max())
    if thresholds.size and not thresholds[0] <= top:
        thresholds[0] = np.nextafter(top, np.inf)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "sensitivity": tpr,
            "specificity": 1.0 - fpr,
            "youden": tpr - fpr,
        }
    )


def select_threshold(
    y_true: object,
    score: object,
    method: str = "youden",
    *,
    target: float | None = None,
    fixed: float | None = None,
    positive_label: int = 1,
) -> ThresholdChoice:
    """Choose a decision threshold from labelled scores.

    Parameters
    ----------
    y_true : array_like
        True labels of the selection set; both classes must be present.
    score : array_like
        Continuous score of the positive class.
    method : str, optional
        One of :data:`THRESHOLD_METHODS`, default ``"youden"``.
    target : float, optional
        Sensitivity or specificity to reach, for the two ``target_*`` rules.
    fixed : float, optional
        The threshold itself, for ``method='fixed'``.
    positive_label : int, optional
        Label treated as positive, default 1.

    Returns
    -------
    ThresholdChoice
        The chosen threshold and the rates it achieves.  For the tuned rules
        the threshold is the midpoint between the lowest score called positive
        and the highest score called negative, which leaves those rates
        unchanged; when no score is called positive it stays just above the
        largest score.

    Raises
    ------
    ValueError
        If the inputs are empty or single-class, the method is unknown, or a
        required argument for the chosen method is missing.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> s = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    >>> choice = select_threshold(y, s, method="target_sensitivity", target=1.0)
    >>> choice.sensitivity
    1.0
    """
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {THRESHOLD_METHODS}")

    y, s = _validated(y_true, score)
    auc = float(roc_auc_score(y == positive_label, s))
    table = roc_table(y, s, positive_label=positive_label)

    if method == "fixed":
        if fixed is None:
            raise ValueError("method='fixed' requires the threshold to be given")
        chosen = float(fixed)
        positive = y == positive_label
        predicted = s >= chosen
        sensitivity = float(np.mean(predicted[positive])) if positive.any() else float("nan")
        specificity = float(np.mean(~predicted[~positive])) if (~positive).any() else float("nan")
        return ThresholdChoice(
            threshold=chosen,
            sensitivity=sensitivity,
            specificity=specificity,
            youden=sensitivity + specificity - 1.0,
            auc=auc,
            method=method,
        )

    if method == "youden":
        row = table.iloc[int(table["youden"].to_numpy().argmax())]
    else:
        axis = "sensitivity" if method == "target_sensitivity" else "specificity"
        other = "specificity" if axis == "sensitivity" else "sensitivity"
        if target is None:
            raise ValueError(f"method={method!r} requires a target")
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"target must lie in [0, 1], got {target}")
        feasible = table[table[axis] >= target]
        if feasible.empty:
            best = float(table[axis].max())
            _LOGGER.warning(
                "no threshold reaches %s >= %.3f (best achievable is %.3f); "
                "falling back to that best point",
                axis,
                target,
                best,
            )
            feasible = table[table[axis] >= best]
        row = feasible.iloc[int(feasible[other].to_numpy().argmax())]

    threshold = float(row["threshold"])
    below = s[s < threshold]
    if threshold <= float(s.max()) and below.size:
        # Any value in (below.max(), threshold] separates the selection set the
        # same way; the midpoint leaves room on both sides for unseen subjects.
        threshold = 0.5 * (threshold + float(below.max()))

    choice = ThresholdChoice(
        threshold=threshold,
        sensitivity=float(row["sensitivity"]),
        specificity=float(row["specificity"]),
        youden=float(row["youden"]),
        auc=auc,
        method=method,
        target=target,
    )
    _LOGGER.info(
        "selected threshold %.4f by %s (sensitivity %.3f, specificity %.3f, AUC %.3f)",
        choice.threshold,
        method,
        choice.sensitivity,
        choice.specificity,
        choice.auc,
    )
    return choice


def threshold_sweep(
    y_true: object,
    score: object,
    thresholds: object = None,
    *,
    positive_label: int = 1,
) -> pd.DataFrame:
    """Evaluate a grid of thresholds, for the operating-point figure.

    Parameters
    ----------
    y_true : array_like
        True labels.
    score : array_like
        Continuous score of the positive class.
    thresholds : array_like, optional
        Thresholds to evaluate; defaults to 101 points spanning ``[0, 1]``.
    positive_label : int, optional
        Label treated as positive, default 1.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold``, ``sensitivity``, ``specificity``, ``youden``,
        ``accuracy`` and ``n_positive``, the number of subjects the threshold
        would forward to the second stage.

    Examples
    --------
    >>> import numpy as np
    >>> sweep = threshold_sweep([0, 1], [0.2, 0.9], thresholds=[0.5])
    >>> sweep["n_positive"].tolist()
    [1]
    """
    y, s = _validated(y_true, score)
    grid = (
        np.linspace(0.0, 1.0, 101)
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float64).ravel()
    )
    positive = y == positive_label

    # Shape (n_thresholds, n_subjects): every threshold in one comparison.
    predicted = s[None, :] >= grid[:, None]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    true_positive = (predicted & positive[None, :]).sum(axis=1)
    true_negative = (~predicted & ~positive[None, :]).sum(axis=1)

    sensitivity = true_positive / n_pos if n_pos else np.full(grid.size, np.nan)
    specificity = true_negative / n_neg if n_neg else np.full(grid.size, np.nan)
    return pd.DataFrame(
        {
            "threshold": grid,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "youden": sensitivity + specificity - 1.0,
            "accuracy": (true_positive + true_negative) / float(y.size),
            "n_positive": predicted.sum(axis=1),
        }
    )
