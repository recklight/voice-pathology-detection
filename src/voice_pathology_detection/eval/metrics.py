"""Evaluation metrics.

Every number in a report is computed here from the predictions and saved as
JSON, so the figures and tables of a run all start from the same values.

The main metric is the unweighted average recall (UAR), the mean of the
per-class recalls over the classes present in ``y_true`` (the same number as
:func:`sklearn.metrics.balanced_accuracy_score`).  With 50 normal and 150
pathological recordings, plain accuracy rewards a model that just prefers the
larger class; UAR does not.  Sensitivity, specificity and the two predictive
values are reported alongside it because they are what a screening test is
judged on.  A model can look strong on frames and weak on people, so the
single-stage routes compute every metric at both levels.

Examples
--------
>>> report = evaluate([0, 0, 1, 1], [0, 1, 1, 1])
>>> round(report["uar"], 3), round(report["sensitivity"], 3)
(0.75, 1.0)

References
----------
.. [1] B. Schuller, S. Steidl, and A. Batliner, "The INTERSPEECH 2009 Emotion
       Challenge", *INTERSPEECH*, pp. 312-315, 2009.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from speechdsp import confusion_report

_LOGGER = logging.getLogger(__name__)

#: Scalar metrics summarised across folds by default.
SUMMARY_KEYS: tuple[str, ...] = (
    "uar",
    "accuracy",
    "macro_f1",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "roc_auc",
)


def predictive_values(
    y_true: object,
    y_pred: object,
    positive_label: int = 1,
) -> tuple[float, float]:
    """Positive and negative predictive value of a two-class decision.

    Where sensitivity and specificity condition on the true class, the
    predictive values condition on the prediction: they answer "given that the
    test said pathological, how often is it right?", which is the question a
    clinician actually asks.  They depend on prevalence, so they are only
    comparable between experiments on the same class balance.

    Parameters
    ----------
    y_true, y_pred : array_like
        True and predicted labels of equal length.
    positive_label : int, optional
        Label treated as positive, default 1.

    Returns
    -------
    ppv : float
        ``TP / (TP + FP)``, or ``nan`` when nothing was predicted positive.
    npv : float
        ``TN / (TN + FN)``, or ``nan`` when nothing was predicted negative.

    Raises
    ------
    ValueError
        If the two arrays have different lengths or are empty.

    Examples
    --------
    >>> predictive_values([1, 1, 0, 0], [1, 0, 0, 0])
    (1.0, 0.6666666666666666)
    """
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if true.size != pred.size:
        raise ValueError(f"y_true has {true.size} entries but y_pred has {pred.size}")
    if true.size == 0:
        raise ValueError("cannot compute predictive values from empty inputs")

    true_positive = np.count_nonzero((true == positive_label) & (pred == positive_label))
    predicted_positive = np.count_nonzero(pred == positive_label)
    true_negative = np.count_nonzero((true != positive_label) & (pred != positive_label))
    predicted_negative = int(pred.size - predicted_positive)

    ppv = true_positive / predicted_positive if predicted_positive else float("nan")
    npv = true_negative / predicted_negative if predicted_negative else float("nan")
    if not predicted_positive or not predicted_negative:
        _LOGGER.warning("the model predicted a single class; one predictive value is undefined")
    return float(ppv), float(npv)


def _roc_auc(y_true: np.ndarray, score: object, labels: np.ndarray) -> float:
    """Area under the ROC curve, binary or one-vs-rest, or ``nan``."""
    if score is None:
        return float("nan")
    values = np.asarray(score, dtype=np.float64)
    present = np.unique(y_true)
    if present.size < 2:
        _LOGGER.debug("ROC AUC is undefined with a single class present")
        return float("nan")
    try:
        if values.ndim == 1:
            return float(roc_auc_score(y_true, values))
        if values.shape[1] == 2:
            return float(roc_auc_score(y_true, values[:, 1]))
        if present.size != labels.size:
            _LOGGER.debug("not every class is present; skipping the one-vs-rest AUC")
            return float("nan")
        return float(roc_auc_score(y_true, values, multi_class="ovr", average="macro"))
    except ValueError as exc:  # pragma: no cover - defensive
        _LOGGER.debug("ROC AUC could not be computed: %s", exc)
        return float("nan")


def evaluate(
    y_true: object,
    y_pred: object,
    score: object = None,
    *,
    labels: Sequence[int] | None = None,
    positive_label: int | Collection[int] | None = 1,
    level: str = "subject",
) -> dict[str, Any]:
    """Score a set of predictions and return a JSON-serialisable report.

    Parameters
    ----------
    y_true, y_pred : array_like
        True and predicted labels of equal length.
    score : array_like, optional
        Continuous scores for the ROC area: either the positive-class score of
        shape ``(n,)`` or the full posterior of shape ``(n, n_classes)``.
    labels : sequence of int, optional
        Label order of the confusion matrix; defaults to the labels observed.
    positive_label : int, collection of int or None, optional
        What counts as positive in sensitivity, specificity, PPV and NPV,
        default 1.  A collection merges several classes into the positive
        side; the four-way routes pass the three disorders, so these metrics
        mean pathological against normal.  ``None`` leaves the four entries
        ``nan``, for a task without a normal class.
    level : str, optional
        Free-form tag recorded in the report, typically ``"frame"`` or
        ``"subject"``.

    Returns
    -------
    dict
        Keys ``level``, ``n``, ``labels``, ``accuracy``, ``uar``, ``macro_f1``,
        ``per_class_recall``, ``per_class_precision``, ``confusion_matrix``,
        ``support``, ``sensitivity``, ``specificity``, ``ppv``, ``npv`` and
        ``roc_auc``.  ``uar`` averages the recall of the classes present in
        ``y_true``, like :func:`sklearn.metrics.balanced_accuracy_score`; a
        listed class with no true sample shows a recall of 0 in
        ``per_class_recall`` but is left out of that mean.

    Raises
    ------
    ValueError
        If the inputs are empty or of different lengths.

    Examples
    --------
    >>> report = evaluate([0, 1, 2], [0, 1, 1], labels=[0, 1, 2])
    >>> report["n"], round(report["accuracy"], 3)
    (3, 0.667)
    """
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if true.size != pred.size:
        raise ValueError(f"y_true has {true.size} entries but y_pred has {pred.size}")
    if true.size == 0:
        raise ValueError("cannot evaluate empty predictions")

    label_list = (
        sorted({int(v) for v in np.unique(np.concatenate([true, pred]))})
        if labels is None
        else [int(v) for v in labels]
    )
    base = confusion_report(true, pred, labels=label_list)
    label_array = np.asarray(label_list, dtype=np.int64)

    if positive_label is None:
        sensitivity = specificity = ppv = npv = float("nan")
    else:
        positives = np.asarray(
            list(positive_label) if isinstance(positive_label, Collection) else [positive_label],
            dtype=np.int64,
        )
        binary_true = np.isin(true, positives).astype(np.int64)
        binary_pred = np.isin(pred, positives).astype(np.int64)
        sensitivity = (
            float(np.mean(binary_pred[binary_true == 1]))
            if np.any(binary_true == 1)
            else float("nan")
        )
        specificity = (
            float(np.mean(1 - binary_pred[binary_true == 0]))
            if np.any(binary_true == 0)
            else float("nan")
        )
        ppv, npv = predictive_values(binary_true, binary_pred, positive_label=1)

    report: dict[str, Any] = {
        "level": level,
        "n": int(true.size),
        "labels": label_list,
        "accuracy": float(base["accuracy"]),
        "uar": float(base["uar"]),
        "macro_f1": float(base["macro_f1"]),
        "per_class_recall": {int(k): float(v) for k, v in base["per_class_recall"].items()},
        "per_class_precision": {int(k): float(v) for k, v in base["per_class_precision"].items()},
        "confusion_matrix": np.asarray(base["confusion_matrix"], dtype=int).tolist(),
        "support": np.asarray(base["support"], dtype=int).tolist(),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "roc_auc": _roc_auc(true, score, label_array),
    }
    _LOGGER.info(
        "%s level: n=%d, UAR=%.4f, accuracy=%.4f",
        level,
        report["n"],
        report["uar"],
        report["accuracy"],
    )
    return report


def summarise_folds(
    folds: Sequence[dict[str, Any]],
    keys: Sequence[str] = SUMMARY_KEYS,
) -> dict[str, dict[str, float]]:
    """Average scalar metrics across folds, ignoring undefined entries.

    Parameters
    ----------
    folds : sequence of dict
        Per-fold reports produced by :func:`evaluate`.
    keys : sequence of str, optional
        Metrics to summarise; defaults to :data:`SUMMARY_KEYS`.

    Returns
    -------
    dict
        ``{metric: {"mean": float, "std": float, "n": int}}``, where ``std`` is
        the population standard deviation across folds and ``n`` counts the
        folds in which the metric was defined.  A metric that is undefined
        everywhere yields ``nan``.

    Raises
    ------
    ValueError
        If ``folds`` is empty.

    Examples
    --------
    >>> summary = summarise_folds([{"uar": 0.6}, {"uar": 0.8}], keys=["uar"])
    >>> round(summary["uar"]["mean"], 3), round(summary["uar"]["std"], 3)
    (0.7, 0.1)
    """
    if not folds:
        raise ValueError("cannot summarise an empty list of folds")

    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            float(fold[key])
            for fold in folds
            if key in fold and fold[key] is not None and np.isfinite(float(fold[key]))
        ]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
        else:
            summary[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return summary


def folds_to_frame(
    folds: Sequence[dict[str, Any]], keys: Sequence[str] = SUMMARY_KEYS
) -> pd.DataFrame:
    """Render per-fold reports as a table, one row per fold.

    Parameters
    ----------
    folds : sequence of dict
        Per-fold reports produced by :func:`evaluate`.
    keys : sequence of str, optional
        Columns to keep; defaults to :data:`SUMMARY_KEYS`.

    Returns
    -------
    pandas.DataFrame
        Columns ``fold``, ``level``, ``n`` and the requested metrics.
    """
    records: list[dict[str, Any]] = []
    for position, fold in enumerate(folds):
        record: dict[str, Any] = {
            "fold": fold.get("fold", position),
            "level": fold.get("level", ""),
            "n": fold.get("n", 0),
        }
        record.update({key: fold.get(key, float("nan")) for key in keys})
        records.append(record)
    return pd.DataFrame.from_records(records)


def confusion_frame(report: dict[str, Any], names: dict[int, str] | None = None) -> pd.DataFrame:
    """Render the confusion matrix of a report as a labelled table.

    Parameters
    ----------
    report : dict
        A report produced by :func:`evaluate`.
    names : dict, optional
        Display name of every label; the codes are used when omitted.

    Returns
    -------
    pandas.DataFrame
        Matrix with true classes on the rows and predicted ones on the columns.
    """
    labels = [int(label) for label in report["labels"]]
    titles = [names.get(label, str(label)) if names else str(label) for label in labels]
    return pd.DataFrame(
        np.asarray(report["confusion_matrix"], dtype=int),
        index=pd.Index(titles, name="true"),
        columns=pd.Index(titles, name="predicted"),
    )
