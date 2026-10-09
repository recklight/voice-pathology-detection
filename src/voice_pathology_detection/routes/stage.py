"""The shared training and evaluation loop.

Detection, the three-way disorder stage and the single-stage four-way route
are the same procedure on different label sets, so the loop lives here once
and each route passes in its labels, model and threshold settings.

One fold of that procedure:

1. split the *files* of the training side again, grouped by subject, into a
   fitting part and an inner validation part;
2. fit the normaliser on the fitting frames only, and apply it everywhere else;
3. fit the classifier on those frames;
4. aggregate the validation frames to subjects and, for the detection task,
   choose the decision threshold there;
5. aggregate the evaluation frames to subjects and score them, at the frame
   level and the subject level separately.

The evaluation fold is touched exactly once, in step 5.  Nothing about the
model, the normalisation statistics or the threshold is derived from it.

Examples
--------
>>> from voice_pathology_detection.config import Config, ModelConfig
>>> from voice_pathology_detection.data.synthetic import make_frame_dataset
>>> from voice_pathology_detection.features.pipeline import FeatureSet
>>> X, frame_file, files = make_frame_dataset(4, 8, 6, labels=(0, 1), seed=0)
>>> features = FeatureSet(X, frame_file, files)
>>> cfg = Config().override({"split.n_splits": 2, "normalization.method": "cmvn"})
>>> result = run_stage(
...     features, cfg, task="detection", labels=(0, 1),
...     model=ModelConfig(name="gmm", params={"n_components": 1}),
...     tune_threshold=False,
... )
>>> len(result.folds)
2
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from ..config import Config, ModelConfig
from ..data.splits import (
    Fold,
    SplitError,
    check_no_group_leakage,
    subject_folds,
    train_val_split,
)
from ..eval.aggregate import AggregateResult, aggregate_frames
from ..eval.metrics import evaluate, summarise_folds
from ..eval.threshold import ThresholdChoice, select_threshold
from ..features.normalize import FrameNormalizer, make_normalizer
from ..features.pipeline import FeatureSet
from ..models.registry import build_model

_LOGGER = logging.getLogger(__name__)


class StageError(RuntimeError):
    """Raised when a stage cannot be trained or evaluated."""


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    """Everything one fold produced.

    Attributes
    ----------
    fold : int
        Fold number.
    subject_ids : numpy.ndarray
        Identifier of every evaluated subject.
    subject_true, subject_pred : numpy.ndarray
        True and predicted labels, one per subject.
    subject_proba : numpy.ndarray
        Mean posterior per subject, shape ``(n_subjects, n_classes)``.
    subject_score : numpy.ndarray
        Continuous score of the positive class, for the detection task.
    frame_metrics, subject_metrics : dict
        Reports from :func:`voice_pathology_detection.eval.metrics.evaluate`.
    threshold : ThresholdChoice or None
        Operating point chosen on the inner validation split, when applicable.
    """

    fold: int
    subject_ids: np.ndarray
    subject_true: np.ndarray
    subject_pred: np.ndarray
    subject_proba: np.ndarray
    subject_score: np.ndarray
    frame_metrics: dict[str, Any]
    subject_metrics: dict[str, Any]
    threshold: ThresholdChoice | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    """Cross-validated outcome of one stage.

    Attributes
    ----------
    task : str
        Name of the stage, used in reports and file names.
    labels : tuple of int
        Label set of the stage, in column order.
    folds : list of FoldOutcome
        Per-fold outcomes.
    predictions : pandas.DataFrame
        Out-of-fold predictions, one row per subject: ``fold``, ``subject_id``,
        ``y_true``, ``y_pred``, ``score`` and one ``proba_<label>`` column per
        class.  Every subject appears exactly once.
    frame_summary, subject_summary : dict
        Mean and standard deviation of each metric across folds.
    pooled_frame, pooled_subject : dict
        Metrics over the concatenated out-of-fold predictions: every evaluated
        frame, and every subject, scored once.
    thresholds : list of float
        Threshold chosen in each fold, for the record.
    config : dict
        The configuration the stage ran with.
    """

    task: str
    labels: tuple[int, ...]
    folds: list[FoldOutcome]
    predictions: pd.DataFrame
    frame_summary: dict[str, dict[str, float]]
    subject_summary: dict[str, dict[str, float]]
    pooled_frame: dict[str, Any]
    pooled_subject: dict[str, Any]
    thresholds: list[float] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary of the stage."""
        return {
            "task": self.task,
            "labels": list(self.labels),
            "n_folds": len(self.folds),
            "frame_summary": self.frame_summary,
            "subject_summary": self.subject_summary,
            "pooled_frame": self.pooled_frame,
            "pooled_subject": self.pooled_subject,
            "thresholds": self.thresholds,
            "folds": [
                {
                    "fold": outcome.fold,
                    "frame": outcome.frame_metrics,
                    "subject": outcome.subject_metrics,
                    "threshold": (
                        None if outcome.threshold is None else outcome.threshold.to_dict()
                    ),
                }
                for outcome in self.folds
            ],
            "config": self.config,
        }


@dataclass(slots=True)
class StageModel:
    """A fitted stage, ready for inference on new recordings.

    Attributes
    ----------
    task : str
        Name of the stage.
    normalizer : FrameNormalizer
        The fitted normaliser.
    model : sklearn.base.BaseEstimator
        The fitted classifier.
    labels : tuple of int
        Label set, in the column order of the posteriors.
    aggregation : str
        Frame-to-subject rule to apply at inference time.
    threshold : float or None
        Operating point, for the detection task.  :meth:`predict_subjects`
        applies it to the score of :attr:`positive_label`.
    positive_label : int
        Label treated as positive by the threshold rule.
    """

    task: str
    normalizer: FrameNormalizer
    model: BaseEstimator
    labels: tuple[int, ...]
    aggregation: str
    threshold: float | None = None
    positive_label: int = 1

    def predict_frames(self, X: object, group_sizes: object = None) -> np.ndarray:
        """Return frame posteriors, with the columns in :attr:`labels` order.

        Parameters
        ----------
        X : array_like
            Raw (un-normalised) frames of shape ``(n_frames, n_dim)``.
        group_sizes : array_like, optional
            Frame count per recording, needed by per-utterance normalisation.

        Returns
        -------
        numpy.ndarray
            Posteriors of shape ``(n_frames, len(labels))``.
        """
        normalised = self.normalizer.transform(X, group_sizes=group_sizes)
        return frame_posteriors(self.model, normalised, self.labels)

    def predict_subjects(self, features: FeatureSet) -> AggregateResult:
        """Predict one label per recording of a feature set.

        Parameters
        ----------
        features : FeatureSet
            Recordings to score.

        Returns
        -------
        AggregateResult
            Subject-level posteriors and decisions.  For a two-class stage
            with a stored threshold, ``pred`` is the positive label wherever
            the score reaches that threshold, as in cross-validation.
        """
        proba = self.predict_frames(features.X, group_sizes=features.group_sizes)
        result = aggregate_frames(
            proba,
            features.group_sizes,
            mode=self.aggregation,
            classes=np.asarray(self.labels),
            threshold=self.threshold,
            positive_label=self.positive_label,
        )
        if self.threshold is None or len(self.labels) != 2 or self.aggregation == "ratio_threshold":
            return result
        negative_label = next(label for label in self.labels if label != self.positive_label)
        decided = np.where(
            result.score(self.positive_label) >= self.threshold,
            self.positive_label,
            negative_label,
        )
        return dataclasses.replace(result, pred=decided.astype(np.int64), threshold=self.threshold)

    def save(self, path: str | Path) -> Path:
        """Persist the stage with :mod:`joblib`.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            The path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        _LOGGER.info("saved the fitted %s stage to %s", self.task, path)
        return path

    @staticmethod
    def load(path: str | Path) -> StageModel:
        """Load a stage written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            File to read.

        Returns
        -------
        StageModel
            The restored stage.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no such model file: {path}")
        return joblib.load(path)


def frame_posteriors(
    model: BaseEstimator,
    X: object,
    labels: Sequence[int],
) -> np.ndarray:
    """Return posteriors with one column per label, in the given order.

    A fold may train on data where a class happens to be absent, and not every
    estimator has ``predict_proba``.  Both cases end up in one matrix whose
    columns always mean the same thing.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A fitted classifier.
    X : array_like
        Features of shape ``(n_samples, n_features)``.
    labels : sequence of int
        Desired column order.

    Returns
    -------
    numpy.ndarray
        Posteriors of shape ``(n_samples, len(labels))``.  Classes the model
        never saw get a column of zeros.

    Notes
    -----
    Estimators without ``predict_proba`` fall back to a softmax of
    ``decision_function``, and failing that to a one-hot encoding of
    ``predict``.  The fallbacks are not calibrated, so the probability-averaging
    aggregation is only meaningful with a model that really has posteriors.
    """
    features = np.asarray(X, dtype=np.float64)
    order = np.asarray(labels)
    known = np.asarray(getattr(model, "classes_", order))
    out = np.zeros((features.shape[0], order.size), dtype=np.float64)

    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exponent = np.exp(scores)
        raw = exponent / exponent.sum(axis=1, keepdims=True)
    else:  # pragma: no cover - every registered model has one of the above
        hard = np.asarray(model.predict(features))
        raw = (hard[:, None] == known[None, :]).astype(np.float64)

    for position, label in enumerate(known):
        target = np.flatnonzero(order == label)
        if target.size:
            out[:, int(target[0])] = raw[:, position]
    return out


def score_mode(aggregation: str) -> str:
    """Return the aggregation mode whose score the threshold is tuned on.

    For ``mean_proba`` that score is the mean posterior.  For the two counting
    rules it is the share of positive frames, which is also what
    ``ratio_threshold`` compares with its threshold.

    Examples
    --------
    >>> score_mode("mean_proba"), score_mode("ratio_threshold")
    ('mean_proba', 'majority_vote')
    """
    return "mean_proba" if aggregation == "mean_proba" else "majority_vote"


def fit_normalizer(
    cfg: Config,
    features: FeatureSet,
    rows: np.ndarray,
) -> tuple[FrameNormalizer, FeatureSet]:
    """Fit the normaliser on the given files and return it with that subset."""
    subset = features.select(rows)
    normalizer = make_normalizer(cfg.normalization)
    normalizer.fit(subset.X, group_sizes=subset.group_sizes)
    return normalizer, subset


def apply_normalizer(normalizer: FrameNormalizer, subset: FeatureSet) -> np.ndarray:
    """Apply a fitted normaliser to a subset."""
    return normalizer.transform(subset.X, group_sizes=subset.group_sizes)


def inner_split(
    files: pd.DataFrame,
    rows: np.ndarray,
    cfg: Config,
    *,
    seed_offset: int = 0,
    context: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Split the training rows again, or report that it cannot be done.

    Parameters
    ----------
    files : pandas.DataFrame
        File index the rows point into, carrying the labels to stratify on.
    rows : numpy.ndarray
        Training rows of a fold.
    cfg : Config
        Configuration supplying the fraction, the seed and the group column.
    seed_offset : int, optional
        Added to the split seed so different folds split differently.
    context : str, optional
        Description used in the warning when the split is impossible.

    Returns
    -------
    fit_rows, val_rows : numpy.ndarray
        The two parts.  When no stratified, grouped validation set can be
        carved out of so few recordings, ``val_rows`` is empty and ``fit_rows``
        is the whole input; the caller then falls back to a fixed threshold
        rather than tuning one on data it also trains on.
    """
    try:
        return train_val_split(
            files,
            rows,
            val_fraction=cfg.split.val_fraction,
            seed=cfg.split.seed + seed_offset,
            group_column=cfg.split.group_by,
        )
    except SplitError as exc:
        _LOGGER.warning(
            "%s: no validation set can be held out (%s); the threshold falls back to a "
            "fixed value instead of being tuned",
            context or "inner split",
            exc,
        )
        return np.asarray(rows, dtype=int), np.asarray([], dtype=int)


def run_stage(
    features: FeatureSet,
    cfg: Config,
    *,
    task: str,
    labels: Sequence[int],
    model: ModelConfig,
    aggregation: str = "mean_proba",
    tune_threshold: bool = False,
    threshold_method: str = "youden",
    fixed_threshold: float | None = None,
    threshold_target: float | None = None,
    positive_label: int | Collection[int] | None = 1,
) -> StageResult:
    """Cross-validate one stage and return every metric it produced.

    Parameters
    ----------
    features : FeatureSet
        Frames, their provenance and the file index.
    cfg : Config
        Configuration supplying the split layout, the normalisation and the seed.
    task : str
        Name of the stage, used in reports.
    labels : sequence of int
        Label set of the stage, in the column order of the posteriors.
    model : ModelConfig
        Estimator to train in every fold.
    aggregation : str, optional
        Frame-to-subject rule, default ``"mean_proba"``.
    tune_threshold : bool, optional
        Choose the decision threshold on an inner validation split, default
        ``False``.  Only meaningful for a two-class stage.
    threshold_method : str, optional
        Selection rule, default ``"youden"``.
    fixed_threshold : float, optional
        The threshold itself, when ``threshold_method`` is ``"fixed"``.
    threshold_target : float, optional
        Operating point for the two ``target_*`` rules.
    positive_label : int, collection of int or None, optional
        Label treated as positive, default 1.  A two-class stage needs a single
        label; a multi-class stage may pass the set of labels that count as
        positive in sensitivity and specificity, or ``None`` to leave them
        undefined (see :func:`voice_pathology_detection.eval.metrics.evaluate`).

    Returns
    -------
    StageResult
        Per-fold and pooled metrics, plus the out-of-fold predictions.

    Raises
    ------
    StageError
        If the feature set holds fewer than two classes, or a fold cannot be
        trained.
    ValueError
        If the split cannot be built, which names the reason.

    Notes
    -----
    The model is fitted on the inner fitting part and *not* refitted after the
    threshold has been chosen, so the threshold always belongs to the model it
    is applied to.
    """
    label_tuple = tuple(int(label) for label in labels)
    present = np.unique(features.file_labels)
    if present.size < 2:
        raise StageError(
            f"stage {task!r} needs at least two classes but the feature set holds only "
            f"{present.tolist()}"
        )
    if tune_threshold and len(label_tuple) != 2:
        raise StageError(
            f"threshold tuning is only defined for two classes, stage {task!r} has "
            f"{len(label_tuple)}"
        )

    folds = subject_folds(
        features.files,
        n_splits=cfg.split.n_splits,
        seed=cfg.split.seed,
        group_column=cfg.split.group_by,
    )
    check_no_group_leakage(features.files, folds, group_column=cfg.split.group_by)

    outcomes: list[FoldOutcome] = []
    records: list[pd.DataFrame] = []
    frame_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold in folds:
        outcome, frame, frames = _run_fold(
            features,
            cfg,
            fold,
            task=task,
            labels=label_tuple,
            model_cfg=model,
            aggregation=aggregation,
            tune_threshold=tune_threshold,
            threshold_method=threshold_method,
            fixed_threshold=fixed_threshold,
            threshold_target=threshold_target,
            positive_label=positive_label,
        )
        outcomes.append(outcome)
        records.append(frame)
        frame_parts.append(frames)

    predictions = pd.concat(records, ignore_index=True)
    pooled_subject = evaluate(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
        predictions[[f"proba_{label}" for label in label_tuple]].to_numpy(),
        labels=label_tuple,
        positive_label=positive_label,
        level="subject",
    )
    frame_true, frame_pred, frame_proba = (
        np.concatenate([part[i] for part in frame_parts]) for i in range(3)
    )
    pooled_frame = evaluate(
        frame_true,
        frame_pred,
        frame_proba,
        labels=label_tuple,
        positive_label=positive_label,
        level="frame",
    )

    result = StageResult(
        task=task,
        labels=label_tuple,
        folds=outcomes,
        predictions=predictions,
        frame_summary=summarise_folds([o.frame_metrics for o in outcomes]),
        subject_summary=summarise_folds([o.subject_metrics for o in outcomes]),
        pooled_frame=pooled_frame,
        pooled_subject=pooled_subject,
        thresholds=[float(o.threshold.threshold) for o in outcomes if o.threshold is not None],
        config={
            "model": {"name": model.name, "params": dict(model.params)},
            "aggregation": aggregation,
            "normalization": cfg.normalization.method,
            "n_splits": cfg.split.n_splits,
            "seed": cfg.seed,
            "tune_threshold": tune_threshold,
            "threshold_method": threshold_method if tune_threshold else None,
        },
    )
    _LOGGER.info(
        "stage %s: subject UAR %.4f +/- %.4f across %d fold(s)",
        task,
        result.subject_summary["uar"]["mean"],
        result.subject_summary["uar"]["std"],
        len(outcomes),
    )
    return result


def _run_fold(
    features: FeatureSet,
    cfg: Config,
    fold: Fold,
    *,
    task: str,
    labels: tuple[int, ...],
    model_cfg: ModelConfig,
    aggregation: str,
    tune_threshold: bool,
    threshold_method: str,
    fixed_threshold: float | None,
    threshold_target: float | None,
    positive_label: int | Collection[int] | None,
) -> tuple[FoldOutcome, pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Train on one fold's training side and score its evaluation side.

    Returns the fold's outcome, its subject predictions, and the true labels,
    predicted labels and posteriors of its evaluation frames.
    """
    threshold_choice: ThresholdChoice | None = None
    needs_validation = tune_threshold and threshold_method != "fixed"

    if needs_validation:
        fit_rows, val_rows = inner_split(
            features.files,
            fold.train,
            cfg,
            seed_offset=fold.index,
            context=f"fold {fold.index} of stage {task!r}",
        )
        needs_validation = val_rows.size > 0
    else:
        fit_rows, val_rows = fold.train, np.asarray([], dtype=int)

    normalizer, fit_set = fit_normalizer(cfg, features, fit_rows)
    estimator = clone(build_model(model_cfg.name, model_cfg.params, seed=cfg.seed))

    fit_X = apply_normalizer(normalizer, fit_set)
    fit_y = fit_set.frame_labels
    if np.unique(fit_y).size < 2:
        raise StageError(
            f"fold {fold.index} of stage {task!r} has a single class in its training "
            f"frames ({np.unique(fit_y).tolist()}); reduce split.n_splits"
        )
    estimator.fit(fit_X, fit_y)

    if needs_validation:
        val_set = features.select(val_rows)
        val_proba = frame_posteriors(estimator, apply_normalizer(normalizer, val_set), labels)
        val_aggregate = aggregate_frames(
            val_proba,
            val_set.group_sizes,
            mode=score_mode(aggregation),
            classes=np.asarray(labels),
        )
        threshold_choice = select_threshold(
            val_set.file_labels,
            val_aggregate.score(positive_label),
            method=threshold_method,
            target=threshold_target,
            fixed=fixed_threshold,
            positive_label=positive_label,
        )
    elif tune_threshold:
        threshold_choice = ThresholdChoice(
            threshold=float(fixed_threshold if fixed_threshold is not None else 0.5),
            sensitivity=float("nan"),
            specificity=float("nan"),
            youden=float("nan"),
            auc=float("nan"),
            method="fixed",
        )

    test_set = features.select(fold.test)
    test_proba = frame_posteriors(estimator, apply_normalizer(normalizer, test_set), labels)
    test_frame_pred = np.asarray(labels)[np.argmax(test_proba, axis=1)]

    frame_metrics = evaluate(
        test_set.frame_labels,
        test_frame_pred,
        test_proba,
        labels=labels,
        positive_label=positive_label,
        level="frame",
    )
    frame_metrics["fold"] = fold.index

    threshold_value: float | None = None
    if tune_threshold and threshold_choice is not None:
        threshold_value = threshold_choice.threshold

    if threshold_value is not None and aggregation != "ratio_threshold":
        # The threshold belongs to this rule's own score (see score_mode), so it
        # makes the decision here in place of the rule's argmax; otherwise a
        # tuned operating point would never take effect.
        subject = aggregate_frames(
            test_proba, test_set.group_sizes, mode=aggregation, classes=np.asarray(labels)
        )
        score = subject.score(positive_label)
        negative_label = next(label for label in labels if label != positive_label)
        subject_pred = np.where(score >= threshold_value, positive_label, negative_label)
    else:
        subject = aggregate_frames(
            test_proba,
            test_set.group_sizes,
            mode=aggregation,
            classes=np.asarray(labels),
            threshold=threshold_value,
            positive_label=positive_label,
        )
        subject_pred = subject.pred
        score = subject.score(positive_label) if len(labels) == 2 else subject.proba.max(axis=1)

    subject_metrics = evaluate(
        test_set.file_labels,
        subject_pred,
        subject.proba,
        labels=labels,
        positive_label=positive_label,
        level="subject",
    )
    subject_metrics["fold"] = fold.index

    frame = pd.DataFrame(
        {
            "fold": fold.index,
            "subject_id": test_set.subjects,
            "y_true": test_set.file_labels,
            "y_pred": np.asarray(subject_pred, dtype=np.int64),
            "score": np.asarray(score, dtype=np.float64),
        }
    )
    for position, label in enumerate(labels):
        frame[f"proba_{label}"] = subject.proba[:, position]

    outcome = FoldOutcome(
        fold=fold.index,
        subject_ids=np.asarray(test_set.subjects),
        subject_true=test_set.file_labels,
        subject_pred=np.asarray(subject_pred, dtype=np.int64),
        subject_proba=subject.proba,
        subject_score=np.asarray(score, dtype=np.float64),
        frame_metrics=frame_metrics,
        subject_metrics=subject_metrics,
        threshold=threshold_choice,
    )
    return outcome, frame, (test_set.frame_labels, test_frame_pred, test_proba)


def fit_stage(
    features: FeatureSet,
    cfg: Config,
    *,
    task: str,
    labels: Sequence[int],
    model: ModelConfig,
    aggregation: str = "mean_proba",
    tune_threshold: bool = False,
    threshold_method: str = "youden",
    fixed_threshold: float | None = None,
    threshold_target: float | None = None,
    positive_label: int = 1,
) -> StageModel:
    """Fit one stage for inference on new recordings.

    Parameters
    ----------
    features : FeatureSet
        Training material.
    cfg : Config
        Configuration supplying the normalisation, the seed and the inner split.
    task : str
        Name of the stage.
    labels : sequence of int
        Label set, in posterior column order.
    model : ModelConfig
        Estimator to train.
    aggregation : str, optional
        Frame-to-subject rule stored with the model, default ``"mean_proba"``.
    tune_threshold : bool, optional
        Choose a threshold on a held-out part of the training data.
    threshold_method : str, optional
        Selection rule, default ``"youden"``.
    fixed_threshold : float, optional
        The threshold itself, for ``method='fixed'``.
    threshold_target : float, optional
        Operating point for the ``target_*`` rules.
    positive_label : int, optional
        Label treated as positive, default 1.  Only a two-class stage with a
        threshold reads it; ``disorder`` and ``flat`` store it unused.

    Returns
    -------
    StageModel
        The fitted normaliser, classifier and threshold.

    Raises
    ------
    StageError
        If the training material holds fewer than two classes.

    Notes
    -----
    A tuned threshold needs files the model was not trained on, so the inner
    split of the cross-validation is used here as well: the model is fitted on
    the fitting part (usually four fifths of the files at
    ``split.val_fraction`` 0.2, fewer when a class has under five groups) and
    the threshold chosen on the rest.  Without threshold tuning every file
    is used for fitting.
    """
    label_tuple = tuple(int(label) for label in labels)
    if np.unique(features.file_labels).size < 2:
        raise StageError(f"stage {task!r} needs at least two classes to be fitted")

    rows = np.arange(features.n_files)
    threshold: float | None = None

    if tune_threshold and threshold_method != "fixed":
        fit_rows, val_rows = inner_split(features.files, rows, cfg, context=f"stage {task!r}")
        if val_rows.size == 0:
            threshold = fixed_threshold if fixed_threshold is not None else 0.5
    else:
        fit_rows, val_rows = rows, np.asarray([], dtype=int)
        threshold = fixed_threshold

    normalizer, fit_set = fit_normalizer(cfg, features, fit_rows)
    estimator = clone(build_model(model.name, model.params, seed=cfg.seed))
    estimator.fit(apply_normalizer(normalizer, fit_set), fit_set.frame_labels)

    if val_rows.size:
        val_set = features.select(val_rows)
        val_proba = frame_posteriors(estimator, apply_normalizer(normalizer, val_set), label_tuple)
        val_aggregate = aggregate_frames(
            val_proba,
            val_set.group_sizes,
            mode=score_mode(aggregation),
            classes=np.asarray(label_tuple),
        )
        threshold = select_threshold(
            val_set.file_labels,
            val_aggregate.score(positive_label),
            method=threshold_method,
            target=threshold_target,
            fixed=fixed_threshold,
            positive_label=positive_label,
        ).threshold

    return StageModel(
        task=task,
        normalizer=normalizer,
        model=estimator,
        labels=label_tuple,
        aggregation=aggregation,
        threshold=threshold,
        positive_label=positive_label,
    )
