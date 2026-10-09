"""The two-stage cascade, run end to end in one process.

Detect first, name the disorder second.  Both stages run in one procedure
because the hand-off between them is where errors are easiest to miscount: a
subject the detector wrongly clears never reaches the second stage and cannot
be recovered, and a normal subject wrongly forwarded is assigned some disorder
no matter what.  Both are part of the cascade's error, and both show up in the
four-way confusion matrix reported here.

Within one fold:

1. the detector is trained on the training files, with its threshold chosen on
   an inner validation split of them;
2. the disorder classifier is trained on the *truly* pathological training
   files: the training labels are known, so there is no reason to feed it the
   detector's mistakes;
3. on the evaluation files, the detector runs first, and only the subjects it
   forwards are scored by the disorder classifier;
4. the two decisions are merged into one label in ``0..3``.

Examples
--------
>>> from voice_pathology_detection.config import Config
>>> from voice_pathology_detection.data.synthetic import make_frame_dataset
>>> from voice_pathology_detection.features.pipeline import FeatureSet
>>> X, frame_file, files = make_frame_dataset(6, 6, 6, seed=0)
>>> cfg = Config().override(
...     {"split.n_splits": 2,
...      "detection.model.name": "gmm", "detection.model.params": {"n_components": 1},
...      "disorder.model.name": "gmm",
...      "disorder.model.params": {"n_components": 1}}
... )
>>> result = run_cascade(FeatureSet(X, frame_file, files), cfg)
>>> sorted(result.predictions.columns)[:3]
['fold', 'forwarded', 'score']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from ..config import Config
from ..data.labels import (
    ALL_LABELS,
    PATHOLOGICAL_LABELS,
    BinaryClass,
    VoiceClass,
    class_name,
    to_binary,
)
from ..data.splits import Fold, check_no_group_leakage, subject_folds
from ..eval.aggregate import aggregate_frames
from ..eval.metrics import evaluate, summarise_folds
from ..eval.threshold import ThresholdChoice, select_threshold
from ..features.pipeline import FeatureSet
from ..models.registry import build_model
from .stage import (
    StageError,
    apply_normalizer,
    fit_normalizer,
    frame_posteriors,
    inner_split,
    score_mode,
)

_LOGGER = logging.getLogger(__name__)

#: Label set of the merged decision.
CASCADE_LABELS: tuple[int, ...] = ALL_LABELS

#: Label set of each stage.
_DETECTION_LABELS: tuple[int, ...] = (int(BinaryClass.NORMAL), int(BinaryClass.PATHOLOGICAL))
_DISORDER_LABELS: tuple[int, ...] = (
    int(VoiceClass.NEOPLASM),
    int(VoiceClass.PHONOTRAUMA),
    int(VoiceClass.VOCAL_PALSY),
)


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """Cross-validated outcome of the two-stage cascade.

    Attributes
    ----------
    predictions : pandas.DataFrame
        One row per subject: ``fold``, ``subject_id``, ``y_true`` (0..3),
        ``y_pred`` (0..3), ``y_true_binary``, ``y_pred_binary``, ``score`` (the
        detector's continuous score) and ``forwarded``.
    detection_summary, disorder_summary, overall_summary : dict
        Mean and standard deviation across folds of the detection metrics, the
        disorder metrics on the forwarded subjects, and the merged four-way
        metrics.  The four-way sensitivity and specificity treat the three
        disorders together as positive, so they mean pathological against
        normal; the disorder view has no normal class and leaves them ``nan``.
    pooled_detection, pooled_disorder, pooled_overall : dict
        The same three views computed over all out-of-fold predictions at once.
    thresholds : list of float
        Detector threshold chosen in each fold.
    forwarded_per_fold : list of int
        How many subjects each fold's detector sent to the second stage.
    config : dict
        The configuration the cascade ran with.
    """

    predictions: pd.DataFrame
    detection_summary: dict[str, dict[str, float]]
    disorder_summary: dict[str, dict[str, float]]
    overall_summary: dict[str, dict[str, float]]
    pooled_detection: dict[str, Any]
    pooled_disorder: dict[str, Any]
    pooled_overall: dict[str, Any]
    thresholds: list[float] = field(default_factory=list)
    forwarded_per_fold: list[int] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary of the cascade."""
        return {
            "task": "cascade",
            "labels": list(CASCADE_LABELS),
            "label_names": {int(label): class_name(label) for label in CASCADE_LABELS},
            "detection_summary": self.detection_summary,
            "disorder_summary": self.disorder_summary,
            "overall_summary": self.overall_summary,
            "pooled_detection": self.pooled_detection,
            "pooled_disorder": self.pooled_disorder,
            "pooled_overall": self.pooled_overall,
            "thresholds": self.thresholds,
            "forwarded_per_fold": self.forwarded_per_fold,
            "config": self.config,
        }


def _train_detector(
    features: FeatureSet,
    cfg: Config,
    fold: Fold,
) -> tuple[Any, Any, ThresholdChoice]:
    """Fit the detector of one fold and choose its threshold on inner data."""
    binary_files = features.files.assign(label=to_binary(features.file_labels))
    binary = FeatureSet(features.X, features.frame_file, binary_files)

    if cfg.detection.threshold_method == "fixed":
        fit_rows, val_rows = fold.train, np.asarray([], dtype=int)
    else:
        # The detector's inner split is stratified on the binary labels it will
        # actually be trained on, not on the four-way diagnosis.
        fit_rows, val_rows = inner_split(
            binary_files,
            fold.train,
            cfg,
            seed_offset=fold.index,
            context=f"fold {fold.index} of the detection stage",
        )

    normalizer, fit_set = fit_normalizer(cfg, binary, fit_rows)
    model = clone(build_model(cfg.detection.model.name, cfg.detection.model.params, seed=cfg.seed))
    fit_y = fit_set.frame_labels
    if np.unique(fit_y).size < 2:
        raise StageError(f"fold {fold.index}: the detector's training frames hold a single class")
    model.fit(apply_normalizer(normalizer, fit_set), fit_y)

    if val_rows.size:
        val_set = binary.select(val_rows)
        val_proba = frame_posteriors(
            model, apply_normalizer(normalizer, val_set), _DETECTION_LABELS
        )
        val_aggregate = aggregate_frames(
            val_proba,
            val_set.group_sizes,
            mode=score_mode(cfg.detection.aggregation),
            classes=np.asarray(_DETECTION_LABELS),
        )
        choice = select_threshold(
            val_set.file_labels,
            val_aggregate.score(int(BinaryClass.PATHOLOGICAL)),
            method=cfg.detection.threshold_method,
            target=cfg.detection.target,
            fixed=cfg.detection.threshold,
            positive_label=int(BinaryClass.PATHOLOGICAL),
        )
    else:
        fixed = cfg.detection.threshold
        choice = ThresholdChoice(
            threshold=float(fixed if fixed is not None else 0.5),
            sensitivity=float("nan"),
            specificity=float("nan"),
            youden=float("nan"),
            auc=float("nan"),
            method="fixed",
        )
    return normalizer, model, choice


def _train_disorder(
    features: FeatureSet,
    cfg: Config,
    fold: Fold,
) -> tuple[Any, Any]:
    """Fit the disorder classifier on the truly pathological training files."""
    labels = features.files["label"].to_numpy()
    pathological_rows = np.asarray(
        [row for row in fold.train if int(labels[row]) in _DISORDER_LABELS], dtype=int
    )
    if pathological_rows.size == 0:
        raise StageError(
            f"fold {fold.index}: no pathological recording in the training side, so the "
            "disorder classifier cannot be trained"
        )
    present = np.unique(labels[pathological_rows])
    if present.size < 2:
        raise StageError(
            f"fold {fold.index}: the disorder classifier needs at least two classes, found "
            f"{[class_name(int(label)) for label in present]}; reduce split.n_splits"
        )

    normalizer, fit_set = fit_normalizer(cfg, features, pathological_rows)
    model = clone(build_model(cfg.disorder.model.name, cfg.disorder.model.params, seed=cfg.seed))
    model.fit(apply_normalizer(normalizer, fit_set), fit_set.frame_labels)
    return normalizer, model


def run_cascade(features: FeatureSet, cfg: Config | None = None) -> CascadeResult:
    """Cross-validate the full two-stage system.

    Parameters
    ----------
    features : FeatureSet
        Frames of the labelled corpus, with four-way labels.
    cfg : Config, optional
        Configuration; the defaults are used when omitted.

    Returns
    -------
    CascadeResult
        Out-of-fold predictions and the three views of the metrics.

    Raises
    ------
    StageError
        If a fold cannot train either stage, which names the reason.

    Notes
    -----
    A subject the detector clears is predicted ``Normal`` and never reaches the
    disorder classifier; a normal subject the detector forwards is given
    whichever disorder scores highest, which is always an error.  Both are
    counted in the merged four-way metrics, so those are the numbers to quote
    for the system as a whole.
    """
    cfg = Config() if cfg is None else cfg
    folds = subject_folds(
        features.files,
        n_splits=cfg.split.n_splits,
        seed=cfg.split.seed,
        group_column=cfg.split.group_by,
    )
    check_no_group_leakage(features.files, folds, group_column=cfg.split.group_by)

    records: list[pd.DataFrame] = []
    detection_metrics: list[dict[str, Any]] = []
    disorder_metrics: list[dict[str, Any]] = []
    overall_metrics: list[dict[str, Any]] = []
    thresholds: list[float] = []
    forwarded_counts: list[int] = []

    for fold in folds:
        detector_norm, detector, choice = _train_detector(features, cfg, fold)
        disorder_norm, disorder = _train_disorder(features, cfg, fold)
        thresholds.append(float(choice.threshold))

        test_set = features.select(fold.test)
        binary_truth = to_binary(test_set.file_labels)

        detection_proba = frame_posteriors(
            detector, apply_normalizer(detector_norm, test_set), _DETECTION_LABELS
        )
        detection_aggregate = aggregate_frames(
            detection_proba,
            test_set.group_sizes,
            mode=score_mode(cfg.detection.aggregation),
            classes=np.asarray(_DETECTION_LABELS),
        )
        score = detection_aggregate.score(int(BinaryClass.PATHOLOGICAL))
        forwarded = score >= choice.threshold
        forwarded_counts.append(int(forwarded.sum()))

        merged = np.full(test_set.n_files, int(VoiceClass.NORMAL), dtype=np.int64)
        disorder_pred = np.full(test_set.n_files, -1, dtype=np.int64)
        if forwarded.any():
            subset = test_set.select(np.flatnonzero(forwarded))
            subset_proba = frame_posteriors(
                disorder, apply_normalizer(disorder_norm, subset), _DISORDER_LABELS
            )
            subset_aggregate = aggregate_frames(
                subset_proba,
                subset.group_sizes,
                mode=cfg.disorder.aggregation,
                classes=np.asarray(_DISORDER_LABELS),
            )
            disorder_pred[forwarded] = subset_aggregate.pred
            merged[forwarded] = subset_aggregate.pred
        else:
            _LOGGER.warning(
                "fold %d: the detector forwarded no subject at threshold %.4f; "
                "every prediction defaults to Normal",
                fold.index,
                choice.threshold,
            )

        binary_pred = np.where(forwarded, int(BinaryClass.PATHOLOGICAL), int(BinaryClass.NORMAL))
        detection_report = evaluate(
            binary_truth,
            binary_pred,
            score,
            labels=_DETECTION_LABELS,
            positive_label=int(BinaryClass.PATHOLOGICAL),
            level="subject",
        )
        detection_report["fold"] = fold.index
        detection_metrics.append(detection_report)

        truly_pathological = forwarded & (binary_truth == int(BinaryClass.PATHOLOGICAL))
        if truly_pathological.any():
            disorder_report = evaluate(
                test_set.file_labels[truly_pathological],
                disorder_pred[truly_pathological],
                labels=_DISORDER_LABELS,
                positive_label=None,
                level="subject",
            )
        else:  # pragma: no cover - only when a fold forwards nothing correct
            disorder_report = {"level": "subject", "n": 0, "uar": float("nan")}
        disorder_report["fold"] = fold.index
        disorder_metrics.append(disorder_report)

        overall_report = evaluate(
            test_set.file_labels,
            merged,
            labels=CASCADE_LABELS,
            positive_label=PATHOLOGICAL_LABELS,
            level="subject",
        )
        overall_report["fold"] = fold.index
        overall_metrics.append(overall_report)

        records.append(
            pd.DataFrame(
                {
                    "fold": fold.index,
                    "subject_id": test_set.subjects,
                    "y_true": test_set.file_labels,
                    "y_pred": merged,
                    "y_true_binary": binary_truth,
                    "y_pred_binary": binary_pred,
                    "score": score,
                    "forwarded": forwarded,
                }
            )
        )

    predictions = pd.concat(records, ignore_index=True)
    truth = predictions["y_true"].to_numpy()
    pooled_overall = evaluate(
        truth,
        predictions["y_pred"].to_numpy(),
        labels=CASCADE_LABELS,
        positive_label=PATHOLOGICAL_LABELS,
        level="subject",
    )
    pooled_detection = evaluate(
        predictions["y_true_binary"].to_numpy(),
        predictions["y_pred_binary"].to_numpy(),
        predictions["score"].to_numpy(),
        labels=_DETECTION_LABELS,
        positive_label=int(BinaryClass.PATHOLOGICAL),
        level="subject",
    )
    correctly_forwarded = predictions["forwarded"].to_numpy() & (
        predictions["y_true_binary"].to_numpy() == int(BinaryClass.PATHOLOGICAL)
    )
    pooled_disorder = (
        evaluate(
            truth[correctly_forwarded],
            predictions["y_pred"].to_numpy()[correctly_forwarded],
            labels=_DISORDER_LABELS,
            positive_label=None,
            level="subject",
        )
        if correctly_forwarded.any()
        else {"level": "subject", "n": 0, "uar": float("nan")}
    )

    result = CascadeResult(
        predictions=predictions,
        detection_summary=summarise_folds(detection_metrics),
        disorder_summary=summarise_folds(disorder_metrics),
        overall_summary=summarise_folds(overall_metrics),
        pooled_detection=pooled_detection,
        pooled_disorder=pooled_disorder,
        pooled_overall=pooled_overall,
        thresholds=thresholds,
        forwarded_per_fold=forwarded_counts,
        config={
            "detection": {
                "model": cfg.detection.model.name,
                "params": dict(cfg.detection.model.params),
                "aggregation": cfg.detection.aggregation,
                "threshold_method": cfg.detection.threshold_method,
            },
            "disorder": {
                "model": cfg.disorder.model.name,
                "params": dict(cfg.disorder.model.params),
                "aggregation": cfg.disorder.aggregation,
            },
            "normalization": cfg.normalization.method,
            "n_splits": cfg.split.n_splits,
            "seed": cfg.seed,
        },
    )
    _LOGGER.info(
        "cascade: four-way UAR %.4f +/- %.4f, detection sensitivity %.4f, "
        "%s subject(s) forwarded per fold",
        result.overall_summary["uar"]["mean"],
        result.overall_summary["uar"]["std"],
        result.detection_summary["sensitivity"]["mean"],
        forwarded_counts,
    )
    return result
