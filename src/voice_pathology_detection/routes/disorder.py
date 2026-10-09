"""Stage two: which disorder, among the pathological recordings.

A three-way decision over Neoplasm, Phonotrauma and Vocal palsy, trained only
on recordings that carry one of those diagnoses.  No threshold is involved: the
subject-level posteriors are compared directly and the largest wins.

Examples
--------
>>> from voice_pathology_detection.config import Config
>>> from voice_pathology_detection.data.synthetic import make_frame_dataset
>>> from voice_pathology_detection.features.pipeline import FeatureSet
>>> X, frame_file, files = make_frame_dataset(5, 6, 6, seed=0)
>>> subset = pathological_subset(FeatureSet(X, frame_file, files))
>>> sorted(set(subset.file_labels.tolist()))
[1, 2, 3]
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..data.labels import DISORDER_LABELS, VoiceClass, class_name, is_pathological
from ..features.pipeline import FeatureSet
from .stage import StageError, StageModel, StageResult, fit_stage, run_stage

_LOGGER = logging.getLogger(__name__)


def pathological_subset(features: FeatureSet) -> FeatureSet:
    """Return the recordings carrying a pathological diagnosis.

    Parameters
    ----------
    features : FeatureSet
        Feature set with four-way labels.

    Returns
    -------
    FeatureSet
        The subset whose labels are 1, 2 or 3.

    Raises
    ------
    StageError
        If no pathological recording is present, or fewer than two of the three
        disorders are, since a classifier cannot be trained on one class.
    """
    mask = is_pathological(features.file_labels)
    if not mask.any():
        raise StageError(
            "the corpus holds no pathological recording; the disorder stage needs "
            f"labels among {[class_name(label) for label in DISORDER_LABELS]}"
        )
    subset = features.select(mask)
    present = np.unique(subset.file_labels)
    if present.size < 2:
        raise StageError(
            f"the disorder stage needs at least two classes, found only "
            f"{[class_name(int(label)) for label in present]}"
        )
    _LOGGER.info(
        "disorder stage: %d recording(s) across %d class(es)", subset.n_files, present.size
    )
    return subset


def run_disorder(features: FeatureSet, cfg: Config | None = None) -> StageResult:
    """Cross-validate the three-way disorder classifier.

    Parameters
    ----------
    features : FeatureSet
        Frames of the labelled corpus; normal recordings are dropped here.
    cfg : Config, optional
        Configuration; the defaults are used when omitted.

    Returns
    -------
    StageResult
        Per-fold and pooled metrics with the out-of-fold subject predictions.
        Sensitivity, specificity and the predictive values are ``nan``: there
        is no normal class here to screen against.

    Raises
    ------
    StageError
        If fewer than two disorders are present, or a fold cannot be trained.
    """
    cfg = Config() if cfg is None else cfg
    subset = pathological_subset(features)
    return run_stage(
        subset,
        cfg,
        task="disorder",
        labels=DISORDER_LABELS,
        model=cfg.disorder.model,
        aggregation=cfg.disorder.aggregation,
        tune_threshold=False,
        positive_label=None,
    )


def fit_disorder(features: FeatureSet, cfg: Config | None = None) -> StageModel:
    """Fit the disorder classifier on all pathological recordings.

    Parameters
    ----------
    features : FeatureSet
        Training material, four-way labelled.
    cfg : Config, optional
        Configuration; the defaults are used when omitted.

    Returns
    -------
    StageModel
        The fitted normaliser and classifier.
    """
    cfg = Config() if cfg is None else cfg
    return fit_stage(
        pathological_subset(features),
        cfg,
        task="disorder",
        labels=DISORDER_LABELS,
        model=cfg.disorder.model,
        aggregation=cfg.disorder.aggregation,
        tune_threshold=False,
        positive_label=int(VoiceClass.NEOPLASM),
    )
