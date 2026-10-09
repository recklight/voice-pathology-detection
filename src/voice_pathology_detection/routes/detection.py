"""Stage one: normal versus pathological.

A screening decision, so the operating point matters as much as the model.  The
threshold is chosen on an inner validation split of each training fold, never
on the fold being evaluated, and the value chosen in every fold is written to
the results so the spread of the operating point can be read off directly.

Examples
--------
>>> from voice_pathology_detection.config import Config
>>> from voice_pathology_detection.data.synthetic import make_frame_dataset
>>> from voice_pathology_detection.features.pipeline import FeatureSet
>>> X, frame_file, files = make_frame_dataset(5, 6, 6, seed=0)
>>> cfg = Config().override(
...     {"split.n_splits": 2, "detection.model.name": "gmm",
...      "detection.model.params": {"n_components": 1}}
... )
>>> result = run_detection(FeatureSet(X, frame_file, files), cfg)
>>> result.labels
(0, 1)
"""

from __future__ import annotations

import logging

from ..config import Config
from ..data.labels import BinaryClass, to_binary
from ..features.pipeline import FeatureSet
from .stage import StageModel, StageResult, fit_stage, run_stage

_LOGGER = logging.getLogger(__name__)

#: Label set of the detection stage.
DETECTION_LABELS: tuple[int, ...] = (int(BinaryClass.NORMAL), int(BinaryClass.PATHOLOGICAL))


def binary_view(features: FeatureSet) -> FeatureSet:
    """Return the same frames with the four-way labels collapsed to two.

    Parameters
    ----------
    features : FeatureSet
        Feature set whose files carry four-way labels.

    Returns
    -------
    FeatureSet
        A set sharing the frame matrix, with ``label`` replaced by its binary
        projection.

    Raises
    ------
    ValueError
        If a label lies outside ``0..3``.

    Examples
    --------
    >>> from voice_pathology_detection.data.synthetic import make_frame_dataset
    >>> X, frame_file, files = make_frame_dataset(2, 4, 3, seed=0)
    >>> sorted(set(binary_view(FeatureSet(X, frame_file, files)).file_labels.tolist()))
    [0, 1]
    """
    files = features.files.assign(label=to_binary(features.file_labels))
    return FeatureSet(X=features.X, frame_file=features.frame_file, files=files)


def run_detection(features: FeatureSet, cfg: Config | None = None) -> StageResult:
    """Cross-validate the normal/pathological detector.

    Parameters
    ----------
    features : FeatureSet
        Frames of the labelled corpus, with four-way or binary labels.
    cfg : Config, optional
        Configuration; the defaults are used when omitted.

    Returns
    -------
    StageResult
        Per-fold and pooled metrics with the out-of-fold subject predictions.

    Raises
    ------
    StageError
        If the corpus holds a single class, or a fold cannot be trained.

    Notes
    -----
    The threshold is always tuned unless ``detection.threshold_method`` is
    ``"fixed"``, in which case ``detection.threshold`` is used as given.
    """
    cfg = Config() if cfg is None else cfg
    binary = binary_view(features)
    _LOGGER.info(
        "detection stage: %d recording(s), %d pathological",
        binary.n_files,
        int((binary.file_labels == int(BinaryClass.PATHOLOGICAL)).sum()),
    )
    return run_stage(
        binary,
        cfg,
        task="detection",
        labels=DETECTION_LABELS,
        model=cfg.detection.model,
        aggregation=cfg.detection.aggregation,
        tune_threshold=True,
        threshold_method=cfg.detection.threshold_method,
        fixed_threshold=cfg.detection.threshold,
        threshold_target=cfg.detection.target,
        positive_label=int(BinaryClass.PATHOLOGICAL),
    )


def fit_detection(features: FeatureSet, cfg: Config | None = None) -> StageModel:
    """Fit the detector for inference, with its threshold chosen on held-out files.

    Parameters
    ----------
    features : FeatureSet
        Training material.
    cfg : Config, optional
        Configuration; the defaults are used when omitted.

    Returns
    -------
    StageModel
        The fitted normaliser, classifier and threshold.  Unless the threshold
        is fixed, the model is fitted on the inner fitting part of the files and
        the threshold chosen on the rest, as in :func:`fit_stage`.
    """
    cfg = Config() if cfg is None else cfg
    return fit_stage(
        binary_view(features),
        cfg,
        task="detection",
        labels=DETECTION_LABELS,
        model=cfg.detection.model,
        aggregation=cfg.detection.aggregation,
        tune_threshold=True,
        threshold_method=cfg.detection.threshold_method,
        fixed_threshold=cfg.detection.threshold,
        threshold_target=cfg.detection.target,
        positive_label=int(BinaryClass.PATHOLOGICAL),
    )
