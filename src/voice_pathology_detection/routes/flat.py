"""Single-stage four-way classification, the comparison route.

Instead of detecting disease and then naming it, one classifier decides among
Normal, Neoplasm, Phonotrauma and Vocal palsy directly.  It goes through the
same evaluation code as the cascade, and I keep it because without it there is
no way to tell whether splitting the problem in two helps or only hides the
errors of the first stage.

Examples
--------
>>> from voice_pathology_detection.config import Config
>>> from voice_pathology_detection.data.synthetic import make_frame_dataset
>>> from voice_pathology_detection.features.pipeline import FeatureSet
>>> X, frame_file, files = make_frame_dataset(5, 6, 6, seed=0)
>>> cfg = Config().override(
...     {"split.n_splits": 2, "flat.model.name": "gmm",
...      "flat.model.params": {"n_components": 1}}
... )
>>> run_flat(FeatureSet(X, frame_file, files), cfg).labels
(0, 1, 2, 3)
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..data.labels import ALL_LABELS, PATHOLOGICAL_LABELS, VoiceClass, class_name
from ..features.pipeline import FeatureSet
from .stage import StageError, StageModel, StageResult, fit_stage, run_stage

_LOGGER = logging.getLogger(__name__)

#: Label set of the single-stage route.
FLAT_LABELS: tuple[int, ...] = ALL_LABELS


def _check_classes(features: FeatureSet) -> None:
    """Warn about missing classes and refuse a degenerate label set."""
    present = np.unique(features.file_labels)
    if present.size < 2:
        raise StageError(
            f"the four-way stage needs at least two classes, found "
            f"{[class_name(int(label)) for label in present]}"
        )
    missing = [label for label in FLAT_LABELS if label not in present]
    if missing:
        _LOGGER.warning(
            "class(es) %s are absent from the corpus; their recall is listed as 0 and left "
            "out of the UAR",
            [class_name(label) for label in missing],
        )


def run_flat(features: FeatureSet, cfg: Config | None = None) -> StageResult:
    """Cross-validate the single-stage four-way classifier.

    Parameters
    ----------
    features : FeatureSet
        Frames of the labelled corpus, with four-way labels.
    cfg : Config, optional
        Configuration; the ``flat`` section supplies the model.

    Returns
    -------
    StageResult
        Per-fold and pooled metrics with the out-of-fold subject predictions.
        Sensitivity and specificity count the three disorders together as
        positive, so they compare with those of the detection route.

    Raises
    ------
    StageError
        If fewer than two classes are present, or a fold cannot be trained.
    """
    cfg = Config() if cfg is None else cfg
    _check_classes(features)
    return run_stage(
        features,
        cfg,
        task="flat",
        labels=FLAT_LABELS,
        model=cfg.flat.model,
        aggregation=cfg.flat.aggregation,
        tune_threshold=False,
        positive_label=PATHOLOGICAL_LABELS,
    )


def fit_flat(features: FeatureSet, cfg: Config | None = None) -> StageModel:
    """Fit the single-stage classifier on all available data.

    Parameters
    ----------
    features : FeatureSet
        Training material.
    cfg : Config, optional
        Configuration; the defaults are used when omitted.

    Returns
    -------
    StageModel
        The fitted normaliser and classifier.
    """
    cfg = Config() if cfg is None else cfg
    _check_classes(features)
    return fit_stage(
        features,
        cfg,
        task="flat",
        labels=FLAT_LABELS,
        model=cfg.flat.model,
        aggregation=cfg.flat.aggregation,
        tune_threshold=False,
        positive_label=int(VoiceClass.NORMAL),
    )
