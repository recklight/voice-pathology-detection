"""Turning frame-level scores into one decision per subject.

The classifier sees 30 ms frames, but the clinical question is about a person.
The rule that gets from one to the other changes the answer, so it is a
setting: ``aggregation`` in each stage's section of the configuration.

``mean_proba``
    Average the posterior probabilities over the frames of a recording and take
    the largest.  This is the default: averaging probabilities keeps the
    confidence of every frame, so a recording that is uncertain throughout is
    treated differently from one that is split between two confident halves.
``majority_vote``
    Count hard frame decisions.  A badly calibrated model hurts it less, but
    the confidence of each frame is thrown away.
``ratio_threshold``
    For the two-class task only: call a subject pathological when the fraction
    of frames labelled pathological reaches a threshold.  This exposes the
    operating point directly, which is what :mod:`voice_pathology_detection.eval.threshold`
    tunes.

Examples
--------
>>> import numpy as np
>>> proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])
>>> result = aggregate_frames(proba, [2, 1])
>>> result.pred.tolist()
[0, 1]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_LOGGER = logging.getLogger(__name__)

#: Aggregation rules implemented here.
AGGREGATION_MODES: tuple[str, ...] = ("mean_proba", "majority_vote", "ratio_threshold")


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """Subject-level outcome of an aggregation.

    Attributes
    ----------
    proba : numpy.ndarray
        Mean posterior probability per subject, shape ``(n_groups, n_classes)``.
    ratio : numpy.ndarray
        Fraction of frames whose hard decision fell on each class, same shape.
    pred : numpy.ndarray
        Predicted label per subject, drawn from :attr:`classes`.
    classes : numpy.ndarray
        Label attached to every column of :attr:`proba`.
    mode : str
        Rule that produced :attr:`pred`.
    threshold : float or None
        Threshold used by ``ratio_threshold``, echoed for the record.
    """

    proba: np.ndarray
    ratio: np.ndarray
    pred: np.ndarray
    classes: np.ndarray
    mode: str
    threshold: float | None = None

    @property
    def n_groups(self) -> int:
        """Number of subjects."""
        return int(self.pred.size)

    def score(self, positive_label: int = 1) -> np.ndarray:
        """Continuous score of the positive class, for ROC-style analysis.

        Parameters
        ----------
        positive_label : int, optional
            Label treated as positive, default 1.

        Returns
        -------
        numpy.ndarray
            The mean posterior of that class under ``mean_proba``, and the
            frame ratio under the two counting rules; in both cases the
            quantity a threshold is compared with.

        Raises
        ------
        ValueError
            If ``positive_label`` is not one of :attr:`classes`.
        """
        column = np.flatnonzero(self.classes == positive_label)
        if column.size == 0:
            raise ValueError(
                f"positive_label={positive_label} is not among the classes {self.classes.tolist()}"
            )
        source = self.proba if self.mode == "mean_proba" else self.ratio
        return source[:, int(column[0])]


def frame_offsets(group_sizes: object) -> np.ndarray:
    """Return the start offset of every group, for :func:`numpy.add.reduceat`.

    Parameters
    ----------
    group_sizes : array_like
        Frame count of every recording, all strictly positive.

    Returns
    -------
    numpy.ndarray
        Offsets of length ``n_groups``, beginning at zero.

    Raises
    ------
    ValueError
        If the sizes are empty or any of them is not positive.  An empty group
        would make ``reduceat`` silently reuse the neighbouring row, so it is
        rejected rather than handled.

    Examples
    --------
    >>> frame_offsets([2, 3, 1]).tolist()
    [0, 2, 5]
    """
    sizes = np.asarray(group_sizes, dtype=np.int64).ravel()
    if sizes.size == 0:
        raise ValueError("group_sizes must not be empty")
    if np.any(sizes <= 0):
        bad = np.flatnonzero(sizes <= 0)[:5].tolist()
        raise ValueError(
            f"every recording must contribute at least one frame, but group(s) {bad} are empty"
        )
    return np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)


def split_by_group(values: object, group_sizes: object) -> list[np.ndarray]:
    """Split a per-frame array into one array per recording.

    Parameters
    ----------
    values : array_like
        Array whose first axis runs over frames.
    group_sizes : array_like
        Frame count of every recording.

    Returns
    -------
    list of numpy.ndarray
        One entry per recording, in file order.

    Examples
    --------
    >>> [part.tolist() for part in split_by_group([1, 2, 3, 4], [1, 3])]
    [[1], [2, 3, 4]]
    """
    array = np.asarray(values)
    sizes = np.asarray(group_sizes, dtype=np.int64).ravel()
    total = int(sizes.sum())
    if array.shape[0] != total:
        raise ValueError(f"group_sizes cover {total} frame(s) but the array has {array.shape[0]}")
    return list(np.split(array, np.cumsum(sizes)[:-1]))


def aggregate_frames(
    proba: object,
    group_sizes: object,
    mode: str = "mean_proba",
    *,
    classes: object = None,
    threshold: float | None = None,
    positive_label: int = 1,
) -> AggregateResult:
    """Reduce frame posteriors to one decision per subject.

    Parameters
    ----------
    proba : array_like
        Frame posteriors of shape ``(n_frames, n_classes)``.  Rows need not sum
        to one, but they must not be negative.
    group_sizes : array_like
        Frame count of every recording; must sum to ``n_frames``.
    mode : str, optional
        One of :data:`AGGREGATION_MODES`, default ``"mean_proba"``.
    classes : array_like, optional
        Label of every column, default ``0 .. n_classes - 1``.
    threshold : float, optional
        Required by ``ratio_threshold``: the fraction of positive frames at or
        above which a subject is called positive.  A value above 1 calls no
        subject positive.  It is accepted because a threshold search that
        settles on flagging nobody returns a value just above the largest
        ratio it saw, and that is above 1 when some subject had every frame
        voted pathological.
    positive_label : int, optional
        Label treated as positive by ``ratio_threshold``, default 1.

    Returns
    -------
    AggregateResult
        Subject-level probabilities, frame ratios and decisions.

    Raises
    ------
    ValueError
        If the shapes disagree, the mode is unknown, a group is empty, or
        ``ratio_threshold`` is requested without a threshold, with a negative
        one, or on more than two classes.

    Examples
    --------
    >>> import numpy as np
    >>> proba = np.array([[0.4, 0.6], [0.45, 0.55], [0.9, 0.1], [0.8, 0.2]])
    >>> aggregate_frames(proba, [2, 2], mode="ratio_threshold", threshold=0.5).pred.tolist()
    [1, 0]
    """
    scores = np.asarray(proba, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"proba must be 2-D (n_frames, n_classes), got shape {scores.shape}")
    if scores.shape[1] < 2:
        raise ValueError("proba must have at least two columns")
    if np.any(scores < 0.0):
        raise ValueError("proba must not contain negative values")
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"unknown aggregation mode {mode!r}; choose one of {AGGREGATION_MODES}")

    sizes = np.asarray(group_sizes, dtype=np.int64).ravel()
    if int(sizes.sum()) != scores.shape[0]:
        raise ValueError(
            f"group_sizes cover {int(sizes.sum())} frame(s) but proba has {scores.shape[0]}"
        )
    offsets = frame_offsets(sizes)

    labels = (
        np.arange(scores.shape[1], dtype=np.int64)
        if classes is None
        else np.asarray(classes).ravel()
    )
    if labels.size != scores.shape[1]:
        raise ValueError(
            f"classes has {labels.size} entries but proba has {scores.shape[1]} column(s)"
        )

    mean_proba = np.add.reduceat(scores, offsets, axis=0) / sizes[:, None]

    # Fraction of hard frame decisions falling on each class.
    hard = np.argmax(scores, axis=1)
    one_hot = np.zeros_like(scores)
    one_hot[np.arange(hard.size), hard] = 1.0
    ratio = np.add.reduceat(one_hot, offsets, axis=0) / sizes[:, None]

    match mode:
        case "mean_proba":
            pred = labels[np.argmax(mean_proba, axis=1)]
        case "majority_vote":
            pred = labels[np.argmax(ratio, axis=1)]
        case "ratio_threshold":
            if threshold is None:
                raise ValueError("mode='ratio_threshold' requires a threshold")
            if not threshold >= 0.0:
                raise ValueError(f"threshold must not be negative, got {threshold}")
            if labels.size != 2:
                raise ValueError(
                    f"mode='ratio_threshold' is only defined for two classes, got {labels.size}"
                )
            positive_column = np.flatnonzero(labels == positive_label)
            if positive_column.size == 0:
                raise ValueError(
                    f"positive_label={positive_label} is not among the classes {labels.tolist()}"
                )
            negative_label = labels[labels != positive_label][0]
            is_positive = ratio[:, int(positive_column[0])] >= threshold
            pred = np.where(is_positive, positive_label, negative_label)
        case _:  # pragma: no cover - guarded above
            raise ValueError(f"unknown aggregation mode: {mode!r}")

    _LOGGER.debug(
        "aggregated %d frame(s) into %d subject decision(s) with mode=%s",
        scores.shape[0],
        sizes.size,
        mode,
    )
    return AggregateResult(
        proba=mean_proba,
        ratio=ratio,
        pred=np.asarray(pred, dtype=np.int64),
        classes=labels.astype(np.int64),
        mode=mode,
        threshold=threshold,
    )
