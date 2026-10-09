"""Subject-grouped, stratified cross-validation splits.

Frames of one recording are strongly correlated.  A split drawn over frames
puts the same speaker on both sides of the boundary and inflates every score
computed after it.  The splits here are drawn over *files* instead, grouped by
subject and stratified by diagnosis, and the routes run
:func:`check_no_group_leakage` on the folds before training on them.

Examples
--------
>>> import pandas as pd
>>> frame = pd.DataFrame(
...     {
...         "subject_id": [f"s{i}" for i in range(8)],
...         "label": [0, 0, 0, 0, 1, 1, 1, 1],
...     }
... )
>>> folds = subject_folds(frame, n_splits=2, seed=0)
>>> len(folds)
2
>>> check_no_group_leakage(frame, folds)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_LOGGER = logging.getLogger(__name__)


class SplitError(ValueError):
    """Raised when a valid grouped split cannot be produced."""


@dataclass(frozen=True, slots=True)
class Fold:
    """One outer cross-validation fold, expressed as row positions.

    Attributes
    ----------
    index : int
        Zero-based fold number.
    train, test : numpy.ndarray
        Integer positions into the file index.
    """

    index: int
    train: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.test)
        if overlap.size:
            raise SplitError(
                f"fold {self.index} lists {overlap.size} row(s) in both the training "
                "and the evaluation side"
            )

    @property
    def sizes(self) -> tuple[int, int]:
        """Number of training and evaluation files."""
        return int(self.train.size), int(self.test.size)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view of the fold."""
        return {
            "fold": self.index,
            "n_train": int(self.train.size),
            "n_test": int(self.test.size),
            "train": self.train.tolist(),
            "test": self.test.tolist(),
        }


def _columns(
    frame: pd.DataFrame, label_column: str, group_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the label and group vectors, with a clear error when missing."""
    missing = {label_column, group_column} - set(frame.columns)
    if missing:
        raise SplitError(
            f"the index lacks the column(s) {sorted(missing)}; available columns are "
            f"{sorted(frame.columns)}"
        )
    labels = frame[label_column].to_numpy()
    groups = frame[group_column].to_numpy()
    return labels, groups


def _validate_feasible(labels: np.ndarray, groups: np.ndarray, n_splits: int) -> None:
    """Fail early, and with a useful message, when a split cannot exist."""
    if labels.size == 0:
        raise SplitError("cannot split an empty index")
    if n_splits < 2:
        raise SplitError(f"n_splits must be at least 2, got {n_splits}")

    unique_groups = np.unique(groups)
    if unique_groups.size < n_splits:
        raise SplitError(
            f"{unique_groups.size} distinct group(s) cannot fill {n_splits} folds; "
            "reduce split.n_splits or add more subjects"
        )
    values, counts = np.unique(labels, return_counts=True)
    if values.size < 2:
        raise SplitError(f"stratification needs at least two classes, found only {values.tolist()}")
    too_small = {int(v): int(c) for v, c in zip(values, counts, strict=True) if c < n_splits}
    if too_small:
        raise SplitError(
            f"class(es) {too_small} have fewer members than the {n_splits} requested folds; "
            "reduce split.n_splits"
        )


def subject_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 0,
    *,
    label_column: str = "label",
    group_column: str = "subject_id",
) -> list[Fold]:
    """Produce stratified folds that never split a subject.

    Parameters
    ----------
    frame : pandas.DataFrame
        File index with a label and a group column.
    n_splits : int, optional
        Number of folds, default 5.
    seed : int, optional
        Seed of the shuffling, default 0, which makes the split reproducible.
    label_column : str, optional
        Column used for stratification, default ``"label"``.
    group_column : str, optional
        Column that must stay inside one fold, default ``"subject_id"``.

    Returns
    -------
    list of Fold
        One entry per fold, with row positions into ``frame``.

    Raises
    ------
    SplitError
        If the index is empty, a required column is missing, or there are too
        few groups or class members for the requested number of folds.

    Notes
    -----
    Uses :class:`sklearn.model_selection.StratifiedGroupKFold`, which balances
    the class proportions across folds while keeping every group intact.
    """
    labels, groups = _columns(frame, label_column, group_column)
    _validate_feasible(labels, groups, n_splits)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        Fold(index=i, train=np.asarray(train), test=np.asarray(test))
        for i, (train, test) in enumerate(splitter.split(np.zeros(len(frame)), labels, groups))
    ]
    _LOGGER.info(
        "built %d subject-grouped fold(s) over %d files and %d subjects (seed=%d)",
        len(folds),
        len(frame),
        np.unique(groups).size,
        seed,
    )
    return folds


def train_val_split(
    frame: pd.DataFrame,
    rows: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 0,
    *,
    label_column: str = "label",
    group_column: str = "subject_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation set out of the training side of a fold.

    Model selection, and the decision threshold above all, has to be made on
    data the final evaluation never sees.  The training rows of a fold are
    therefore split once more, still grouped and still stratified.

    Parameters
    ----------
    frame : pandas.DataFrame
        The full file index.
    rows : numpy.ndarray
        Row positions forming the training side of a fold.
    val_fraction : float, optional
        Share held out, default 0.2, at most 0.5.  The validation set is the
        first fold of a stratified, grouped split into
        ``k = round(1 / val_fraction)`` parts, so about ``1 / k`` of the files
        are held out: a fifth for 0.2, a third for 0.3, a half for 0.4 or 0.5.
        When a class has fewer than ``k`` groups, ``k`` drops to that count.
    seed : int, optional
        Seed of the shuffling, default 0.
    label_column, group_column : str, optional
        Columns used for stratification and grouping.

    Returns
    -------
    fit_rows, val_rows : numpy.ndarray
        Row positions into ``frame``; the two arrays are disjoint and their
        union is ``rows``.

    Raises
    ------
    SplitError
        If the requested fraction cannot be honoured with the available groups,
        for instance when a class has a single group (one speaker) on the
        training side, or the held-out part would miss a class.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> frame = pd.DataFrame(
    ...     {"subject_id": [f"s{i}" for i in range(10)], "label": [0] * 5 + [1] * 5}
    ... )
    >>> fit, val = train_val_split(frame, np.arange(10), val_fraction=0.4, seed=0)
    >>> sorted(np.concatenate([fit, val]).tolist()) == list(range(10))
    True
    """
    if not 0.0 < val_fraction <= 0.5:
        raise SplitError(f"val_fraction must lie in (0, 0.5], got {val_fraction}")
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        raise SplitError("cannot split an empty set of training rows")

    subset = frame.iloc[rows]
    inner_splits = max(2, round(1.0 / val_fraction))
    labels, groups = _columns(subset, label_column, group_column)
    unique_groups = np.unique(groups).size
    values = np.unique(labels)
    # Count groups, not files: a speaker with several recordings is one unit.
    groups_per_class = min(np.unique(groups[labels == value]).size for value in values)
    inner_splits = min(inner_splits, unique_groups, groups_per_class)
    if inner_splits < 2 or values.size < 2:
        raise SplitError(
            f"cannot hold out a stratified validation set from {rows.size} file(s) "
            f"covering {unique_groups} subject(s) and classes {values.tolist()}; "
            "lower split.val_fraction or provide more data"
        )

    splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    fit_local, val_local = next(splitter.split(np.zeros(len(subset)), labels, groups))
    if np.unique(labels[val_local]).size < values.size:
        raise SplitError(
            f"the validation set held out from {unique_groups} subject(s) misses a class; "
            "lower split.val_fraction or provide more data"
        )
    _LOGGER.debug(
        "inner split: %d fitting file(s), %d validation file(s)", fit_local.size, val_local.size
    )
    return rows[fit_local], rows[val_local]


def check_no_group_leakage(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    group_column: str = "subject_id",
) -> None:
    """Assert that no group appears on both sides of any fold.

    Parameters
    ----------
    frame : pandas.DataFrame
        The file index the folds index into.
    folds : sequence of Fold
        Folds to verify.
    group_column : str, optional
        Column holding the group identifiers.

    Raises
    ------
    SplitError
        If a group straddles a fold boundary, naming the offending group.
    """
    groups = frame[group_column].to_numpy()
    for fold in folds:
        shared = np.intersect1d(groups[fold.train], groups[fold.test])
        if shared.size:
            raise SplitError(
                f"fold {fold.index} leaks {shared.size} subject(s) across the split, "
                f"for example {shared[:3].tolist()}"
            )
    _LOGGER.debug("verified %d fold(s) for subject leakage", len(folds))
