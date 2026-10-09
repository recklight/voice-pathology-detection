"""The prediction table that ``predict`` writes for unlabelled recordings.

Two columns, one row per recording: the file name and the predicted class in
``0..3``.  The file name is written as text, because identifiers such as
``001`` lose their leading zeros the moment a spreadsheet decides they are
numbers, and ``1`` cannot be joined back to the recording ``001``.

Examples
--------
>>> import tempfile, pathlib
>>> with tempfile.TemporaryDirectory() as tmp:
...     out = write_prediction_table(pathlib.Path(tmp) / "pred.csv", ["001", "002"], [0, 2])
...     out.read_text(encoding="utf-8").splitlines()[0]
'File name,Prediction'
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .data.labels import ALL_LABELS, class_name

_LOGGER = logging.getLogger(__name__)

#: Column headers of the prediction table.
FILE_COLUMN = "File name"
PREDICTION_COLUMN = "Prediction"


def build_prediction_table(
    file_names: Sequence[str],
    predictions: object,
    *,
    strip_suffix: bool = True,
) -> pd.DataFrame:
    """Assemble the prediction table.

    Parameters
    ----------
    file_names : sequence of str
        Identifier of every recording, in any order.
    predictions : array_like
        Predicted class of every recording, in ``0..3``.
    strip_suffix : bool, optional
        Drop a file extension such as ``.wav`` from the identifiers, default
        ``True``.

    Returns
    -------
    pandas.DataFrame
        Two columns, with the file names stored as strings.

    Raises
    ------
    ValueError
        If the two inputs differ in length, are empty, a prediction lies
        outside the valid range, or an identifier repeats.

    Examples
    --------
    >>> build_prediction_table(["001.wav"], [2])["Prediction"].tolist()
    [2]
    """
    names = [str(name) for name in file_names]
    values = np.asarray(predictions).ravel()
    if len(names) != values.size:
        raise ValueError(
            f"{len(names)} file name(s) but {values.size} prediction(s); every "
            "recording needs exactly one prediction"
        )
    if not names:
        raise ValueError("cannot build an empty prediction table")

    codes = np.asarray(values, dtype=np.int64)
    unknown = np.setdiff1d(codes, np.asarray(ALL_LABELS, dtype=np.int64))
    if unknown.size:
        raise ValueError(
            f"prediction(s) {unknown.tolist()} are outside the valid range {list(ALL_LABELS)}"
        )

    if strip_suffix:
        names = [Path(name).stem for name in names]
    duplicates = pd.Index(names)[pd.Index(names).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"file name(s) {duplicates[:5]} appear more than once")

    return pd.DataFrame({FILE_COLUMN: names, PREDICTION_COLUMN: codes})


def write_prediction_table(
    path: str | Path,
    file_names: Sequence[str],
    predictions: object,
    *,
    strip_suffix: bool = True,
) -> Path:
    """Write the prediction table to ``.xlsx`` or ``.csv``.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination; the format follows the suffix, ``.xlsx`` by default for an
        unknown one.
    file_names : sequence of str
        Identifier of every recording.
    predictions : array_like
        Predicted class of every recording.
    strip_suffix : bool, optional
        Drop file extensions from the identifiers, default ``True``.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Raises
    ------
    ValueError
        If the inputs are inconsistent; see :func:`build_prediction_table`.
    ImportError
        If an ``.xlsx`` file is requested without ``openpyxl`` installed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = build_prediction_table(file_names, predictions, strip_suffix=strip_suffix)

    if path.suffix.lower() == ".csv":
        table.to_csv(path, index=False)
    else:
        try:
            table.to_excel(path, index=False)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"writing {path.suffix} needs openpyxl; install it or choose a .csv path"
            ) from exc

    distribution = {
        class_name(int(code)): int(count)
        for code, count in table[PREDICTION_COLUMN].value_counts().items()
    }
    _LOGGER.info("wrote %d prediction(s) to %s: %s", len(table), path, distribution)
    return path


def read_prediction_table(path: str | Path) -> pd.DataFrame:
    """Read a prediction table back.

    Parameters
    ----------
    path : str or pathlib.Path
        File to read, ``.csv`` or ``.xlsx``.

    Returns
    -------
    pandas.DataFrame
        Two columns, with the file names as strings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the expected columns are missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such prediction table: {path}")
    table = (
        pd.read_csv(path, dtype={FILE_COLUMN: str})
        if path.suffix.lower() == ".csv"
        else pd.read_excel(path, dtype={FILE_COLUMN: str})
    )
    missing = {FILE_COLUMN, PREDICTION_COLUMN} - set(table.columns)
    if missing:
        raise ValueError(
            f"{path} lacks the column(s) {sorted(missing)}; found {list(table.columns)}"
        )
    table[FILE_COLUMN] = table[FILE_COLUMN].astype(str)
    return table
