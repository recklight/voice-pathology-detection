"""The file index that every later stage reads.

The index is a :class:`pandas.DataFrame` with one row per recording.  It is the
only place that knows where the audio lives, which subject it belongs to and
what its diagnosis is.  Features, splits, training and reporting all read it
instead of working out paths of their own.

The corpus can be laid out in two ways:

* one directory per diagnosis, with names such as ``Normal`` or
  ``Pathological/Vocal palsy``;
* a flat directory plus a demographics table that gives the diagnosis of every
  file name.  When a table is given, folder names are not used for labels.

:func:`build_index` then checks the per-class file counts of the training
split against the expected layout.

Examples
--------
>>> import pandas as pd
>>> frame = pd.DataFrame(
...     {"path": ["a.wav", "b.wav"], "subject_id": ["a", "b"], "label": [0, 2]}
... )
>>> attach_binary_labels(frame)["label_binary"].tolist()
[0, 1]
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from ..config import Config
from .labels import (
    CLASS_NAMES,
    FOLDER_ALIASES,
    check_class_counts,
    class_name,
    parse_label,
    to_binary,
)

_LOGGER = logging.getLogger(__name__)

#: Columns every index is guaranteed to carry.
INDEX_COLUMNS: tuple[str, ...] = (
    "path",
    "file_name",
    "subject_id",
    "label",
    "label_name",
    "label_binary",
    "split",
)

#: Audio extensions searched for when scanning a directory.
AUDIO_SUFFIXES: tuple[str, ...] = (".wav",)

#: Column aliases accepted in a demographics table, lower-cased.
_DEMOGRAPHIC_ALIASES: dict[str, str] = {
    "file name": "file_name",
    "filename": "file_name",
    "file": "file_name",
    "diagnosis": "label",
    "diagnosis(0-3)": "label",
    "diagnosis (0-3)": "label",
    "label": "label",
    "class": "label",
    "gender": "gender",
    "gender(1=male, 2=female)": "gender",
    "sex": "gender",
    "age": "age",
    # ``subject_id`` in the index names one recording; a person with several
    # recordings is given by this column instead (see split.group_by).
    "subject": "speaker_id",
    "subject id": "speaker_id",
    "subject_id": "speaker_id",
    "speaker": "speaker_id",
    "speaker id": "speaker_id",
    "speaker_id": "speaker_id",
}


class DatasetError(RuntimeError):
    """Raised when a corpus cannot be indexed."""


def _label_from_parts(parts: Sequence[str]) -> int | None:
    """Return the label implied by the deepest matching directory name."""
    for part in reversed(parts):
        key = part.strip().lower()
        if key in FOLDER_ALIASES:
            return FOLDER_ALIASES[key]
    return None


def _iter_audio(root: Path, suffixes: Iterable[str]) -> list[Path]:
    """Return every audio file under ``root``, sorted so the order never depends on the OS."""
    wanted = {suffix.lower() for suffix in suffixes}
    files = [
        path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in wanted
    ]
    return files


def _subject_id(path: Path, root: Path) -> str:
    """Derive a stable subject identifier from the location of a recording.

    One recording per subject is the usual layout of this corpus, and file
    names repeat across diagnosis folders, so the relative path without its
    suffix is used rather than the bare stem.
    """
    relative = path.relative_to(root).with_suffix("")
    return relative.as_posix()


def scan_class_directories(
    root: str | Path,
    *,
    split: str = "train",
    suffixes: Iterable[str] = AUDIO_SUFFIXES,
    require_labels: bool = True,
) -> pd.DataFrame:
    """Index a corpus laid out as one directory per diagnosis.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory to walk recursively.
    split : str, optional
        Value written into the ``split`` column, default ``"train"``.
    suffixes : iterable of str, optional
        Audio extensions to pick up, default ``(".wav",)``.
    require_labels : bool, optional
        Raise when a file sits outside any recognised diagnosis folder
        (default).  Set to ``False`` to index unlabelled material, whose
        ``label`` column is then ``-1``.

    Returns
    -------
    pandas.DataFrame
        Index with the columns of :data:`INDEX_COLUMNS`.

    Raises
    ------
    DatasetError
        If the directory is missing, holds no audio, or (with
        ``require_labels``) contains files with no recognisable diagnosis.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(
            f"no such data directory: {root}. Point paths.data_root at a local copy "
            "of the corpus; no audio ships with this package."
        )

    files = _iter_audio(root, suffixes)
    if not files:
        raise DatasetError(
            f"{root} contains no {'/'.join(suffixes)} file. Check paths.data_root, "
            "paths.train_dir and paths.test_dir."
        )

    rows: list[dict[str, object]] = []
    unlabelled: list[str] = []
    for path in files:
        relative = path.relative_to(root)
        label = _label_from_parts(relative.parts[:-1])
        if label is None:
            if require_labels:
                unlabelled.append(relative.as_posix())
                continue
            label = -1
        rows.append(
            {
                "path": path.as_posix(),
                "file_name": path.name,
                "subject_id": _subject_id(path, root),
                "label": int(label),
                "split": split,
            }
        )

    if unlabelled:
        preview = ", ".join(unlabelled[:5])
        raise DatasetError(
            f"{len(unlabelled)} file(s) are not inside a recognised diagnosis folder "
            f"(e.g. {preview}). Expected folder names: {sorted(set(FOLDER_ALIASES))}"
        )

    frame = pd.DataFrame(rows)
    _LOGGER.info("indexed %d recording(s) under %s", len(frame), root)
    return finalise_index(frame)


def read_demographics(path: str | Path) -> pd.DataFrame:
    """Read a demographics table mapping file names to diagnoses.

    Parameters
    ----------
    path : str or pathlib.Path
        Spreadsheet (``.xlsx``) or delimited text file (``.csv``, ``.tsv``).
        Column names are matched case-insensitively against a set of aliases,
        so headers such as ``"File name"`` or ``"Diagnosis(0-3)"`` are
        understood.  Every cell is read as text, so a file name such as
        ``001`` keeps its leading zeros.

    Returns
    -------
    pandas.DataFrame
        Frame with at least ``file_name`` and ``label`` columns, plus
        ``speaker_id``, ``age`` and ``gender`` when present.

    Raises
    ------
    DatasetError
        If the file is missing, lacks a file-name or diagnosis column, gives
        one field in two columns (``Speaker`` and ``Subject ID``, say), or has
        a row whose diagnosis is blank or cannot be read; the message names
        the columns or files concerned.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"no such demographics file: {path}")

    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        raw = pd.read_csv(path, sep=separator, dtype=str)
    else:
        raw = pd.read_excel(path, dtype=str)

    renamed = {
        column: _DEMOGRAPHIC_ALIASES[str(column).strip().lower()]
        for column in raw.columns
        if str(column).strip().lower() in _DEMOGRAPHIC_ALIASES
    }
    # Two headers for one field (say "Speaker" and "Subject ID") would leave
    # two columns of the same name, and pandas then fails much later with a
    # message that names neither.
    headers_by_field: dict[str, list[str]] = {}
    for column, field in renamed.items():
        headers_by_field.setdefault(field, []).append(repr(str(column)))
    for field, headers in headers_by_field.items():
        if len(headers) > 1:
            raise DatasetError(
                f"{path}: the columns {' and '.join(headers)} name the same field, {field}; "
                "keep one of them"
            )
    frame = raw.rename(columns=renamed)
    missing = {"file_name", "label"} - set(frame.columns)
    if missing:
        raise DatasetError(
            f"{path} lacks the column(s) {sorted(missing)}; found {list(raw.columns)}. "
            f"Recognised headers are {sorted(set(_DEMOGRAPHIC_ALIASES))}"
        )

    kept = [c for c in ("file_name", "label", "speaker_id", "age", "gender") if c in frame]
    frame = frame.loc[:, kept].dropna(subset=["file_name", "label"], how="all")
    frame["file_name"] = frame["file_name"].astype(str).str.strip()

    blank = frame["label"].isna() | (frame["label"].astype(str).str.strip() == "")
    if blank.any():
        names = frame.loc[blank, "file_name"].tolist()
        raise DatasetError(
            f"{path} gives no diagnosis for {len(names)} file(s) (e.g. {', '.join(names[:5])}); "
            "fill in a code 0..3 or a class name"
        )
    labels: list[int] = []
    for name, value in zip(frame["file_name"], frame["label"], strict=True):
        try:
            labels.append(parse_label(value))
        except ValueError as exc:
            raise DatasetError(f"{path}, row of {name}: {exc}") from exc
    frame["label"] = labels
    if "age" in frame:
        frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    _LOGGER.info("read %d demographic row(s) from %s", len(frame), path)
    return frame


def attach_binary_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``label_binary`` and ``label_name`` filled in.

    Parameters
    ----------
    frame : pandas.DataFrame
        Index carrying a ``label`` column.

    Returns
    -------
    pandas.DataFrame
        Copy with the derived columns.  Unlabelled rows (``label < 0``) keep a
        ``label_binary`` of ``-1``.
    """
    out = frame.copy()
    labels = out["label"].to_numpy(dtype=int)
    known = labels >= 0
    binary = pd.Series(-1, index=out.index, dtype=int)
    if known.any():
        binary.loc[known] = to_binary(labels[known])
    out["label_binary"] = binary
    out["label_name"] = [class_name(int(label)) if label >= 0 else "Unknown" for label in labels]
    return out


def finalise_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an index and return it with the standard columns and order.

    Parameters
    ----------
    frame : pandas.DataFrame
        Index with at least ``path``, ``subject_id`` and ``label``.

    Returns
    -------
    pandas.DataFrame
        Sorted copy carrying every column of :data:`INDEX_COLUMNS`.

    Raises
    ------
    DatasetError
        If a required column is missing or a subject identifier repeats.
    """
    required = {"path", "subject_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise DatasetError(f"the index lacks the required column(s) {sorted(missing)}")

    out = frame.copy()
    if "file_name" not in out:
        out["file_name"] = [Path(str(p)).name for p in out["path"]]
    if "split" not in out:
        out["split"] = "train"
    out = attach_binary_labels(out)

    duplicated = out["subject_id"][out["subject_id"].duplicated()].unique()
    if duplicated.size:
        raise DatasetError(
            f"subject identifiers must be unique, but {duplicated[:5].tolist()} repeat. "
            "Grouped cross-validation cannot keep a subject inside one fold otherwise."
        )

    extra = [column for column in out.columns if column not in INDEX_COLUMNS]
    ordered = [*INDEX_COLUMNS, *extra]
    return out.loc[:, ordered].sort_values("subject_id", ignore_index=True)


def build_index(
    cfg: Config,
    *,
    split: str = "train",
    strict_counts: bool = False,
    expected_counts: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Build the file index described by a configuration.

    When ``paths.demographics`` is set, the diagnosis of every recording is
    taken from that table; otherwise it is inferred from the directory names.

    Parameters
    ----------
    cfg : Config
        Configuration whose ``paths`` section locates the corpus.
    split : str, optional
        ``"train"`` (default) or ``"test"``.  The test split is indexed without
        requiring labels.
    strict_counts : bool, optional
        Raise when the per-class file counts differ from ``expected_counts``.
    expected_counts : dict, optional
        Reference counts; defaults to
        :data:`~voice_pathology_detection.data.labels.EXPECTED_TRAIN_COUNTS`,
        the layout of the corpus cited in the README.

    Returns
    -------
    pandas.DataFrame
        The validated index.

    Raises
    ------
    DatasetError
        If the corpus cannot be indexed, or a demographics table does not cover
        every recording.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    root = cfg.paths.train_root if split == "train" else cfg.paths.test_root
    labelled = split == "train"
    demographics_path = cfg.paths.demographics_path if labelled else None

    if demographics_path is not None:
        frame = _index_with_demographics(root, demographics_path, split=split)
    else:
        frame = scan_class_directories(root, split=split, require_labels=labelled)

    if labelled:
        counts = {int(label): int(count) for label, count in frame["label"].value_counts().items()}
        check_class_counts(counts, expected_counts, strict=strict_counts)
        _LOGGER.info(
            "class distribution: %s",
            {class_name(label): counts.get(label, 0) for label in sorted(CLASS_NAMES)},
        )
    return frame


def _lookup(table: pd.DataFrame, column: str) -> dict[str, object]:
    """Map the file names of a demographics table, with and without suffix, to a column."""
    by_name = {
        str(name): value
        for name, value in zip(table["file_name"], table[column], strict=True)
        if not (isinstance(value, float) and pd.isna(value)) and value != ""
    }
    # The table may identify a recording either with or without its suffix.
    return {**{Path(name).stem: value for name, value in by_name.items()}, **by_name}


def _find(lookup: dict[str, object], file_name: str) -> object | None:
    """Look a recording up by its full name first, then by its stem."""
    if file_name in lookup:
        return lookup[file_name]
    return lookup.get(Path(file_name).stem)


def _index_with_demographics(root: Path, table_path: Path, *, split: str) -> pd.DataFrame:
    """Index a flat directory using a demographics table for the labels.

    When the table has a speaker column, it is carried into the index as
    ``speaker_id``; a recording the table gives no speaker for keeps its own
    ``subject_id`` there, so it forms a group of its own under
    ``split.group_by: speaker_id``.
    """
    listing = scan_class_directories(root, split=split, require_labels=False)
    demographics = read_demographics(table_path)
    by_label = _lookup(demographics, "label")

    labels: list[int] = []
    unmatched: list[str] = []
    for file_name in listing["file_name"]:
        name = str(file_name)
        label = _find(by_label, name)
        if label is None:
            unmatched.append(name)
            labels.append(-1)
        else:
            labels.append(int(label))

    if unmatched:
        preview = ", ".join(unmatched[:5])
        raise DatasetError(
            f"{len(unmatched)} recording(s) are missing from {table_path} (e.g. {preview}). "
            "Every training file needs a diagnosis."
        )

    listing = listing.copy()
    listing["label"] = labels
    if "speaker_id" in demographics:
        by_speaker = _lookup(demographics, "speaker_id")
        listing["speaker_id"] = [
            str(_find(by_speaker, str(name)) or subject)
            for name, subject in zip(listing["file_name"], listing["subject_id"], strict=True)
        ]
    return finalise_index(listing)


def summarise_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a per-class summary of an index.

    Parameters
    ----------
    frame : pandas.DataFrame
        A file index.

    Returns
    -------
    pandas.DataFrame
        One row per class with the number of files and, when the column is
        present, the total number of frames.
    """
    columns: dict[str, object] = {"n_files": frame.groupby("label").size()}
    if "n_frames" in frame:
        columns["n_frames"] = frame.groupby("label")["n_frames"].sum()
    summary = pd.DataFrame(columns).reset_index()
    summary["label_name"] = [
        class_name(int(label)) if int(label) >= 0 else "Unknown" for label in summary["label"]
    ]
    return summary.loc[:, ["label", "label_name", *columns]]
