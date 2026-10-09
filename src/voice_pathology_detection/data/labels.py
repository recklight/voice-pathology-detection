"""Class definitions shared by every stage of the pipeline.

The label codes, their display names, the mapping to the binary
normal/pathological task and the folder-name aliases of a corpus laid out as
one directory per diagnosis are all defined here.  No other module hard-codes
a class index.

Examples
--------
>>> VoiceClass.PHONOTRAUMA.value
2
>>> class_name(3)
'Vocal palsy'
>>> to_binary([0, 1, 2, 3]).tolist()
[0, 1, 1, 1]
"""

from __future__ import annotations

import logging
from enum import IntEnum

import numpy as np

_LOGGER = logging.getLogger(__name__)


class VoiceClass(IntEnum):
    """The four diagnostic classes of the four-way task.

    An ``IntEnum``, so the members compare equal to the plain integers stored
    in label arrays.
    """

    NORMAL = 0
    NEOPLASM = 1
    PHONOTRAUMA = 2
    VOCAL_PALSY = 3


class BinaryClass(IntEnum):
    """The two classes of the detection task."""

    NORMAL = 0
    PATHOLOGICAL = 1


#: Display name of every four-way class.
CLASS_NAMES: dict[int, str] = {
    VoiceClass.NORMAL: "Normal",
    VoiceClass.NEOPLASM: "Neoplasm",
    VoiceClass.PHONOTRAUMA: "Phonotrauma",
    VoiceClass.VOCAL_PALSY: "Vocal palsy",
}

#: Display name of the two detection classes.
BINARY_NAMES: dict[int, str] = {
    BinaryClass.NORMAL: "Normal",
    BinaryClass.PATHOLOGICAL: "Pathological",
}

#: Label codes that count as pathological.
PATHOLOGICAL_LABELS: tuple[int, ...] = (
    int(VoiceClass.NEOPLASM),
    int(VoiceClass.PHONOTRAUMA),
    int(VoiceClass.VOCAL_PALSY),
)

#: Label order of the three-way disorder task.
DISORDER_LABELS: tuple[int, ...] = PATHOLOGICAL_LABELS

#: Label order of the four-way task.
ALL_LABELS: tuple[int, ...] = tuple(int(member) for member in VoiceClass)

#: Number of training recordings per class in the corpus cited in the README,
#: whose four classes this package uses.
#:
#: This is a property of that particular corpus, not a requirement of the
#: pipeline.  It is the default reference of :func:`check_class_counts`, so
#: that a corpus assembled class by class can be checked against the layout it
#: is supposed to have; any other material should pass its own counts, or none.
EXPECTED_TRAIN_COUNTS: dict[int, int] = {
    VoiceClass.NORMAL: 50,
    VoiceClass.NEOPLASM: 40,
    VoiceClass.PHONOTRAUMA: 60,
    VoiceClass.VOCAL_PALSY: 50,
}

#: Folder names accepted when a corpus is organised one directory per class.
FOLDER_ALIASES: dict[str, int] = {
    "normal": int(VoiceClass.NORMAL),
    "healthy": int(VoiceClass.NORMAL),
    "neoplasm": int(VoiceClass.NEOPLASM),
    "phonotrauma": int(VoiceClass.PHONOTRAUMA),
    "vocal palsy": int(VoiceClass.VOCAL_PALSY),
    "vocal_palsy": int(VoiceClass.VOCAL_PALSY),
    "vocalpalsy": int(VoiceClass.VOCAL_PALSY),
    "palsy": int(VoiceClass.VOCAL_PALSY),
}


def class_name(label: int, *, binary: bool = False) -> str:
    """Return the display name of a label code.

    Parameters
    ----------
    label : int
        Label code.
    binary : bool, optional
        Look the code up in the two-class table instead of the four-class one.

    Returns
    -------
    str
        Human readable class name.

    Raises
    ------
    ValueError
        If the code is not a known class.

    Examples
    --------
    >>> class_name(1)
    'Neoplasm'
    >>> class_name(1, binary=True)
    'Pathological'
    """
    table = BINARY_NAMES if binary else CLASS_NAMES
    try:
        return table[int(label)]
    except KeyError as exc:
        raise ValueError(
            f"unknown label {label!r}; valid codes are {sorted(int(code) for code in table)}"
        ) from exc


def to_binary(labels: object) -> np.ndarray:
    """Map four-way labels onto the normal/pathological task.

    Parameters
    ----------
    labels : array_like of int
        Four-way label codes.

    Returns
    -------
    numpy.ndarray
        Integer array where ``0`` stays normal and ``1, 2, 3`` all become
        pathological.

    Raises
    ------
    ValueError
        If any code lies outside ``0..3``.

    Examples
    --------
    >>> to_binary([0, 2]).tolist()
    [0, 1]
    """
    arr = np.asarray(labels, dtype=np.int64).ravel()
    unknown = np.setdiff1d(arr, np.asarray(ALL_LABELS, dtype=np.int64))
    if unknown.size:
        raise ValueError(f"labels contain unknown code(s) {unknown.tolist()}; expected 0..3")
    return (arr != int(VoiceClass.NORMAL)).astype(np.int64)


def is_pathological(labels: object) -> np.ndarray:
    """Boolean mask of the entries that are not normal.

    Parameters
    ----------
    labels : array_like of int
        Four-way label codes.

    Returns
    -------
    numpy.ndarray
        Boolean array of the same length.
    """
    return to_binary(labels).astype(bool)


def parse_label(value: object) -> int:
    """Coerce a folder name, class name or number into a label code.

    Parameters
    ----------
    value : str, int or float
        Either a numeric code, a display name or a folder alias such as
        ``"Vocal palsy"`` or ``"vocal_palsy"``.  A whole-number float such as
        ``2.0`` or ``"2.0"`` is accepted too, since a spreadsheet column with a
        blank cell is read as floats.

    Returns
    -------
    int
        The label code.

    Raises
    ------
    ValueError
        If the value cannot be interpreted as a class.

    Examples
    --------
    >>> parse_label("Vocal_Palsy")
    3
    >>> parse_label(2)
    2
    >>> parse_label(1.0)
    1
    """
    if isinstance(value, int | np.integer) and not isinstance(value, bool):
        code = int(value)
        if code not in CLASS_NAMES:
            raise ValueError(f"unknown label code {code}; valid codes are {sorted(CLASS_NAMES)}")
        return code
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return parse_label(int(value))
    text = str(value).strip()
    if text.isdigit():
        return parse_label(int(text))
    whole, dot, fraction = text.partition(".")
    if dot and whole.isdigit() and set(fraction) <= {"0"}:
        return parse_label(int(whole))
    key = text.lower().replace("-", " ").replace("  ", " ")
    if key in FOLDER_ALIASES:
        return FOLDER_ALIASES[key]
    key_underscore = key.replace(" ", "_")
    if key_underscore in FOLDER_ALIASES:
        return FOLDER_ALIASES[key_underscore]
    raise ValueError(
        f"cannot interpret {value!r} as a class; expected a code 0..3 or one of "
        f"{sorted(set(FOLDER_ALIASES))}"
    )


def check_class_counts(
    counts: dict[int, int],
    expected: dict[int, int] | None = None,
    *,
    strict: bool = True,
) -> None:
    """Compare observed per-class file counts with the expected ones.

    Files go missing most easily while a corpus is put together class by
    class, so the counts are checked rather than trusted.

    Parameters
    ----------
    counts : dict
        Observed number of files per label code.
    expected : dict, optional
        Reference counts; defaults to :data:`EXPECTED_TRAIN_COUNTS`, which
        describes the corpus cited in the README.  Material assembled
        differently should pass its own counts here.
    strict : bool, optional
        Raise on a mismatch (default) rather than only logging a warning.

    Raises
    ------
    ValueError
        If ``strict`` and the counts differ from the reference.

    Examples
    --------
    >>> check_class_counts({0: 1, 1: 1}, {0: 1, 1: 1})
    >>> try:
    ...     check_class_counts({0: 1}, {0: 2}, strict=True)
    ... except ValueError as exc:
    ...     print(str(exc).split(";")[0])
    per-class file counts differ from the reference layout (Normal: found 1, expected 2)
    """
    expected = EXPECTED_TRAIN_COUNTS if expected is None else expected
    mismatches = {
        label: (counts.get(label, 0), want)
        for label, want in expected.items()
        if counts.get(label, 0) != want
    }
    if not mismatches:
        _LOGGER.info("class counts match the reference layout: %s", counts)
        return
    detail = ", ".join(
        f"{class_name(label)}: found {got}, expected {want}"
        for label, (got, want) in sorted(mismatches.items())
    )
    message = (
        f"per-class file counts differ from the reference layout ({detail}); "
        "the reference describes one particular corpus, so pass expected_counts "
        "to check your own material against its own layout"
    )
    if strict:
        raise ValueError(message)
    _LOGGER.warning("%s", message)
