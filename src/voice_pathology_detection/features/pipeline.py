"""From a file index to a frame matrix, with the bookkeeping kept intact.

Frames are classified but subjects are diagnosed, so the link between the two
has to survive every step: normalisation, training, aggregation and the
selection of the pathological subset in the second stage of the cascade.
:class:`FeatureSet` is that link.  It carries the frame matrix, the file each
frame came from, and the file index itself, and every operation that subsets
the data goes through :meth:`FeatureSet.select` so the three stay consistent.

Examples
--------
>>> import numpy as np, pandas as pd
>>> files = pd.DataFrame(
...     {"subject_id": ["a", "b"], "label": [0, 1], "n_frames": [2, 3]}
... )
>>> fs = FeatureSet(np.zeros((5, 4)), np.array([0, 0, 1, 1, 1]), files)
>>> fs.group_sizes.tolist()
[2, 3]
>>> fs.select([1]).n_frames
3
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from speechdsp import read_wav

from ..config import Config, MFCCConfig, TrimConfig
from ..data.trim import trim_audio
from .mfcc import FeatureError, extract_features

_LOGGER = logging.getLogger(__name__)

#: Version tag of the cache layout; bump it when the feature format changes.
CACHE_VERSION = "1"


@dataclass(slots=True)
class FeatureSet:
    """Frames, their provenance and the file index they belong to.

    Attributes
    ----------
    X : numpy.ndarray
        Frame matrix of shape ``(n_frames, n_dim)``.
    frame_file : numpy.ndarray
        For every frame, the row position of its file in :attr:`files`.  Frames
        of one file are contiguous and files appear in index order, which is
        what makes :attr:`group_sizes` a valid partition.
    files : pandas.DataFrame
        File index with at least a ``label`` column, its row positions matching
        the values in :attr:`frame_file`.  An ``n_frames`` column is filled in
        from :attr:`frame_file` when absent and checked against it otherwise.
    """

    X: np.ndarray
    frame_file: np.ndarray
    files: pd.DataFrame

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.frame_file = np.asarray(self.frame_file, dtype=np.int64).ravel()
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        if self.X.shape[0] != self.frame_file.size:
            raise ValueError(
                f"X has {self.X.shape[0]} frame(s) but frame_file has "
                f"{self.frame_file.size} entries"
            )
        self.files = self.files.reset_index(drop=True)
        if self.frame_file.size and self.frame_file.max() >= len(self.files):
            raise ValueError(
                f"frame_file refers to file {int(self.frame_file.max())} but the index "
                f"holds only {len(self.files)} row(s)"
            )
        if np.any(np.diff(self.frame_file) < 0):
            raise ValueError(
                "frames must be grouped by file and ordered by file position; "
                "the frame-to-file map is not monotonic"
            )
        counts = np.bincount(self.frame_file, minlength=len(self.files))
        if "n_frames" not in self.files.columns:
            self.files = self.files.assign(n_frames=counts)
        elif not np.array_equal(self.files["n_frames"].to_numpy(dtype=np.int64), counts):
            wrong = np.flatnonzero(self.files["n_frames"].to_numpy(dtype=np.int64) != counts)
            raise ValueError(
                f"files['n_frames'] disagrees with frame_file for {wrong.size} file(s), "
                f"e.g. row {int(wrong[0])}: {int(self.files['n_frames'].iloc[wrong[0]])} "
                f"listed, {int(counts[wrong[0]])} in the frame map"
            )

    # -- shape -----------------------------------------------------------
    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.X.shape[0])

    @property
    def n_dim(self) -> int:
        """Feature dimension."""
        return int(self.X.shape[1])

    @property
    def n_files(self) -> int:
        """Number of recordings."""
        return len(self.files)

    @property
    def group_sizes(self) -> np.ndarray:
        """Frame count of every recording, in file order."""
        return self.files["n_frames"].to_numpy(dtype=np.int64)

    # -- labels ----------------------------------------------------------
    @property
    def file_labels(self) -> np.ndarray:
        """Label of every recording."""
        return self.files["label"].to_numpy(dtype=np.int64)

    @property
    def frame_labels(self) -> np.ndarray:
        """Label of every frame, broadcast from its file."""
        return self.file_labels[self.frame_file]

    @property
    def subjects(self) -> np.ndarray:
        """Subject identifier of every recording."""
        return self.files["subject_id"].to_numpy()

    @property
    def frame_subjects(self) -> np.ndarray:
        """Subject identifier of every frame."""
        return self.subjects[self.frame_file]

    # -- subsetting ------------------------------------------------------
    def select(self, rows: object) -> FeatureSet:
        """Return the subset covering the given files.

        Parameters
        ----------
        rows : array_like
            Either a boolean mask over the files or their integer positions.
            Positions are sorted before use, so the result keeps the order of
            the index and its ``group_sizes`` stay a valid partition.

        Returns
        -------
        FeatureSet
            A new set holding only the requested recordings, with
            ``frame_file`` renumbered to the new file positions.

        Raises
        ------
        IndexError
            If a position lies outside the index.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> files = pd.DataFrame({"subject_id": list("abc"), "label": [0, 1, 2]})
        >>> fs = FeatureSet(np.zeros((3, 2)), np.arange(3), files)
        >>> fs.select([True, False, True]).files["subject_id"].tolist()
        ['a', 'c']
        """
        positions = np.asarray(rows)
        if positions.dtype == bool:
            if positions.size != self.n_files:
                raise IndexError(
                    f"boolean mask of length {positions.size} does not match {self.n_files} file(s)"
                )
            positions = np.flatnonzero(positions)
        positions = np.unique(np.asarray(positions, dtype=np.int64))
        if positions.size and (positions.min() < 0 or positions.max() >= self.n_files):
            raise IndexError(f"file position(s) out of range for an index of {self.n_files} row(s)")

        keep = np.isin(self.frame_file, positions)
        remap = np.full(self.n_files, -1, dtype=np.int64)
        remap[positions] = np.arange(positions.size, dtype=np.int64)
        return FeatureSet(
            X=self.X[keep],
            frame_file=remap[self.frame_file[keep]],
            files=self.files.iloc[positions].reset_index(drop=True),
        )

    def frames_of(self, file_position: int) -> np.ndarray:
        """Return the frames contributed by one recording.

        Parameters
        ----------
        file_position : int
            Row position of the recording in :attr:`files`.

        Returns
        -------
        numpy.ndarray
            The rows of :attr:`X` belonging to that recording.
        """
        if not 0 <= file_position < self.n_files:
            raise IndexError(
                f"file position {file_position} is outside an index of {self.n_files} row(s)"
            )
        return self.X[self.frame_file == file_position]

    def with_features(self, X: np.ndarray) -> FeatureSet:
        """Return a copy whose frame matrix is replaced by ``X``.

        Parameters
        ----------
        X : numpy.ndarray
            New frame matrix with the same number of rows.

        Returns
        -------
        FeatureSet
            A set sharing the provenance bookkeeping of this one.
        """
        array = np.asarray(X, dtype=np.float64)
        if array.shape[0] != self.n_frames:
            raise ValueError(
                f"replacement matrix has {array.shape[0]} frame(s), expected {self.n_frames}"
            )
        return FeatureSet(X=array, frame_file=self.frame_file.copy(), files=self.files.copy())

    # -- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the set to a ``.npz`` file plus a sidecar index.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination of the ``.npz`` file; the file index is written next to
            it with a ``.index.csv`` suffix.

        Returns
        -------
        pathlib.Path
            The path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, frame_file=self.frame_file)
        self.files.to_csv(path.with_suffix(".index.csv"), index=False)
        _LOGGER.info("saved %d frame(s) of %d file(s) to %s", self.n_frames, self.n_files, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> FeatureSet:
        """Read a set written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Path of the ``.npz`` file.

        Returns
        -------
        FeatureSet
            The reconstructed set.

        Raises
        ------
        FileNotFoundError
            If either the archive or its sidecar index is missing.

        Notes
        -----
        Identifiers and paths are read back as text, so ``001`` stays ``001``.
        """
        path = Path(path)
        sidecar = path.with_suffix(".index.csv")
        if not path.is_file():
            raise FileNotFoundError(f"no such feature archive: {path}")
        if not sidecar.is_file():
            raise FileNotFoundError(f"missing the file index next to {path}: {sidecar}")
        with np.load(path) as archive:
            X, frame_file = archive["X"], archive["frame_file"]
        text_columns = ("path", "file_name", "subject_id", "speaker_id", "label_name", "split")
        files = pd.read_csv(sidecar, dtype=dict.fromkeys(text_columns, str))
        return cls(X=X, frame_file=frame_file, files=files)


def _cache_key(path: Path, mfcc_cfg: MFCCConfig, trim_cfg: TrimConfig) -> str:
    """Build a cache file name from the recording and the settings that shape it.

    The recording is identified by its path, size and modification time rather
    than by its contents, which keeps the lookup cheap; touching a file is
    therefore enough to invalidate its entry.
    """
    try:
        stat = path.stat()
        fingerprint = [str(path.resolve()), stat.st_size, int(stat.st_mtime)]
    except OSError:  # pragma: no cover - only if the file vanishes mid-run
        fingerprint = [str(path)]
    payload = {
        "version": CACHE_VERSION,
        "file": fingerprint,
        "mfcc": {f: getattr(mfcc_cfg, f) for f in mfcc_cfg.__slots__},
        "trim": {f: getattr(trim_cfg, f) for f in trim_cfg.__slots__},
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def extract_file(
    path: str | Path,
    mfcc_cfg: MFCCConfig | None = None,
    trim_cfg: TrimConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Read one recording and return its feature matrix.

    Parameters
    ----------
    path : str or pathlib.Path
        WAV file to analyse.
    mfcc_cfg : MFCCConfig, optional
        Front-end settings.
    trim_cfg : TrimConfig, optional
        Trimming settings applied before the front end.
    cache_dir : str or pathlib.Path, optional
        Directory holding cached matrices.  A cache entry is reused when the
        path, size, modification time and both configurations are unchanged.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_frames, n_dim)``.

    Raises
    ------
    FileNotFoundError
        If the recording does not exist.
    FeatureError
        If the recording is too short to yield a frame.
    """
    path = Path(path)
    mfcc_cfg = MFCCConfig() if mfcc_cfg is None else mfcc_cfg
    trim_cfg = TrimConfig() if trim_cfg is None else trim_cfg

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{_cache_key(path, mfcc_cfg, trim_cfg)}.npy"
        if cache_path.is_file():
            _LOGGER.debug("cache hit for %s", path.name)
            return np.load(cache_path)

    audio, sr = read_wav(path)
    trimmed = trim_audio(audio, sr, trim_cfg)
    features = extract_features(trimmed.audio, sr, mfcc_cfg)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
    return features


def extract_corpus(
    index: pd.DataFrame,
    cfg: Config | None = None,
    *,
    cache_dir: str | Path | None = None,
    skip_errors: bool = True,
) -> FeatureSet:
    """Turn a file index into a :class:`FeatureSet`.

    Parameters
    ----------
    index : pandas.DataFrame
        File index with at least ``path`` and ``label`` columns.
    cfg : Config, optional
        Configuration supplying the front-end and trimming settings; the
        defaults are used when omitted.
    cache_dir : str or pathlib.Path, optional
        Feature cache directory; ``None`` disables caching.
    skip_errors : bool, optional
        Drop the recordings the front end rejects and log how many (default),
        or re-raise the first failure.

    Returns
    -------
    FeatureSet
        Frames of every recording that could be analysed, with the file index
        reduced to those recordings.

    Raises
    ------
    ValueError
        If the index is empty or lacks a ``path`` column, or if no recording
        could be analysed at all.
    FeatureError
        If ``skip_errors`` is ``False`` and a recording cannot be analysed.

    Notes
    -----
    Skipped recordings are counted in one warning that names the first three.
    Dropped silently, they would make a class that lost files look the same as
    a class that never had them.
    """
    cfg = Config() if cfg is None else cfg
    if "path" not in index.columns:
        raise ValueError(f"the index needs a 'path' column, found {sorted(index.columns)}")
    if len(index) == 0:
        raise ValueError("cannot extract features from an empty index")

    blocks: list[np.ndarray] = []
    frame_file: list[np.ndarray] = []
    kept_rows: list[int] = []
    counts: list[int] = []
    failures: list[tuple[str, str]] = []

    for position, (_, row) in enumerate(index.iterrows()):
        path = Path(str(row["path"]))
        try:
            features = extract_file(path, cfg.mfcc, cfg.trim, cache_dir=cache_dir)
        except (FileNotFoundError, FeatureError, ValueError) as exc:
            if not skip_errors:
                raise
            failures.append((path.name, str(exc)))
            continue
        blocks.append(features)
        frame_file.append(np.full(features.shape[0], len(kept_rows), dtype=np.int64))
        kept_rows.append(position)
        counts.append(int(features.shape[0]))

    if failures:
        _LOGGER.warning(
            "skipped %d of %d recording(s); first failures: %s",
            len(failures),
            len(index),
            "; ".join(f"{name}: {reason}" for name, reason in failures[:3]),
        )
    if not blocks:
        raise ValueError(
            f"none of the {len(index)} recording(s) could be analysed; "
            "check the sampling rate, the trimming settings and mfcc.frame_ms"
        )

    files = index.iloc[kept_rows].reset_index(drop=True).assign(n_frames=counts)
    feature_set = FeatureSet(
        X=np.vstack(blocks), frame_file=np.concatenate(frame_file), files=files
    )
    _LOGGER.info(
        "extracted %d frame(s) of %d dimension(s) from %d recording(s)",
        feature_set.n_frames,
        feature_set.n_dim,
        feature_set.n_files,
    )
    return feature_set
