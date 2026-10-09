"""Waveform trimming applied before the cepstral front end.

Clinical recordings of a sustained vowel often carry several seconds of
conversation between the clinician and the patient at either end.  That
conversation is not phonation.  Left in, it mixes two very different acoustic
conditions into one feature matrix; in this task I rank it as the costliest
preprocessing mistake.

Besides ``none``, ``trim.mode`` selects one of two strategies:

``fixed_edges``
    Drop a fixed amount of audio from each end, but only from recordings long
    enough to hold any conversation.  It always cuts the same amount and does
    not depend on the recording level.
``energy_vad``
    Locate the phonation with the two-threshold endpoint detector of
    :func:`speechdsp.endpoint_detect`, which adapts to each recording instead of
    assuming a fixed layout.

Both return a :class:`TrimResult` with the kept boundaries, so a cut can be
logged or written back out to listen to.

Examples
--------
>>> import numpy as np
>>> from voice_pathology_detection.config import TrimConfig
>>> sr = 8000
>>> x = np.ones(sr * 6)
>>> result = trim_audio(x, sr, TrimConfig(edge_seconds=1.0, min_seconds=4.0))
>>> result.start, result.end
(8000, 40000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from speechdsp import endpoint_detect

from ..config import TrimConfig

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrimResult:
    """Outcome of a trimming decision.

    Attributes
    ----------
    audio : numpy.ndarray
        The trimmed waveform.
    start, end : int
        Sample boundaries of the kept segment in the original waveform, so that
        ``original[start:end]`` reproduces :attr:`audio`.
    mode : str
        Strategy that produced the boundaries.
    applied : bool
        ``False`` when the waveform was returned unchanged, either because the
        mode was ``none`` or because a guard rejected the proposed cut.
    """

    audio: np.ndarray
    start: int
    end: int
    mode: str
    applied: bool

    @property
    def n_samples(self) -> int:
        """Length of the kept segment in samples."""
        return int(self.end - self.start)

    def duration(self, sr: int) -> float:
        """Duration of the kept segment in seconds."""
        if sr <= 0:
            raise ValueError("sr must be positive")
        return self.n_samples / float(sr)


def _validated(x: object, sr: int) -> np.ndarray:
    """Return ``x`` as a 1-D float array after checking the sampling rate."""
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("cannot trim an empty waveform")
    return arr


def trim_fixed_edges(
    x: object,
    sr: int,
    edge_seconds: float = 2.0,
    min_seconds: float = 4.5,
) -> TrimResult:
    """Remove a fixed amount of audio from both ends of a long recording.

    Parameters
    ----------
    x : array_like
        Input waveform.
    sr : int
        Sampling rate in Hz.
    edge_seconds : float, optional
        Seconds removed from each end, default 2.
    min_seconds : float, optional
        Recordings at or below this duration are returned untouched, default
        4.5.  Short prompts consist of phonation only and must not be cut.

    Returns
    -------
    TrimResult
        The trimmed waveform and its boundaries.

    Raises
    ------
    ValueError
        If ``sr`` is not positive, the waveform is empty or ``edge_seconds`` is
        negative.

    Examples
    --------
    >>> import numpy as np
    >>> res = trim_fixed_edges(np.ones(1000), 1000, edge_seconds=0.1, min_seconds=0.5)
    >>> res.start, res.end
    (100, 900)
    """
    arr = _validated(x, sr)
    if edge_seconds < 0:
        raise ValueError(f"edge_seconds must not be negative, got {edge_seconds}")

    if arr.size <= round(min_seconds * sr):
        _LOGGER.debug(
            "recording of %d samples is at or below the %.2f s guard; left untouched",
            arr.size,
            min_seconds,
        )
        return TrimResult(arr, 0, int(arr.size), "fixed_edges", applied=False)

    edge = round(edge_seconds * sr)
    start, end = edge, int(arr.size) - edge
    if end - start <= 0:
        _LOGGER.warning(
            "trimming %.2f s from both ends would empty a recording of %d samples; "
            "keeping it unchanged",
            edge_seconds,
            arr.size,
        )
        return TrimResult(arr, 0, int(arr.size), "fixed_edges", applied=False)
    return TrimResult(arr[start:end], start, end, "fixed_edges", applied=True)


def trim_energy_vad(
    x: object,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> TrimResult:
    """Keep the phonated segment found by the energy endpoint detector.

    Parameters
    ----------
    x : array_like
        Input waveform.
    sr : int
        Sampling rate in Hz.
    frame_ms, hop_ms : float, optional
        Analysis grid of the detector.

    Returns
    -------
    TrimResult
        The detected segment; ``applied`` is ``False`` when the detector fell
        back to the whole recording.

    Notes
    -----
    The detection itself is :func:`speechdsp.endpoint_detect`: the usual
    two-threshold algorithm on short-time energy, refined by the zero-crossing
    rate.
    """
    arr = _validated(x, sr)
    start, end = endpoint_detect(arr, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    applied = not (start == 0 and end == arr.size)
    if not applied:
        _LOGGER.debug("endpoint detector kept the whole recording of %d samples", arr.size)
    return TrimResult(arr[start:end], int(start), int(end), "energy_vad", applied=applied)


def trim_audio(x: object, sr: int, cfg: TrimConfig | None = None) -> TrimResult:
    """Apply the trimming strategy named by the configuration.

    A trim that would leave less than ``cfg.min_output_seconds`` of audio is
    rejected and the original waveform kept, so that a mis-set parameter can
    never silently destroy a recording.

    Parameters
    ----------
    x : array_like
        Input waveform.
    sr : int
        Sampling rate in Hz.
    cfg : TrimConfig, optional
        Trimming settings; the defaults are used when omitted.

    Returns
    -------
    TrimResult
        The trimmed waveform and the boundaries that produced it.

    Raises
    ------
    ValueError
        If ``sr`` is not positive or the waveform is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from voice_pathology_detection.config import TrimConfig
    >>> res = trim_audio(np.ones(500), 1000, TrimConfig(mode="none"))
    >>> res.applied
    False
    """
    cfg = TrimConfig() if cfg is None else cfg
    arr = _validated(x, sr)

    match cfg.mode:
        case "none":
            return TrimResult(arr, 0, int(arr.size), "none", applied=False)
        case "fixed_edges":
            result = trim_fixed_edges(
                arr, sr, edge_seconds=cfg.edge_seconds, min_seconds=cfg.min_seconds
            )
        case "energy_vad":
            result = trim_energy_vad(arr, sr, frame_ms=cfg.frame_ms, hop_ms=cfg.hop_ms)
        case _:  # pragma: no cover - the configuration validates the mode
            raise ValueError(f"unsupported trim mode: {cfg.mode!r}")

    floor = round(cfg.min_output_seconds * sr)
    if result.applied and result.n_samples < floor:
        _LOGGER.warning(
            "%s trimming would leave only %d samples (< %.2f s); keeping the original %d samples",
            cfg.mode,
            result.n_samples,
            cfg.min_output_seconds,
            arr.size,
        )
        return TrimResult(arr, 0, int(arr.size), cfg.mode, applied=False)
    return result
