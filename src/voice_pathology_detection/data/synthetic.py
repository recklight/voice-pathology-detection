"""Synthetic sustained vowels, for demonstrations and tests.

No recording of a human voice ships with this package, and none should.
Everything that needs data to run (the example scripts, the tests, the
``demo`` command of the CLI) builds its own corpus here from a source-filter
model: a glottal pulse train with adjustable jitter, shimmer, tremor and
turbulent noise, shaped by three formant resonators tuned to the vowel /a/.

The per-class settings are caricatures, not clinical models.  I chose them so
the four classes are separable enough to run the pipeline end to end; given a
seed, the output is deterministic.

Examples
--------
>>> import numpy as np
>>> x = synthesize_voice(0, sr=8000, duration=0.5, rng=np.random.default_rng(0))
>>> x.shape
(4000,)
>>> float(np.abs(x).max()) <= 1.0
True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from speechdsp import write_wav

from .labels import CLASS_NAMES, VoiceClass, parse_label

_LOGGER = logging.getLogger(__name__)

#: Formant centre frequencies and bandwidths of the vowel /a/, in Hz.
FORMANTS: tuple[tuple[float, float], ...] = ((730.0, 90.0), (1090.0, 110.0), (2440.0, 140.0))


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """Source parameters of one synthetic voice class.

    Attributes
    ----------
    f0 : float
        Mean fundamental frequency in Hz.
    jitter : float
        Relative standard deviation of the cycle-to-cycle frequency.
    shimmer : float
        Relative standard deviation of the cycle-to-cycle amplitude.
    noise : float
        Amplitude of the turbulent noise relative to the harmonic source.
    tremor_hz, tremor_depth : float
        Rate and depth of a slow amplitude modulation.
    subharmonic : float
        Amplitude of a component at half the fundamental, which is what makes a
        voice sound rough.
    n_harmonics : int
        Number of harmonics in the glottal source.
    """

    f0: float
    jitter: float
    shimmer: float
    noise: float
    tremor_hz: float
    tremor_depth: float
    subharmonic: float
    n_harmonics: int


#: Source settings per class.  Deliberately exaggerated, see the module note.
PROFILES: dict[int, VoiceProfile] = {
    int(VoiceClass.NORMAL): VoiceProfile(
        f0=155.0,
        jitter=0.003,
        shimmer=0.02,
        noise=0.015,
        tremor_hz=4.5,
        tremor_depth=0.02,
        subharmonic=0.0,
        n_harmonics=32,
    ),
    int(VoiceClass.NEOPLASM): VoiceProfile(
        f0=138.0,
        jitter=0.030,
        shimmer=0.12,
        noise=0.320,
        tremor_hz=5.0,
        tremor_depth=0.06,
        subharmonic=0.05,
        n_harmonics=14,
    ),
    int(VoiceClass.PHONOTRAUMA): VoiceProfile(
        f0=196.0,
        jitter=0.014,
        shimmer=0.08,
        noise=0.140,
        tremor_hz=5.5,
        tremor_depth=0.05,
        subharmonic=0.35,
        n_harmonics=24,
    ),
    int(VoiceClass.VOCAL_PALSY): VoiceProfile(
        f0=118.0,
        jitter=0.020,
        shimmer=0.16,
        noise=0.230,
        tremor_hz=6.5,
        tremor_depth=0.45,
        subharmonic=0.10,
        n_harmonics=18,
    ),
}


def _formant_filter(x: np.ndarray, sr: int) -> np.ndarray:
    """Shape a glottal source with a cascade of two-pole formant resonators."""
    out = x
    for centre, bandwidth in FORMANTS:
        if centre >= sr / 2.0:
            continue
        radius = float(np.exp(-np.pi * bandwidth / sr))
        theta = 2.0 * np.pi * centre / sr
        denominator = np.asarray([1.0, -2.0 * radius * np.cos(theta), radius**2])
        numerator = np.asarray([1.0 - radius])
        out = lfilter(numerator, denominator, out)
    return np.asarray(out, dtype=np.float64)


def synthesize_voice(
    label: object,
    sr: int = 44100,
    duration: float = 1.0,
    rng: np.random.Generator | None = None,
    *,
    profile: VoiceProfile | None = None,
) -> np.ndarray:
    """Synthesise one sustained vowel of the requested class.

    Parameters
    ----------
    label : int or str
        Class code or name, resolved through
        :func:`voice_pathology_detection.data.labels.parse_label`.
    sr : int, optional
        Sampling rate in Hz, default 44100.
    duration : float, optional
        Length in seconds, default 1.
    rng : numpy.random.Generator, optional
        Source of randomness; a fresh default generator is used when omitted,
        which makes the call non-reproducible, so pass one in tests.
    profile : VoiceProfile, optional
        Override the source settings of the class.

    Returns
    -------
    numpy.ndarray
        Waveform of ``round(duration * sr)`` samples, peak-normalised to 0.9.

    Raises
    ------
    ValueError
        If the sampling rate or the duration is not positive.

    Notes
    -----
    The instantaneous phase is integrated from a jittered fundamental, so the
    harmonics stay phase-coherent; this is what separates a voice-like signal
    from a bank of independent sinusoids.
    """
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    code = parse_label(label)
    profile = PROFILES[code] if profile is None else profile
    rng = np.random.default_rng() if rng is None else rng

    n = round(duration * sr)
    t = np.arange(n, dtype=np.float64) / sr

    # Jittered fundamental: a smooth random walk around the mean f0.
    raw_jitter = rng.normal(0.0, profile.jitter, size=n)
    smooth = np.convolve(raw_jitter, np.ones(64) / 64.0, mode="same")
    f0_t = profile.f0 * (1.0 + smooth)
    phase = 2.0 * np.pi * np.cumsum(f0_t) / sr

    harmonics = np.arange(1, profile.n_harmonics + 1, dtype=np.float64)
    amplitudes = 1.0 / harmonics  # -6 dB per octave source tilt
    source = (amplitudes[:, None] * np.sin(harmonics[:, None] * phase[None, :])).sum(axis=0)
    if profile.subharmonic > 0.0:
        source += profile.subharmonic * np.sin(0.5 * phase)

    shimmer = 1.0 + rng.normal(0.0, profile.shimmer, size=n)
    tremor = 1.0 + profile.tremor_depth * np.sin(2.0 * np.pi * profile.tremor_hz * t)
    source = source * shimmer * tremor
    source = source + profile.noise * rng.normal(0.0, 1.0, size=n) * np.abs(source).mean()

    shaped = _formant_filter(source, sr)
    fade = min(n // 20 + 1, n)
    envelope = np.ones(n)
    ramp = np.linspace(0.0, 1.0, fade)
    envelope[:fade] = ramp
    envelope[n - fade :] = ramp[::-1]
    shaped = shaped * envelope

    peak = float(np.abs(shaped).max())
    if peak <= 0.0:  # pragma: no cover - only reachable for a degenerate profile
        return np.zeros(n)
    return 0.9 * shaped / peak


def write_synthetic_corpus(
    root: str | Path,
    counts: dict[int, int] | None = None,
    *,
    sr: int = 16000,
    duration: float = 1.0,
    seed: int = 0,
    split: str = "train",
) -> Path:
    """Write a small synthetic corpus laid out as one folder per class.

    Parameters
    ----------
    root : str or pathlib.Path
        Destination directory; created if missing.
    counts : dict, optional
        Number of files per label code, default four of each class.
    sr : int, optional
        Sampling rate, default 16000, which keeps the files small.
    duration : float, optional
        Duration of every recording in seconds, default 1.
    seed : int, optional
        Seed making the corpus reproducible, default 0.
    split : str, optional
        Name of the sub-folder written under ``root``, default ``"train"``.

    Returns
    -------
    pathlib.Path
        The directory that was written, i.e. ``root / split``.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     out = write_synthetic_corpus(tmp, {0: 1, 1: 1}, sr=8000, duration=0.2)
    ...     sorted(p.name for p in out.iterdir())
    ['Neoplasm', 'Normal']
    """
    counts = dict.fromkeys(CLASS_NAMES, 4) if counts is None else counts
    target = Path(root) / split
    rng = np.random.default_rng(seed)

    for label, count in sorted(counts.items()):
        code = parse_label(label)
        folder = target / CLASS_NAMES[code]
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(int(count)):
            audio = synthesize_voice(code, sr=sr, duration=duration, rng=rng)
            write_wav(folder / f"{i + 1:03d}.wav", audio, sr)
    total = sum(int(c) for c in counts.values())
    _LOGGER.info("wrote %d synthetic recording(s) to %s", total, target)
    return target


def make_frame_dataset(
    n_files_per_class: int = 6,
    n_frames: int = 20,
    n_dim: int = 39,
    labels: tuple[int, ...] = (0, 1, 2, 3),
    separation: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build a labelled frame matrix directly, bypassing audio synthesis.

    For tests that need a feature matrix with known structure and no
    front-end cost.

    Parameters
    ----------
    n_files_per_class : int, optional
        Number of recordings per class, default 6.
    n_frames : int, optional
        Frames contributed by every recording, default 20.
    n_dim : int, optional
        Feature dimension, default 39.
    labels : tuple of int, optional
        Classes to generate, default all four.
    separation : float, optional
        Distance between the class means in units of the within-class standard
        deviation; larger values make the problem easier.
    seed : int, optional
        Seed, default 0.

    Returns
    -------
    X : numpy.ndarray
        Frame matrix of shape ``(n_files * n_frames, n_dim)``.
    frame_file : numpy.ndarray
        For every frame, the row position of the file it came from.
    files : pandas.DataFrame
        File index with ``path``, ``subject_id``, ``label`` and ``n_frames``.

    Examples
    --------
    >>> X, frame_file, files = make_frame_dataset(2, 5, 4, labels=(0, 1))
    >>> X.shape, files.shape[0], int(frame_file.max())
    ((20, 4), 4, 3)
    """
    if n_files_per_class < 1 or n_frames < 1 or n_dim < 1:
        raise ValueError("n_files_per_class, n_frames and n_dim must all be positive")

    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 1.0, size=(len(labels), n_dim)) * separation

    blocks: list[np.ndarray] = []
    frame_file: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    file_position = 0
    for class_index, label in enumerate(labels):
        for i in range(n_files_per_class):
            offset = rng.normal(0.0, 0.25, size=n_dim)  # per-subject bias
            block = centres[class_index] + offset + rng.normal(0.0, 1.0, size=(n_frames, n_dim))
            blocks.append(block)
            frame_file.append(np.full(n_frames, file_position, dtype=int))
            rows.append(
                {
                    "path": f"synthetic/{CLASS_NAMES[int(label)]}/{i + 1:03d}.wav",
                    "file_name": f"{i + 1:03d}.wav",
                    "subject_id": f"{CLASS_NAMES[int(label)]}/{i + 1:03d}",
                    "label": int(label),
                    "split": "train",
                    "n_frames": int(n_frames),
                }
            )
            file_position += 1

    files = pd.DataFrame(rows)
    return np.vstack(blocks), np.concatenate(frame_file), files
