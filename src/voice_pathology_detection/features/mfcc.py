"""Cepstral front end in the HTK ``MFCC_0`` coefficient order.

The numerical work (pre-emphasis, framing, the mel filterbank, the discrete
cosine transform and the sinusoidal lifter) is done by :func:`speechdsp.mfcc`,
from a separate package with its own tests.  The parts specific to this
project are here:

* the band limit applied to the waveform before analysis (a 64 Hz high-pass by
  default, see :class:`~voice_pathology_detection.config.MFCCConfig`);
* the HTK coefficient order ``[c1 ... cL, c0]``, where the zeroth term is the
  *last* column of the block rather than the first;
* the HTK scaling of ``c0``, which uses the same ``sqrt(2 / n_mels)`` factor as
  every other coefficient, whereas an orthonormal DCT-II scales its zeroth term
  by ``sqrt(1 / n_mels)``;
* frame-count bookkeeping, and an error naming the problem for a recording
  too short to yield a frame or holding a non-finite sample.

Examples
--------
>>> import numpy as np
>>> from voice_pathology_detection.config import MFCCConfig
>>> sr = 16000
>>> cfg = MFCCConfig(sample_rate=sr, low_freq=None)
>>> x = np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
>>> extract_static(x, sr, cfg).shape[1]
13

References
----------
.. [1] S. B. Davis and P. Mermelstein, "Comparison of parametric
       representations for monosyllabic word recognition in continuously spoken
       sentences", *IEEE Transactions on Acoustics, Speech, and Signal
       Processing*, 28(4):357-366, 1980.
.. [2] S. Young, G. Evermann, M. Gales, et al., *The HTK Book (for HTK Version
       3.4)*, Cambridge University Engineering Department, 2006.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt
from speechdsp import mfcc as _speechdsp_mfcc

from ..config import MFCCConfig
from .deltas import append_deltas

_LOGGER = logging.getLogger(__name__)

#: Factor that turns the orthonormal DCT-II ``c0`` into the HTK one:
#: ``sqrt(2 / n_mels) / sqrt(1 / n_mels) = sqrt(2)`` (see the module docstring).
C0_SCALE: float = float(np.sqrt(2.0))

#: Order of the Butterworth filter used for the optional band limit.
BAND_FILTER_ORDER: int = 4


class FeatureError(ValueError):
    """Raised when a waveform cannot be turned into features."""


def n_fft_for(sr: int, cfg: MFCCConfig) -> int:
    """Return the FFT size to analyse a signal sampled at ``sr``.

    Parameters
    ----------
    sr : int
        Actual sampling rate of the waveform, which may differ from
        ``cfg.sample_rate``.
    cfg : MFCCConfig
        Front-end settings.

    Returns
    -------
    int
        ``cfg.n_fft`` when set and large enough, otherwise the next power of
        two at or above the frame length, so that no sample is discarded.
    """
    frame_len = max(round(sr * cfg.frame_ms / 1000.0), 1)
    if cfg.n_fft is None:
        return 1 << max(frame_len - 1, 1).bit_length()
    if cfg.n_fft < frame_len:
        bumped = 1 << max(frame_len - 1, 1).bit_length()
        _LOGGER.warning(
            "mfcc.n_fft=%d is smaller than the %d-sample frame at %d Hz; using %d instead",
            cfg.n_fft,
            frame_len,
            sr,
            bumped,
        )
        return bumped
    return int(cfg.n_fft)


def band_limit(x: np.ndarray, sr: int, band: tuple[float, float] | None) -> np.ndarray:
    """Restrict a waveform to a frequency band with a zero-phase filter.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform.
    sr : int
        Sampling rate in Hz.
    band : tuple of float or None
        ``(low, high)`` limits in Hz.  ``None``, or a band that already spans
        the whole spectrum, leaves the signal untouched.

    Returns
    -------
    numpy.ndarray
        The filtered waveform, or the input when no filtering was needed.

    Notes
    -----
    :func:`scipy.signal.sosfiltfilt` is used so the band limit introduces no
    phase distortion, which matters because the frames are cut on the filtered
    signal.  Very short recordings cannot be filtered with zero phase; they are
    passed through unchanged with a warning.
    """
    if band is None:
        return x
    low, high = band
    nyquist = sr / 2.0
    wants_high = low > 0.0
    wants_low = high < nyquist
    if not (wants_high or wants_low):
        return x

    if wants_high and wants_low:
        critical: float | list[float] = [low, high]
        btype = "bandpass"
    elif wants_high:
        critical, btype = low, "highpass"
    else:
        critical, btype = high, "lowpass"

    sos = butter(BAND_FILTER_ORDER, critical, btype=btype, fs=sr, output="sos")
    padlen = 3 * (2 * BAND_FILTER_ORDER + 1)
    if x.size <= padlen:
        _LOGGER.warning(
            "waveform of %d samples is too short for zero-phase band limiting; skipping it",
            x.size,
        )
        return x
    return np.asarray(sosfiltfilt(sos, x), dtype=np.float64)


def extract_static(x: object, sr: int, cfg: MFCCConfig | None = None) -> np.ndarray:
    """Compute static cepstral coefficients in ``[c1 ... cL, c0]`` order.

    Parameters
    ----------
    x : array_like
        Input waveform, flattened to 1-D.
    sr : int
        Sampling rate in Hz.
    cfg : MFCCConfig, optional
        Front-end settings; the defaults are used when omitted.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, cfg.n_static)`` whose last column is the
        zeroth cepstral coefficient.

    Raises
    ------
    FeatureError
        If the waveform is empty, holds a NaN or infinite sample, or is too
        short to fill a single analysis frame (naming the duration that would
        be required).  The zero-phase band filter would spread a single bad
        sample over the whole recording, so it is refused up front.

    Notes
    -----
    Liftering is applied to ``c1 ... cL`` only; the lifter weight of the zeroth
    coefficient is one by construction, so moving that column does not change
    its value.
    """
    cfg = MFCCConfig() if cfg is None else cfg
    if sr <= 0:
        raise FeatureError(f"sr must be positive, got {sr}")
    signal = np.asarray(x, dtype=np.float64).ravel()
    if signal.size == 0:
        raise FeatureError("cannot extract features from an empty waveform")
    if not np.all(np.isfinite(signal)):
        bad = int(np.count_nonzero(~np.isfinite(signal)))
        raise FeatureError(
            f"waveform holds {bad} non-finite sample(s) (NaN or inf); the recording "
            "cannot be analysed"
        )
    if sr != cfg.sample_rate:
        _LOGGER.warning(
            "waveform is sampled at %d Hz but mfcc.sample_rate is %d Hz; "
            "the analysis follows the actual rate",
            sr,
            cfg.sample_rate,
        )

    frame_len = max(round(sr * cfg.frame_ms / 1000.0), 1)
    if signal.size < frame_len:
        raise FeatureError(
            f"recording of {signal.size} samples ({signal.size / sr:.3f} s) is shorter than "
            f"one {cfg.frame_ms:g} ms frame ({frame_len} samples at {sr} Hz); "
            "shorten mfcc.frame_ms or drop the recording"
        )

    signal = band_limit(signal, sr, cfg.band)
    coeffs = _speechdsp_mfcc(
        signal,
        sr,
        n_mfcc=cfg.n_static,
        n_mels=cfg.n_mels,
        frame_ms=cfg.frame_ms,
        hop_ms=cfg.hop_ms,
        n_fft=n_fft_for(sr, cfg),
        preemph=cfg.preemph,
        lifter=cfg.lifter,
        append_energy=False,
    )
    if coeffs.shape[0] == 0:  # pragma: no cover - guarded by the length check above
        raise FeatureError("the front end produced no frames")
    if not np.all(np.isfinite(coeffs)):  # pragma: no cover - the input is checked above
        raise FeatureError("the front end produced non-finite coefficients")

    # Move c0 from the first to the last column and restore the HTK scaling.
    return np.hstack([coeffs[:, 1:], C0_SCALE * coeffs[:, :1]])


def extract_features(x: object, sr: int, cfg: MFCCConfig | None = None) -> np.ndarray:
    """Compute the full feature matrix for one recording.

    Parameters
    ----------
    x : array_like
        Input waveform.
    sr : int
        Sampling rate in Hz.
    cfg : MFCCConfig, optional
        Front-end settings; the defaults are used when omitted.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, cfg.n_dim)``.  With the delta expansion
        enabled the columns are laid out as
        ``[c1..cL, c0 | dc1..dcL, dc0 | ddc1..ddcL, ddc0]``.

    Raises
    ------
    FeatureError
        If the waveform cannot be analysed.

    Examples
    --------
    >>> import numpy as np
    >>> from voice_pathology_detection.config import MFCCConfig
    >>> cfg = MFCCConfig(sample_rate=8000, low_freq=None)
    >>> x = np.sin(2 * np.pi * 180 * np.arange(8000) / 8000)
    >>> extract_features(x, 8000, cfg).shape[1]
    39
    """
    cfg = MFCCConfig() if cfg is None else cfg
    static = extract_static(x, sr, cfg)
    if not cfg.use_deltas:
        return static
    return append_deltas(static, delta_width=cfg.delta_width, accel_width=cfg.accel_width)
