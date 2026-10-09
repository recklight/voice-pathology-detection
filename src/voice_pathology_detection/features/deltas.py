"""Dynamic (delta and acceleration) cepstral coefficients.

A static cepstrum describes one 30 ms slice on its own, while tremor, roughness
and breathiness that waxes and wanes show up in how that slice *changes*.
Regression differences are the usual way to put that into the feature vector.

The regression itself is :func:`speechdsp.delta`.  Two choices around it are
specific to this project: the acceleration is a difference of the velocity
rather than a second-order difference of the static coefficients, and the
velocity and the acceleration may use different window widths.

Examples
--------
>>> import numpy as np
>>> static = np.arange(10, dtype=float)[:, None]
>>> velocity = regression_delta(static, width=5)
>>> bool(np.allclose(velocity[3:-3], 1.0))
True

References
----------
.. [1] S. Furui, "Speaker-independent isolated word recognition using dynamic
       features of speech spectrum", *IEEE Transactions on Acoustics, Speech,
       and Signal Processing*, 34(1):52-59, 1986.
.. [2] S. Young, G. Evermann, M. Gales, et al., *The HTK Book (for HTK Version
       3.4)*, Cambridge University Engineering Department, 2006, section 5.9.
"""

from __future__ import annotations

import logging

import numpy as np
from speechdsp import delta as _speechdsp_delta

_LOGGER = logging.getLogger(__name__)


def _check_width(name: str, width: int) -> int:
    """Validate a regression window width and return it as an int."""
    width = int(width)
    if width < 3 or width % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= 3, got {width}")
    return width


def regression_delta(feats: object, width: int = 7) -> np.ndarray:
    """First-order regression difference over a symmetric window.

    Computes ``d[t] = sum_n n * (c[t+n] - c[t-n]) / (2 * sum_n n^2)`` for
    ``n = 1 .. (width - 1) / 2``, padding the sequence by edge replication so
    the frame count is preserved.

    Parameters
    ----------
    feats : array_like
        Feature matrix of shape ``(n_frames, n_dim)``.
    width : int, optional
        Regression window length in frames; odd and at least 3, default 7
        (a half-window of three frames).

    Returns
    -------
    numpy.ndarray
        Differences of the same shape as ``feats``.

    Raises
    ------
    ValueError
        If ``feats`` is not two-dimensional or ``width`` is even or too small.

    Examples
    --------
    >>> import numpy as np
    >>> regression_delta(np.zeros((4, 2))).shape
    (4, 2)
    """
    array = np.asarray(feats, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"feats must be a 2-D array of shape (n_frames, n_dim), got shape {array.shape}"
        )
    return _speechdsp_delta(array, width=_check_width("width", width))


def append_deltas(
    static: object,
    delta_width: int = 7,
    accel_width: int = 5,
) -> np.ndarray:
    """Expand static coefficients with velocity and acceleration blocks.

    Parameters
    ----------
    static : array_like
        Static feature matrix of shape ``(n_frames, n_dim)``.
    delta_width : int, optional
        Regression window of the velocity, default 7.
    accel_width : int, optional
        Regression window of the acceleration, default 5.  The acceleration is
        the regression difference *of the velocity*, which is smoother than a
        direct second-order difference of the static coefficients.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, 3 * n_dim)`` laid out as
        ``[static | velocity | acceleration]``.

    Raises
    ------
    ValueError
        If the input is not two-dimensional or a width is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> append_deltas(np.zeros((6, 13))).shape
    (6, 39)
    """
    array = np.asarray(static, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"static must be a 2-D array of shape (n_frames, n_dim), got shape {array.shape}"
        )
    delta_width = _check_width("delta_width", delta_width)
    accel_width = _check_width("accel_width", accel_width)

    if array.shape[0] < max(delta_width, accel_width):
        _LOGGER.debug(
            "only %d frame(s) for regression windows of %d/%d; edge padding dominates",
            array.shape[0],
            delta_width,
            accel_width,
        )

    velocity = regression_delta(array, width=delta_width)
    acceleration = regression_delta(velocity, width=accel_width)
    return np.hstack([array, velocity, acceleration])


def split_blocks(features: object, block_size: int) -> np.ndarray:
    """View a feature matrix as ``(n_frames, n_blocks, block_size)``.

    The static, velocity and acceleration parts of a vector are contiguous
    blocks of equal width, and block-wise normalisation is much clearer
    written against that three-dimensional view than against column slices.

    Parameters
    ----------
    features : array_like
        Feature matrix of shape ``(n_frames, n_dim)``.
    block_size : int
        Width of one block; must divide ``n_dim`` exactly.

    Returns
    -------
    numpy.ndarray
        Reshaped view of shape ``(n_frames, n_dim // block_size, block_size)``.

    Raises
    ------
    ValueError
        If the dimension is not a multiple of ``block_size``, which usually
        means the front end and the normaliser disagree about the layout.

    Examples
    --------
    >>> import numpy as np
    >>> split_blocks(np.zeros((4, 39)), 13).shape
    (4, 3, 13)
    """
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"features must be a 2-D array of shape (n_frames, n_dim), got shape {array.shape}"
        )
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    n_dim = array.shape[1]
    if n_dim % block_size:
        raise ValueError(
            f"feature dimension {n_dim} is not a multiple of block_size={block_size}; "
            "check that normalization.block_size matches the front end layout"
        )
    return array.reshape(array.shape[0], n_dim // block_size, block_size)
