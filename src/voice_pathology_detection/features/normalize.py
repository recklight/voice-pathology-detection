"""Cepstral normalisation strategies.

Three quite different operations all go by "cepstral normalisation", and a
result that does not say which one it used cannot be compared with anything.
Here they are separate methods, chosen with ``normalization.method``:

``cvn_block``
    A *per-frame* operation.  Each 13-dimensional block of one frame (static,
    velocity, acceleration) is divided by the spread of its own coefficients.
    It carries no state at all, so it can never leak information between the
    training and the evaluation side, and it normalises every recording
    independently of the corpus it happens to sit in.
``cmn`` / ``cvn`` / ``cmvn``
    The textbook operation: each coefficient is normalised *along the time
    axis*.  Statistics are estimated on the training frames only and then
    applied unchanged to the evaluation frames, which is what
    :meth:`GlobalCepstralNormalizer.fit` and ``transform`` enforce.
``cmvn_per_utt``
    The same formula, but with statistics taken from each recording on its own,
    which removes the channel of that recording and is again stateless.

Every normaliser exposes the same small protocol, ``fit(X, group_sizes=None)``
and ``transform(X, group_sizes=None)``, and also satisfies the scikit-learn
transformer interface so that it can be dropped into a
:class:`sklearn.pipeline.Pipeline`.

Examples
--------
>>> import numpy as np
>>> from voice_pathology_detection.config import NormalizationConfig
>>> X = np.arange(12, dtype=float).reshape(2, 6)
>>> norm = make_normalizer(NormalizationConfig(method="cvn_block", block_size=3))
>>> out = norm.fit_transform(X)
>>> out.shape
(2, 6)

References
----------
.. [1] S. Furui, "Cepstral analysis technique for automatic speaker
       verification", *IEEE Transactions on Acoustics, Speech, and Signal
       Processing*, 29(2):254-272, 1981.
.. [2] O. Viikki and K. Laurila, "Cepstral domain segmental feature vector
       normalization for noise robust speech recognition", *Speech
       Communication*, 25(1-3):133-147, 1998.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from speechdsp import cmn, cmvn, cvn

from ..config import NormalizationConfig
from .deltas import split_blocks

_LOGGER = logging.getLogger(__name__)

#: Per-utterance formulas available, mapped to their speechdsp implementation.
_UTTERANCE_FUNCTIONS = {"cmn": cmn, "cvn": cvn, "cmvn": cmvn}


def _as_matrix(X: object) -> np.ndarray:
    """Return ``X`` as a 2-D float array, with a clear error otherwise."""
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"expected a 2-D feature matrix (n_frames, n_dim), got shape {array.shape}"
        )
    return array


def _check_group_sizes(group_sizes: object, n_frames: int) -> np.ndarray:
    """Validate that ``group_sizes`` partitions ``n_frames`` exactly."""
    sizes = np.asarray(group_sizes, dtype=np.int64).ravel()
    if sizes.size == 0:
        raise ValueError("group_sizes must not be empty")
    if np.any(sizes <= 0):
        raise ValueError("every entry of group_sizes must be positive")
    total = int(sizes.sum())
    if total != n_frames:
        raise ValueError(
            f"group_sizes sum to {total} frames but the matrix has {n_frames}; "
            "the frame-to-file bookkeeping is out of step"
        )
    return sizes


class FrameNormalizer(BaseEstimator, TransformerMixin):
    """Base class of the frame-level normalisers.

    Subclasses implement :meth:`_transform`.  The base class handles input
    validation, the optional ``group_sizes`` argument and the scikit-learn
    ``n_features_in_`` bookkeeping.
    """

    #: Whether :meth:`transform` needs to know the frame count of each recording.
    needs_groups: bool = False

    def fit(self, X: object, y: object = None, group_sizes: object = None) -> FrameNormalizer:
        """Estimate whatever statistics the strategy requires.

        Parameters
        ----------
        X : array_like
            Training frames of shape ``(n_frames, n_dim)``.
        y : ignored
            Present for scikit-learn compatibility.
        group_sizes : array_like, optional
            Frame count of every recording in ``X``.

        Returns
        -------
        FrameNormalizer
            ``self``, fitted.
        """
        array = _as_matrix(X)
        self.n_features_in_ = int(array.shape[1])
        self._fit(array, group_sizes)
        return self

    def transform(self, X: object, group_sizes: object = None) -> np.ndarray:
        """Apply the normalisation.

        Parameters
        ----------
        X : array_like
            Frames of shape ``(n_frames, n_dim)``.
        group_sizes : array_like, optional
            Frame count of every recording in ``X``; required by the
            per-utterance strategies.

        Returns
        -------
        numpy.ndarray
            Normalised copy of ``X``.

        Raises
        ------
        ValueError
            If the dimension differs from the one seen during ``fit``, or if
            ``group_sizes`` is needed and missing.
        """
        array = _as_matrix(X)
        expected = getattr(self, "n_features_in_", None)
        if expected is not None and array.shape[1] != expected:
            raise ValueError(
                f"this normaliser was fitted on {expected} feature(s) but received {array.shape[1]}"
            )
        if self.needs_groups and group_sizes is None:
            raise ValueError(
                f"{type(self).__name__} needs group_sizes, one frame count per recording"
            )
        return self._transform(array, group_sizes)

    def fit_transform(self, X: object, y: object = None, **kwargs: object) -> np.ndarray:
        """Fit on ``X`` and return the normalised matrix."""
        group_sizes = kwargs.pop("group_sizes", None)
        return self.fit(X, y, group_sizes=group_sizes).transform(X, group_sizes=group_sizes)

    # -- hooks -----------------------------------------------------------
    def _fit(self, X: np.ndarray, group_sizes: object) -> None:
        """Estimate statistics; stateless strategies do nothing."""

    def _transform(self, X: np.ndarray, group_sizes: object) -> np.ndarray:
        """Return the normalised matrix."""
        raise NotImplementedError  # pragma: no cover - abstract


class IdentityNormalizer(FrameNormalizer):
    """Pass features through unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.ones((3, 4))
    >>> bool(np.array_equal(IdentityNormalizer().fit_transform(X), X))
    True
    """

    def _transform(self, X: np.ndarray, group_sizes: object) -> np.ndarray:
        return X.copy()


class BlockVarianceNormalizer(FrameNormalizer):
    """Divide every block of every frame by the spread of its coefficients.

    For frame ``t`` and block ``b`` spanning the columns
    ``[b * block_size, (b + 1) * block_size)``, the block is divided by the
    standard deviation (or the variance) of its own ``block_size``
    coefficients.  The operation is therefore local to a single frame: it
    equalises the overall level of the static, velocity and acceleration parts
    of a vector without ever looking at another frame or another recording.

    Parameters
    ----------
    block_size : int, optional
        Width of one block, default 13.  Must divide the feature dimension.
    divisor : {'std', 'var'}, optional
        Whether to divide by the standard deviation (default) or by the
        variance.  ``std`` keeps the result dimensionless, which is why it is
        the default; ``var`` also changes the relative weight of blocks with
        different dynamic ranges.
    eps : float, optional
        Floor added to the divisor so a constant block cannot divide by zero.

    Attributes
    ----------
    n_features_in_ : int
        Feature dimension seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]])
    >>> out = BlockVarianceNormalizer(block_size=3).fit_transform(X)
    >>> bool(np.allclose(out.reshape(1, 2, 3).std(axis=2), 1.0))
    True
    """

    def __init__(self, block_size: int = 13, divisor: str = "std", eps: float = 1e-8) -> None:
        self.block_size = block_size
        self.divisor = divisor
        self.eps = eps

    def _fit(self, X: np.ndarray, group_sizes: object) -> None:
        if self.divisor not in {"std", "var"}:
            raise ValueError(f"divisor must be 'std' or 'var', got {self.divisor!r}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        split_blocks(X, self.block_size)  # validates the layout early

    def _transform(self, X: np.ndarray, group_sizes: object) -> np.ndarray:
        blocks = split_blocks(X, self.block_size)
        spread = blocks.std(axis=2, keepdims=True)
        if self.divisor == "var":
            spread = spread**2
        n_degenerate = int(np.count_nonzero(spread <= self.eps))
        if n_degenerate:
            _LOGGER.debug(
                "%d block(s) have a spread at or below eps and are only floored", n_degenerate
            )
        return (blocks / (spread + self.eps)).reshape(X.shape)


class GlobalCepstralNormalizer(FrameNormalizer):
    """Normalise each coefficient along the time axis, using training statistics.

    Cepstral mean and/or variance normalisation as found in the literature.
    The mean and standard deviation are estimated during :meth:`fit`, on the
    training frames of the current fold only, and reused unchanged on the
    evaluation frames, so no statistic of the evaluation fold reaches the model.

    Parameters
    ----------
    mode : {'cmn', 'cvn', 'cmvn'}, optional
        Whether to remove the mean, scale to unit variance, or both (default).

    Attributes
    ----------
    scaler_ : sklearn.preprocessing.StandardScaler
        The fitted scaler holding ``mean_`` and ``scale_``.
    n_features_in_ : int
        Feature dimension seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> train, test = rng.normal(size=(50, 4)), rng.normal(size=(10, 4))
    >>> norm = GlobalCepstralNormalizer().fit(train)
    >>> bool(np.allclose(norm.transform(train).mean(axis=0), 0.0, atol=1e-12))
    True
    """

    def __init__(self, mode: str = "cmvn") -> None:
        self.mode = mode

    def _fit(self, X: np.ndarray, group_sizes: object) -> None:
        if self.mode not in {"cmn", "cvn", "cmvn"}:
            raise ValueError(f"mode must be one of 'cmn', 'cvn', 'cmvn', got {self.mode!r}")
        self.scaler_ = StandardScaler(
            with_mean=self.mode in {"cmn", "cmvn"},
            with_std=self.mode in {"cvn", "cmvn"},
        ).fit(X)
        _LOGGER.debug("fitted %s statistics on %d training frame(s)", self.mode, X.shape[0])

    def _transform(self, X: np.ndarray, group_sizes: object) -> np.ndarray:
        if not hasattr(self, "scaler_"):
            raise ValueError("this normaliser must be fitted before it can transform")
        return np.asarray(self.scaler_.transform(X), dtype=np.float64)


class PerUtteranceNormalizer(FrameNormalizer):
    """Apply a cepstral normalisation to each recording on its own.

    Parameters
    ----------
    mode : {'cmn', 'cvn', 'cmvn'}, optional
        Formula applied within every recording, default ``'cmvn'``.

    Notes
    -----
    Statistics never cross a recording boundary, so this strategy is stateless
    and cannot leak anything between folds.  It does assume that each recording
    is long enough for its own mean and variance to be meaningful; a warning is
    logged for recordings of a single frame.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.vstack([np.arange(6.0).reshape(3, 2), np.arange(6.0).reshape(3, 2) * 5])
    >>> out = PerUtteranceNormalizer().fit_transform(X, group_sizes=[3, 3])
    >>> bool(np.allclose(out[:3], out[3:]))
    True
    """

    needs_groups = True

    def __init__(self, mode: str = "cmvn") -> None:
        self.mode = mode

    def _fit(self, X: np.ndarray, group_sizes: object) -> None:
        if self.mode not in _UTTERANCE_FUNCTIONS:
            raise ValueError(
                f"mode must be one of {sorted(_UTTERANCE_FUNCTIONS)}, got {self.mode!r}"
            )

    def _transform(self, X: np.ndarray, group_sizes: object) -> np.ndarray:
        sizes = _check_group_sizes(group_sizes, X.shape[0])
        if np.any(sizes < 2):
            _LOGGER.warning(
                "%d recording(s) contribute a single frame; their per-utterance "
                "statistics are degenerate",
                int(np.count_nonzero(sizes < 2)),
            )
        function = _UTTERANCE_FUNCTIONS[self.mode]
        boundaries = np.cumsum(sizes)[:-1]
        return np.vstack([function(part) for part in np.split(X, boundaries)])


def make_normalizer(cfg: NormalizationConfig | None = None) -> FrameNormalizer:
    """Build the normaliser named by a configuration.

    Parameters
    ----------
    cfg : NormalizationConfig, optional
        Normalisation settings; the defaults are used when omitted.

    Returns
    -------
    FrameNormalizer
        An unfitted normaliser.

    Raises
    ------
    ValueError
        If the method name is not recognised.

    Examples
    --------
    >>> from voice_pathology_detection.config import NormalizationConfig
    >>> type(make_normalizer(NormalizationConfig(method="none"))).__name__
    'IdentityNormalizer'
    """
    cfg = NormalizationConfig() if cfg is None else cfg
    match cfg.method:
        case "none":
            return IdentityNormalizer()
        case "cvn_block":
            return BlockVarianceNormalizer(
                block_size=cfg.block_size, divisor=cfg.divisor, eps=cfg.eps
            )
        case "cmn" | "cvn" | "cmvn":
            return GlobalCepstralNormalizer(mode=cfg.method)
        case "cmvn_per_utt":
            return PerUtteranceNormalizer(mode="cmvn")
        case _:  # pragma: no cover - the configuration validates the name
            raise ValueError(f"unsupported normalisation method: {cfg.method!r}")
