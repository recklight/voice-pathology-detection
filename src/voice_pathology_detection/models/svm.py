"""Support vector baselines.

A neural network is only worth its complexity if it beats a shallow model on
the same features and the same folds, so both support vector machines
sit in the model registry and the comparison is a change of ``model.name``.

Both variants standardise the features first.  The margin of a support vector
machine depends on the scale of each feature, and the cepstral coefficients of
the static and the acceleration blocks differ in range by an order of
magnitude.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = np.vstack([rng.normal(-2, 1, (30, 4)), rng.normal(2, 1, (30, 4))])
>>> y = np.array([0] * 30 + [1] * 30)
>>> float(build_svm().fit(X, y).score(X, y)) > 0.9
True

References
----------
.. [1] C. Cortes and V. Vapnik, "Support-vector networks", *Machine Learning*,
       20(3):273-297, 1995.
"""

from __future__ import annotations

import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

_LOGGER = logging.getLogger(__name__)


def build_svm(
    kernel: str = "rbf",
    *,
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: str | dict | None = "balanced",
    probability: bool = True,
    max_iter: int = -1,
    random_state: int = 0,
    **extra: object,
) -> Pipeline:
    """Build a standardising support vector classifier.

    Parameters
    ----------
    kernel : str, optional
        Kernel of :class:`sklearn.svm.SVC`, default ``"rbf"``.
    C : float, optional
        Regularisation strength, default 1.
    gamma : str or float, optional
        Kernel coefficient, default ``"scale"``.
    class_weight : str, dict or None, optional
        Default ``"balanced"``, which compensates the 1:3 class imbalance of
        this task instead of letting the majority class dominate the margin.
    probability : bool, optional
        Fit the Platt scaling needed for ``predict_proba``, default ``True``.
        Required by the probability-averaging aggregation rule, at the cost of
        an internal cross-validation during ``fit``.
    max_iter : int, optional
        Iteration cap of the solver, default ``-1`` (no limit).
    random_state : int, optional
        Seed of the probability calibration, default 0.
    **extra
        Further :class:`~sklearn.svm.SVC` keywords.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Scaler followed by the classifier.

    Raises
    ------
    ValueError
        If ``C`` is not positive.

    Examples
    --------
    >>> build_svm(kernel="linear").named_steps["svc"].kernel
    'linear'
    """
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    class_weight=class_weight,
                    probability=probability,
                    max_iter=max_iter,
                    random_state=random_state,
                    **extra,
                ),
            ),
        ]
    )


def build_linear_svm(
    *,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 5000,
    random_state: int = 0,
    **extra: object,
) -> Pipeline:
    """Build a standardising linear support vector classifier.

    Much faster than the kernel version on the hundreds of thousands of frames
    a full corpus produces, and a useful check on whether a non-linear boundary
    is buying anything.

    Parameters
    ----------
    C : float, optional
        Regularisation strength, default 1.
    class_weight : str, dict or None, optional
        Default ``"balanced"``.
    max_iter : int, optional
        Iteration cap, default 5000.
    random_state : int, optional
        Seed, default 0.
    **extra
        Further :class:`~sklearn.svm.LinearSVC` keywords.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Scaler followed by the classifier.

    Notes
    -----
    :class:`~sklearn.svm.LinearSVC` has no ``predict_proba``; with the
    probability-averaging aggregation rule, use :func:`build_svm` instead.
    """
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                LinearSVC(
                    C=C,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=random_state,
                    **extra,
                ),
            ),
        ]
    )
