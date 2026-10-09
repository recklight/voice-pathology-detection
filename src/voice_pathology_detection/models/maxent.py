"""Maximum-entropy parameter updates solved by Newton's method.

An alternative to the plain gradient step of contrastive divergence.  Instead
of moving the parameters along the difference between the data and model
statistics, the update treats the parameters as Lagrange multipliers of a
maximum-entropy problem with a Gaussian prior of variance ``var`` on each one,
and solves the resulting stationarity condition exactly:

.. math::

    f(\\gamma) = F \\, \\sigma^2 \\exp(\\gamma \\, S) + \\gamma + \\lambda
                 - \\mu \\, \\sigma^2 = 0,

where :math:`F` is the model expectation of a feature, :math:`\\mu` the
expectation under the data, :math:`\\lambda` the current parameter,
:math:`\\sigma^2` the prior variance and :math:`S = \\sum_i F_i` a scalar
coupling term.  The derivative

.. math::

    f'(\\gamma) = F \\, S \\, \\sigma^2 \\exp(\\gamma \\, S) + 1

is available in closed form, so Newton's method converges in a handful of
iterations, and all parameters are solved together in one vectorised loop.

The effect of the prior is a pull towards zero that grows with the size of the
update, which damps the large steps contrastive divergence can take on the
first epochs.

Examples
--------
>>> import numpy as np
>>> gamma = maxent_gamma(
...     np.array([0.3, 0.7]), np.array([0.4, 0.5]), np.array([0.1, -0.2])
... )
>>> bool(np.all(np.abs(maxent_residual(gamma, np.array([0.3, 0.7]),
...     np.array([0.4, 0.5]), np.array([0.1, -0.2]))) < 1e-6))
True

References
----------
.. [1] E. T. Jaynes, "Information theory and statistical mechanics", *Physical
       Review*, 106(4):620-630, 1957.  The maximum-entropy principle.
.. [2] S. F. Chen and R. Rosenfeld, "A survey of smoothing techniques for ME
       models", *IEEE Transactions on Speech and Audio Processing*,
       8(1):37-50, 2000.  The Gaussian-prior update whose stationarity
       condition is solved here.
.. [3] G. E. Hinton, "Training products of experts by minimizing contrastive
       divergence", *Neural Computation*, 14(8):1771-1800, 2002.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_LOGGER = logging.getLogger(__name__)

#: Largest exponent argument evaluated, to keep ``exp`` from overflowing.
_EXP_CLIP = 60.0


@dataclass(frozen=True, slots=True)
class MaxEntSolution:
    """Outcome of a maximum-entropy solve.

    Attributes
    ----------
    gamma : numpy.ndarray
        The solved multipliers, one per parameter.
    residual : float
        Sum of the absolute residuals at the solution.
    n_iter : int
        Newton iterations actually performed.
    converged : bool
        Whether the residual fell below the tolerance.
    """

    gamma: np.ndarray
    residual: float
    n_iter: int
    converged: bool


def maxent_residual(
    gamma: object,
    f_model: object,
    mu_data: object,
    lam: object,
    var: float = 0.1,
    sum_f: float | None = None,
) -> np.ndarray:
    """Evaluate the stationarity condition of the maximum-entropy problem.

    Parameters
    ----------
    gamma : array_like
        Current multipliers.
    f_model : array_like
        Feature expectations under the model.
    mu_data : array_like
        Feature expectations under the data.
    lam : array_like
        Current parameter values.
    var : float, optional
        Prior variance, default 0.1.
    sum_f : float, optional
        The scalar coupling term; defaults to ``f_model.sum()``.

    Returns
    -------
    numpy.ndarray
        The residual ``f(gamma)``, elementwise.

    Raises
    ------
    ValueError
        If the arrays do not broadcast to one shape or ``var`` is not positive.
    """
    if var <= 0:
        raise ValueError(f"var must be positive, got {var}")
    g = np.asarray(gamma, dtype=np.float64)
    f = np.asarray(f_model, dtype=np.float64)
    mu = np.asarray(mu_data, dtype=np.float64)
    la = np.asarray(lam, dtype=np.float64)
    if not (g.shape == f.shape == mu.shape == la.shape):
        raise ValueError(
            "gamma, f_model, mu_data and lam must share one shape, got "
            f"{g.shape}, {f.shape}, {mu.shape}, {la.shape}"
        )
    total = float(f.sum()) if sum_f is None else float(sum_f)
    exponent = np.clip(g * total, -_EXP_CLIP, _EXP_CLIP)
    return f * var * np.exp(exponent) + g + la - mu * var


def maxent_gamma(
    f_model: object,
    mu_data: object,
    lam: object,
    var: float = 0.1,
    *,
    gamma0: object = None,
    tol: float = 1e-3,
    max_iter: int = 100,
    damping: float = 1.0,
) -> np.ndarray:
    """Solve the maximum-entropy condition for every parameter at once.

    Parameters
    ----------
    f_model : array_like
        Feature expectations under the model, typically from a short Gibbs
        chain.
    mu_data : array_like
        Feature expectations under the data.
    lam : array_like
        Current parameter values, which act as the multipliers of the previous
        iteration.
    var : float, optional
        Prior variance, default 0.1.  Smaller values pull the solution harder
        towards zero.
    gamma0 : array_like, optional
        Starting point; defaults to zeros, which is the solution when the data
        and model statistics agree and the parameters vanish.
    tol : float, optional
        Convergence threshold on the summed absolute residual, default 1e-3.
    max_iter : int, optional
        Maximum Newton iterations, default 100.
    damping : float, optional
        Step scaling in ``(0, 1]``, default 1 (an undamped Newton step).

    Returns
    -------
    numpy.ndarray
        The solved multipliers, shaped like ``f_model``.

    Raises
    ------
    ValueError
        If the shapes disagree, or ``var``, ``tol``, ``max_iter`` or ``damping``
        are out of range.

    Notes
    -----
    Non-convergence is logged rather than raised: the last iterate is still a
    descent direction, and aborting training because one mini-batch was slow to
    converge would be worse than taking a slightly inexact step.

    Examples
    --------
    >>> import numpy as np
    >>> gamma = maxent_gamma(np.full(4, 0.5), np.full(4, 0.5), np.zeros(4))
    >>> bool(np.all(np.abs(gamma) < 1e-6))
    True
    """
    return solve_maxent(
        f_model,
        mu_data,
        lam,
        var,
        gamma0=gamma0,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    ).gamma


def solve_maxent(
    f_model: object,
    mu_data: object,
    lam: object,
    var: float = 0.1,
    *,
    gamma0: object = None,
    tol: float = 1e-3,
    max_iter: int = 100,
    damping: float = 1.0,
) -> MaxEntSolution:
    """Like :func:`maxent_gamma`, but reporting the convergence diagnostics.

    Parameters
    ----------
    f_model, mu_data, lam : array_like
        See :func:`maxent_gamma`.
    var : float, optional
        Prior variance, default 0.1.
    gamma0 : array_like, optional
        Starting point, default zeros.
    tol : float, optional
        Convergence threshold on the summed absolute residual.
    max_iter : int, optional
        Maximum Newton iterations.
    damping : float, optional
        Step scaling in ``(0, 1]``.

    Returns
    -------
    MaxEntSolution
        The multipliers together with the final residual and iteration count.
    """
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    if not 0.0 < damping <= 1.0:
        raise ValueError(f"damping must lie in (0, 1], got {damping}")

    f = np.asarray(f_model, dtype=np.float64)
    mu = np.asarray(mu_data, dtype=np.float64)
    la = np.asarray(lam, dtype=np.float64)
    gamma = np.zeros_like(f) if gamma0 is None else np.array(gamma0, dtype=np.float64, copy=True)
    total = float(f.sum())

    residual = maxent_residual(gamma, f, mu, la, var=var, sum_f=total)
    converged = False
    iteration = 0
    while iteration < max_iter:
        if float(np.abs(residual).sum()) < tol:
            converged = True
            break
        iteration += 1
        exponent = np.clip(gamma * total, -_EXP_CLIP, _EXP_CLIP)
        derivative = f * total * var * np.exp(exponent) + 1.0
        # The derivative is 1 plus a term that vanishes with f, so it stays far
        # from zero for any realistic statistic; the guard covers the rest.
        derivative = np.where(np.abs(derivative) < 1e-12, np.sign(derivative) + 1e-12, derivative)
        gamma = gamma - damping * residual / derivative
        residual = maxent_residual(gamma, f, mu, la, var=var, sum_f=total)

    final = float(np.abs(residual).sum())
    converged = converged or final < tol
    if not converged:
        _LOGGER.debug(
            "maximum-entropy solve stopped at residual %.3e after %d iteration(s)",
            final,
            iteration,
        )
    return MaxEntSolution(gamma=gamma, residual=final, n_iter=iteration, converged=converged)


def sigmoid(x: object) -> np.ndarray:
    """Logistic function, evaluated without overflow.

    Parameters
    ----------
    x : array_like
        Input values.

    Returns
    -------
    numpy.ndarray
        ``1 / (1 + exp(-x))``, computed branch-wise so that large negative
        inputs do not overflow the exponential.

    Examples
    --------
    >>> float(sigmoid(0.0))
    0.5
    >>> bool(sigmoid(-1000.0) < 1e-12)
    True
    """
    array = np.asarray(x, dtype=np.float64)
    out = np.empty_like(array)
    positive = array >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_negative = np.exp(array[~positive])
    out[~positive] = exp_negative / (1.0 + exp_negative)
    return out
