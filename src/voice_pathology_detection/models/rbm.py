"""Restricted Boltzmann machines, trained by contrastive divergence.

A restricted Boltzmann machine is the building block of a deep belief network:
one layer of visible units, one layer of hidden units, no connections inside
either layer.  Trained without labels, it learns a hidden representation that
reconstructs its input.  The layer-wise pre-training in
:mod:`voice_pathology_detection.models.dbn` stacks several of them to give the
supervised network its starting point.

Two update rules are available.  ``cd`` is the usual contrastive divergence
gradient step; ``maxent`` solves the maximum-entropy stationarity condition of
:mod:`voice_pathology_detection.models.maxent` instead, which damps large
updates through a Gaussian prior on the parameters.

Visible units are Bernoulli, so inputs must lie in ``[0, 1]``; the callers in
this package scale cepstral features with a
:class:`sklearn.preprocessing.MinMaxScaler` before pre-training.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = rng.random((40, 6))
>>> rbm = BernoulliRBM(n_hidden=4, n_epochs=2, seed=0).fit(X)
>>> rbm.transform(X).shape
(40, 4)

References
----------
.. [1] G. E. Hinton, S. Osindero, and Y.-W. Teh, "A fast learning algorithm for
       deep belief nets", *Neural Computation*, 18(7):1527-1554, 2006.
.. [2] G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of
       data with neural networks", *Science*, 313(5786):504-507, 2006.
.. [3] G. E. Hinton, "A practical guide to training restricted Boltzmann
       machines", UTML TR 2010-003, University of Toronto, 2010.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .maxent import sigmoid, solve_maxent

_LOGGER = logging.getLogger(__name__)

#: Update rules accepted by :class:`BernoulliRBM`.
UPDATE_RULES: tuple[str, ...] = ("cd", "maxent")


class BernoulliRBM(BaseEstimator, TransformerMixin):
    """A Bernoulli-Bernoulli restricted Boltzmann machine.

    Parameters
    ----------
    n_hidden : int, optional
        Number of hidden units, default 64.
    learning_rate : float, optional
        Step size of the parameter update, default 0.05.
    n_epochs : int, optional
        Passes over the training data, default 10.
    batch_size : int, optional
        Mini-batch size, default 64.
    cd_steps : int, optional
        Gibbs sweeps in the negative phase, default 1 (standard CD-1).
    momentum : float, optional
        Momentum applied to the update, default 0.5.
    weight_decay : float, optional
        L2 penalty on the weights, default 0.
    update : {'cd', 'maxent'}, optional
        Update rule, default ``'cd'``.
    maxent_var : float, optional
        Prior variance of the maximum-entropy rule, default 0.1.
    maxent_tol : float, optional
        Convergence threshold of the Newton solve, default 1e-3.
    maxent_max_iter : int, optional
        Maximum Newton iterations per update, default 20.
    seed : int, optional
        Seed of the weight initialisation and the Gibbs sampling, default 0.
    verbose : bool, optional
        Log the reconstruction error after every epoch, default ``False``.

    Attributes
    ----------
    weights_ : numpy.ndarray
        Weight matrix of shape ``(n_visible, n_hidden)``.
    visible_bias_ : numpy.ndarray
        Visible bias of shape ``(n_visible,)``.
    hidden_bias_ : numpy.ndarray
        Hidden bias of shape ``(n_hidden,)``.
    reconstruction_error_ : list of float
        Mean squared reconstruction error after each epoch.
    n_features_in_ : int
        Number of visible units.

    Notes
    -----
    Weights are initialised from a zero-mean Gaussian with standard deviation
    0.01 and the hidden biases from zero, as in [3]; the visible biases also
    start at zero (the guide suggests ``log[p_i / (1 - p_i)]``).  The hidden
    states driven by the data are sampled as binary values to produce the
    reconstruction, which keeps each hidden unit to one bit and acts as a
    regulariser [3]; the reconstruction and all the statistics use
    probabilities, which lowers the variance of the gradient.
    """

    def __init__(
        self,
        n_hidden: int = 64,
        learning_rate: float = 0.05,
        n_epochs: int = 10,
        batch_size: int = 64,
        cd_steps: int = 1,
        momentum: float = 0.5,
        weight_decay: float = 0.0,
        update: str = "cd",
        maxent_var: float = 0.1,
        maxent_tol: float = 1e-3,
        maxent_max_iter: int = 20,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cd_steps = cd_steps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.update = update
        self.maxent_var = maxent_var
        self.maxent_tol = maxent_tol
        self.maxent_max_iter = maxent_max_iter
        self.seed = seed
        self.verbose = verbose

    # -- validation ------------------------------------------------------
    def _check_params(self) -> None:
        """Validate the hyper-parameters before any work is done."""
        if self.n_hidden < 1:
            raise ValueError(f"n_hidden must be positive, got {self.n_hidden}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.cd_steps < 1:
            raise ValueError(f"cd_steps must be at least 1, got {self.cd_steps}")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must lie in [0, 1), got {self.momentum}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must not be negative, got {self.weight_decay}")
        if self.update not in UPDATE_RULES:
            raise ValueError(f"update must be one of {UPDATE_RULES}, got {self.update!r}")

    @staticmethod
    def _check_input(X: object) -> np.ndarray:
        """Validate the visible data and return it as a float matrix."""
        array = np.asarray(X, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_visible), got shape {array.shape}")
        if array.shape[0] == 0:
            raise ValueError("cannot train on an empty matrix")
        if not np.all(np.isfinite(array)):
            raise ValueError("X contains non-finite values")
        low, high = float(array.min()), float(array.max())
        if low < -1e-9 or high > 1.0 + 1e-9:
            raise ValueError(
                f"Bernoulli visible units need inputs in [0, 1], got the range "
                f"[{low:.3f}, {high:.3f}]; scale the features first, for example with "
                "sklearn.preprocessing.MinMaxScaler"
            )
        return np.clip(array, 0.0, 1.0)

    # -- inference -------------------------------------------------------
    def hidden_probabilities(self, visible: object) -> np.ndarray:
        """Posterior probability of the hidden units.

        Parameters
        ----------
        visible : array_like
            Visible activations of shape ``(n_samples, n_visible)``.

        Returns
        -------
        numpy.ndarray
            Probabilities of shape ``(n_samples, n_hidden)``.
        """
        array = np.asarray(visible, dtype=np.float64)
        return sigmoid(array @ self.weights_ + self.hidden_bias_)

    def visible_probabilities(self, hidden: object) -> np.ndarray:
        """Posterior probability of the visible units.

        Parameters
        ----------
        hidden : array_like
            Hidden activations of shape ``(n_samples, n_hidden)``.

        Returns
        -------
        numpy.ndarray
            Probabilities of shape ``(n_samples, n_visible)``.
        """
        array = np.asarray(hidden, dtype=np.float64)
        return sigmoid(array @ self.weights_.T + self.visible_bias_)

    def gibbs(
        self, visible: np.ndarray, rng: np.random.Generator, steps: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run a short Gibbs chain starting from ``visible``.

        Parameters
        ----------
        visible : numpy.ndarray
            Starting visible activations.
        rng : numpy.random.Generator
            Source of the Bernoulli samples.
        steps : int, optional
            Number of full sweeps, default 1.

        Returns
        -------
        visible_out : numpy.ndarray
            Visible probabilities at the end of the chain.
        hidden_out : numpy.ndarray
            Hidden probabilities given those visible units.
        """
        visible_state = visible
        hidden_probability = self.hidden_probabilities(visible_state)
        for _ in range(steps):
            hidden_sample = (rng.random(hidden_probability.shape) < hidden_probability).astype(
                np.float64
            )
            visible_state = self.visible_probabilities(hidden_sample)
            hidden_probability = self.hidden_probabilities(visible_state)
        return visible_state, hidden_probability

    # -- training --------------------------------------------------------
    def _apply_update(
        self,
        gradient: dict[str, np.ndarray],
        velocity: dict[str, np.ndarray],
        statistics: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Move the parameters according to the configured update rule."""
        if self.update == "cd":
            for name, parameter in self._parameters().items():
                velocity[name] = (
                    self.momentum * velocity[name] + self.learning_rate * (gradient[name])
                )
                parameter += velocity[name]
            self.weights_ -= self.learning_rate * self.weight_decay * self.weights_
            return

        # Maximum-entropy rule: solve for the multipliers, then step along them.
        for name, parameter in self._parameters().items():
            model_stat, data_stat = statistics[name]
            solution = solve_maxent(
                model_stat.ravel(),
                data_stat.ravel(),
                parameter.ravel(),
                var=self.maxent_var,
                tol=self.maxent_tol,
                max_iter=self.maxent_max_iter,
            )
            step = solution.gamma.reshape(parameter.shape)
            velocity[name] = self.momentum * velocity[name] + self.learning_rate * step
            parameter += velocity[name]

    def _parameters(self) -> dict[str, np.ndarray]:
        """The three parameter arrays, by name, as mutable views."""
        return {
            "weights": self.weights_,
            "visible_bias": self.visible_bias_,
            "hidden_bias": self.hidden_bias_,
        }

    def fit(self, X: object, y: object = None) -> BernoulliRBM:
        """Train the machine by contrastive divergence.

        Parameters
        ----------
        X : array_like
            Visible data of shape ``(n_samples, n_visible)``, values in
            ``[0, 1]``.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        BernoulliRBM
            ``self``, fitted.

        Raises
        ------
        ValueError
            If a hyper-parameter is out of range, or the data is empty, not
            two-dimensional, non-finite or outside ``[0, 1]``.
        """
        self._check_params()
        data = self._check_input(X)
        n_samples, n_visible = data.shape
        rng = np.random.default_rng(self.seed)

        self.n_features_in_ = int(n_visible)
        self.weights_ = rng.normal(0.0, 0.01, size=(n_visible, self.n_hidden))
        self.visible_bias_ = np.zeros(n_visible)
        self.hidden_bias_ = np.zeros(self.n_hidden)
        self.reconstruction_error_ = []

        velocity = {name: np.zeros_like(p) for name, p in self._parameters().items()}
        batch_size = min(self.batch_size, n_samples)

        for epoch in range(self.n_epochs):
            order = rng.permutation(n_samples)
            squared_error = 0.0
            for start in range(0, n_samples, batch_size):
                batch = data[order[start : start + batch_size]]
                if batch.shape[0] == 0:  # pragma: no cover - defensive
                    continue

                positive_hidden = self.hidden_probabilities(batch)
                negative_visible, negative_hidden = self.gibbs(batch, rng, steps=self.cd_steps)
                scale = 1.0 / batch.shape[0]

                data_weights = batch.T @ positive_hidden * scale
                model_weights = negative_visible.T @ negative_hidden * scale
                data_visible = batch.mean(axis=0)
                model_visible = negative_visible.mean(axis=0)
                data_hidden = positive_hidden.mean(axis=0)
                model_hidden = negative_hidden.mean(axis=0)

                gradient = {
                    "weights": data_weights - model_weights,
                    "visible_bias": data_visible - model_visible,
                    "hidden_bias": data_hidden - model_hidden,
                }
                statistics = {
                    "weights": (model_weights, data_weights),
                    "visible_bias": (model_visible, data_visible),
                    "hidden_bias": (model_hidden, data_hidden),
                }
                self._apply_update(gradient, velocity, statistics)
                squared_error += float(np.sum((batch - negative_visible) ** 2))

            error = squared_error / float(data.size)
            self.reconstruction_error_.append(error)
            if self.verbose:
                _LOGGER.info(
                    "RBM epoch %d/%d: reconstruction error %.5f",
                    epoch + 1,
                    self.n_epochs,
                    error,
                )
        _LOGGER.debug(
            "trained an RBM with %d visible and %d hidden unit(s); final error %.5f",
            n_visible,
            self.n_hidden,
            self.reconstruction_error_[-1],
        )
        return self

    def transform(self, X: object) -> np.ndarray:
        """Project data onto the hidden layer.

        Parameters
        ----------
        X : array_like
            Visible data of shape ``(n_samples, n_visible)``.

        Returns
        -------
        numpy.ndarray
            Hidden probabilities of shape ``(n_samples, n_hidden)``.

        Raises
        ------
        ValueError
            If the machine is unfitted or the dimension does not match.
        """
        if not hasattr(self, "weights_"):
            raise ValueError("this RBM must be fitted before it can transform")
        array = np.asarray(X, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected a matrix with {self.n_features_in_} column(s), got shape {array.shape}"
            )
        return self.hidden_probabilities(array)

    def reconstruct(self, X: object) -> np.ndarray:
        """Pass data through the hidden layer and back.

        Parameters
        ----------
        X : array_like
            Visible data of shape ``(n_samples, n_visible)``.

        Returns
        -------
        numpy.ndarray
            Reconstructed visible probabilities of the same shape.
        """
        return self.visible_probabilities(self.transform(X))
