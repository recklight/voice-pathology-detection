"""The supervised frame classifier: a sigmoid multi-layer perceptron.

The network is trained by stochastic gradient descent on the cross-entropy of a
softmax output, which is what :class:`sklearn.neural_network.MLPClassifier`
implements.  The only thing added here is a guard around early stopping.

Early stopping chooses the number of epochs.  ``MLPClassifier`` holds out a
random, stratified slice of the frame matrix it is given, which in the routes
is the fitting part of a training fold, so the evaluation fold never takes part.
The slice is drawn over frames, though, and its frames come from the same
recordings as the frames the network trains on; the stopping epoch is
therefore chosen on familiar speakers and may come a little late.  This only
affects when training stops, never which data is evaluated.  With
``early_stopping: false`` nothing is held out: the same patience rule
(``n_iter_no_change`` epochs without improvement) watches the training loss
instead, and ``max_iter`` caps the number of epochs.

On very small inputs, which is what the tests and the synthetic demonstration
feed it, that slice becomes too small to use.
:class:`MLPClassifierAutoValidation` detects that case, disables early stopping
for that one fit and says so in the log, instead of failing or validating on
three frames.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = np.vstack([rng.normal(-2, 1, (60, 4)), rng.normal(2, 1, (60, 4))])
>>> y = np.array([0] * 60 + [1] * 60)
>>> model = build_mlp(
...     hidden_layer_sizes=(8,), max_iter=300, learning_rate_init=0.05,
...     early_stopping=False, random_state=0,
... ).fit(X, y)
>>> float(model.score(X, y)) > 0.9
True

References
----------
.. [1] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning
       representations by back-propagating errors", *Nature*, 323:533-536, 1986.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier

_LOGGER = logging.getLogger(__name__)

#: Default topology: five hidden layers of two hundred units.
DEFAULT_HIDDEN_LAYERS: tuple[int, ...] = (200, 200, 200, 200, 200)

#: Smallest usable held-out validation set, in samples.
MIN_VALIDATION_SAMPLES: int = 10


class MLPClassifierAutoValidation(MLPClassifier):
    """An ``MLPClassifier`` that disables early stopping it cannot honour.

    Behaves exactly like its base class, except that a fit whose internal
    validation split would hold fewer than :data:`MIN_VALIDATION_SAMPLES`
    samples, or would leave a class unrepresented, runs without early stopping
    and logs a warning.  The constructor parameter itself is restored
    afterwards, so the estimator keeps the configuration it was given.
    """

    def _validation_is_feasible(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Whether an internal validation split would be large enough to use."""
        n_samples = int(np.asarray(X).shape[0])
        held_out = int(np.floor(n_samples * float(self.validation_fraction)))
        _, counts = np.unique(np.asarray(y), return_counts=True)
        smallest_class_share = int(np.floor(counts.min() * float(self.validation_fraction)))
        return held_out >= MIN_VALIDATION_SAMPLES and smallest_class_share >= 1

    def fit(self, X: object, y: object) -> MLPClassifierAutoValidation:
        """Fit the network, relaxing early stopping when the data is too small.

        Parameters
        ----------
        X : array_like
            Training features of shape ``(n_samples, n_features)``.
        y : array_like
            Training labels.

        Returns
        -------
        MLPClassifierAutoValidation
            ``self``, fitted.
        """
        requested = self.early_stopping
        try:
            if requested and not self._validation_is_feasible(np.asarray(X), np.asarray(y)):
                _LOGGER.warning(
                    "%d training sample(s) are too few for a %.0f%% validation split "
                    "that holds every class; early stopping is off for this fit",
                    np.asarray(X).shape[0],
                    100.0 * float(self.validation_fraction),
                )
                self.early_stopping = False
            return super().fit(X, y)
        finally:
            self.early_stopping = requested


def build_mlp(
    hidden_layer_sizes: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
    *,
    activation: str = "logistic",
    solver: str = "sgd",
    learning_rate_init: float = 0.001,
    momentum: float = 0.5,
    batch_size: int | str = 90,
    max_iter: int = 30,
    alpha: float = 1e-4,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 5,
    shuffle: bool = True,
    random_state: int = 0,
    **extra: object,
) -> MLPClassifierAutoValidation:
    """Construct the frame classifier.

    Parameters
    ----------
    hidden_layer_sizes : sequence of int, optional
        Units per hidden layer, default five layers of two hundred.
    activation : str, optional
        Hidden activation, default ``"logistic"``.
    solver : str, optional
        Optimiser, default ``"sgd"``.
    learning_rate_init : float, optional
        Initial step size, default 0.001.
    momentum : float, optional
        Momentum of the SGD update, default 0.5.
    batch_size : int or str, optional
        Mini-batch size, default 90.
    max_iter : int, optional
        Maximum epochs, default 30.
    alpha : float, optional
        L2 penalty, default 1e-4.
    early_stopping : bool, optional
        Hold out part of the training data to choose the epoch, default
        ``True``.
    validation_fraction : float, optional
        Share held out for that purpose, default 0.1.
    n_iter_no_change : int, optional
        Patience of the early stopping rule, default 5.
    shuffle : bool, optional
        Shuffle the samples every epoch, default ``True``.
    random_state : int, optional
        Seed of the initialisation and the shuffling, default 0.
    **extra
        Any further ``MLPClassifier`` keyword.

    Returns
    -------
    MLPClassifierAutoValidation
        The unfitted estimator.

    Raises
    ------
    ValueError
        If the topology is empty or contains a non-positive layer.

    Examples
    --------
    >>> build_mlp(hidden_layer_sizes=(16, 8)).hidden_layer_sizes
    (16, 8)
    """
    sizes = tuple(int(size) for size in hidden_layer_sizes)
    if not sizes:
        raise ValueError("hidden_layer_sizes must name at least one layer")
    if any(size < 1 for size in sizes):
        raise ValueError(f"every hidden layer must have at least one unit, got {sizes}")

    return MLPClassifierAutoValidation(
        hidden_layer_sizes=sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        momentum=momentum,
        batch_size=batch_size,
        max_iter=max_iter,
        alpha=alpha,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        shuffle=shuffle,
        random_state=random_state,
        **extra,
    )
