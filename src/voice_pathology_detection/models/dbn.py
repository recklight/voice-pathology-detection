"""Deep belief network: unsupervised pre-training, then supervised fine-tuning.

The classic recipe of Hinton, Osindero and Teh: train a stack of restricted
Boltzmann machines one layer at a time on unlabelled features, each taking the
hidden activations of the one below as its input, then unfold the stack into a
feed-forward network, add a softmax output layer and fine-tune the whole thing
with back-propagation.

What distinguishes a DBN from a plain multi-layer perceptron is the
pre-training, so :class:`DBNClassifier` runs it on the training features before
fine-tuning.  ``pretrain=False`` keeps the same topology and seed but starts
from scikit-learn's random (Glorot) initialisation [3], which makes the two
directly comparable.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = np.vstack([rng.normal(-2, 1, (60, 5)), rng.normal(2, 1, (60, 5))])
>>> y = np.array([0] * 60 + [1] * 60)
>>> model = DBNClassifier(hidden_layer_sizes=(8,), pretrain_epochs=2, max_iter=40).fit(X, y)
>>> model.predict_proba(X).shape
(120, 2)

References
----------
.. [1] G. E. Hinton, S. Osindero, and Y.-W. Teh, "A fast learning algorithm for
       deep belief nets", *Neural Computation*, 18(7):1527-1554, 2006.
.. [2] G. E. Hinton and R. R. Salakhutdinov, "Reducing the dimensionality of
       data with neural networks", *Science*, 313(5786):504-507, 2006.
.. [3] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep
       feedforward neural networks", *AISTATS*, pp. 249-256, 2010.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from .mlp import DEFAULT_HIDDEN_LAYERS, MLPClassifierAutoValidation
from .rbm import BernoulliRBM

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LayerWeights:
    """Weights of one pre-trained layer.

    Attributes
    ----------
    weights : numpy.ndarray
        Matrix of shape ``(n_in, n_out)``, in the orientation
        :class:`sklearn.neural_network.MLPClassifier` expects.
    bias : numpy.ndarray
        Bias vector of shape ``(n_out,)``.
    reconstruction_error : float
        Reconstruction error of the layer after its last pre-training epoch.
    """

    weights: np.ndarray
    bias: np.ndarray
    reconstruction_error: float

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the weight matrix."""
        return int(self.weights.shape[0]), int(self.weights.shape[1])


def pretrain_dbn(
    X: object,
    hidden_layer_sizes: Sequence[int],
    *,
    n_epochs: int = 10,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    cd_steps: int = 1,
    momentum: float = 0.5,
    update: str = "cd",
    seed: int = 0,
) -> list[LayerWeights]:
    """Greedily pre-train a stack of restricted Boltzmann machines.

    Parameters
    ----------
    X : array_like
        Unlabelled features of shape ``(n_samples, n_features)``, scaled to
        ``[0, 1]``.
    hidden_layer_sizes : sequence of int
        Units in each layer of the stack.
    n_epochs : int, optional
        Pre-training epochs per layer, default 10.
    learning_rate : float, optional
        Step size of the layer updates, default 0.05.
    batch_size : int, optional
        Mini-batch size, default 64.
    cd_steps : int, optional
        Gibbs sweeps per update, default 1.
    momentum : float, optional
        Momentum of the layer updates, default 0.5.
    update : {'cd', 'maxent'}, optional
        Update rule handed to each machine, default ``'cd'``.
    seed : int, optional
        Base seed; layer ``i`` uses ``seed + i`` so the layers differ.

    Returns
    -------
    list of LayerWeights
        One entry per hidden layer, in order.

    Raises
    ------
    ValueError
        If the topology is empty, a layer is not positive, or the data is not a
        two-dimensional matrix in ``[0, 1]``.

    Notes
    -----
    Each layer is trained on the *probabilities* of the layer below rather than
    on samples of them, which keeps the input to the next machine smooth and is
    the usual choice for a deterministic pre-training pass.

    Examples
    --------
    >>> import numpy as np
    >>> layers = pretrain_dbn(np.random.default_rng(0).random((30, 6)), (4, 3), n_epochs=1)
    >>> [layer.shape for layer in layers]
    [(6, 4), (4, 3)]
    """
    sizes = tuple(int(size) for size in hidden_layer_sizes)
    if not sizes:
        raise ValueError("hidden_layer_sizes must name at least one layer")
    if any(size < 1 for size in sizes):
        raise ValueError(f"every hidden layer must have at least one unit, got {sizes}")

    activations = np.asarray(X, dtype=np.float64)
    layers: list[LayerWeights] = []
    for depth, size in enumerate(sizes):
        machine = BernoulliRBM(
            n_hidden=size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            cd_steps=cd_steps,
            momentum=momentum,
            update=update,
            seed=seed + depth,
        ).fit(activations)
        layers.append(
            LayerWeights(
                weights=machine.weights_.copy(),
                bias=machine.hidden_bias_.copy(),
                reconstruction_error=float(machine.reconstruction_error_[-1]),
            )
        )
        activations = machine.transform(activations)
        _LOGGER.info(
            "pre-trained layer %d/%d (%d -> %d units), reconstruction error %.5f",
            depth + 1,
            len(sizes),
            layers[-1].shape[0],
            size,
            layers[-1].reconstruction_error,
        )
    return layers


class DBNClassifier(BaseEstimator, ClassifierMixin):
    """A sigmoid network initialised by unsupervised layer-wise pre-training.

    Parameters
    ----------
    hidden_layer_sizes : sequence of int, optional
        Topology of the hidden layers, default five layers of two hundred.
    pretrain : bool, optional
        Run the pre-training (default).  ``False`` leaves the random
        initialisation in place, which is the comparison baseline.
    pretrain_epochs : int, optional
        Pre-training epochs per layer, default 5.
    pretrain_learning_rate : float, optional
        Step size of the pre-training updates, default 0.05.
    pretrain_batch_size : int, optional
        Mini-batch size during pre-training, default 64.
    cd_steps : int, optional
        Gibbs sweeps per pre-training update, default 1.
    update : {'cd', 'maxent'}, optional
        Pre-training update rule, default ``'cd'``.
    activation, solver, learning_rate_init, momentum, batch_size, max_iter,
    alpha, early_stopping, validation_fraction, n_iter_no_change : optional
        Forwarded to the fine-tuning network; see
        :func:`voice_pathology_detection.models.mlp.build_mlp`.
    random_state : int, optional
        Seed of the pre-training and the fine-tuning, default 0.

    Attributes
    ----------
    scaler_ : sklearn.preprocessing.MinMaxScaler
        Maps the features into ``[0, 1]`` for the Bernoulli visible units.
    layers_ : list of LayerWeights
        The pre-trained weights, empty when ``pretrain`` is ``False``.
    mlp_ : MLPClassifierAutoValidation
        The fine-tuned network.
    classes_ : numpy.ndarray
        Class labels seen during ``fit``.
    n_features_in_ : int
        Number of input features.

    Notes
    -----
    The pre-trained matrices are written into the hidden layers of the network
    before fine-tuning; the softmax output layer has no unsupervised
    counterpart and keeps its random initialisation.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
        *,
        pretrain: bool = True,
        pretrain_epochs: int = 5,
        pretrain_learning_rate: float = 0.05,
        pretrain_batch_size: int = 64,
        cd_steps: int = 1,
        update: str = "cd",
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
        random_state: int = 0,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.pretrain = pretrain
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_learning_rate = pretrain_learning_rate
        self.pretrain_batch_size = pretrain_batch_size
        self.cd_steps = cd_steps
        self.update = update
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def _build_network(self) -> MLPClassifierAutoValidation:
        """Create the fine-tuning network, ready to be warm-started."""
        return MLPClassifierAutoValidation(
            hidden_layer_sizes=tuple(int(size) for size in self.hidden_layer_sizes),
            activation=self.activation,
            solver=self.solver,
            learning_rate_init=self.learning_rate_init,
            momentum=self.momentum,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            alpha=self.alpha,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            warm_start=True,
        )

    def _seed_network(
        self,
        network: MLPClassifierAutoValidation,
        X: np.ndarray,
        y: np.ndarray,
        layers: list[LayerWeights],
    ) -> None:
        """Allocate the network's parameters and overwrite them with ``layers``."""
        classes = np.unique(y)
        # One sample per class is enough to allocate coefs_ of the right shape;
        # the pre-trained layers overwrite the hidden matrices right after.
        seed_rows = np.asarray([int(np.flatnonzero(y == label)[0]) for label in classes])

        # ``partial_fit`` rejects early stopping and warns about a batch larger
        # than the data, so both are relaxed for this one allocating call and
        # restored before the real fit reads them again.
        early_stopping, batch_size = network.early_stopping, network.batch_size
        network.early_stopping = False
        if isinstance(batch_size, int | np.integer):
            network.batch_size = min(int(batch_size), int(seed_rows.size))
        try:
            network.partial_fit(X[seed_rows], y[seed_rows], classes=classes)
        finally:
            network.early_stopping = early_stopping
            network.batch_size = batch_size

        if early_stopping:
            # The allocating call ran without early stopping, so it left the
            # score history empty; the warm-started fit skips initialisation and
            # would otherwise append to it.
            network.validation_scores_ = []
            network.best_validation_score_ = -np.inf

        if len(layers) > len(network.coefs_) - 1:  # pragma: no cover - defensive
            raise ValueError(
                f"pre-trained {len(layers)} layer(s) but the network has only "
                f"{len(network.coefs_) - 1} hidden layer(s)"
            )
        for depth, layer in enumerate(layers):
            expected = network.coefs_[depth].shape
            if layer.weights.shape != expected:
                raise ValueError(
                    f"pre-trained layer {depth} has shape {layer.weights.shape} but the "
                    f"network expects {expected}"
                )
            network.coefs_[depth] = layer.weights.copy()
            network.intercepts_[depth] = layer.bias.copy()

    def fit(self, X: object, y: object) -> DBNClassifier:
        """Pre-train, then fine-tune the network.

        Parameters
        ----------
        X : array_like
            Training features of shape ``(n_samples, n_features)``.
        y : array_like
            Training labels; at least two classes must be present.

        Returns
        -------
        DBNClassifier
            ``self``, fitted.

        Raises
        ------
        ValueError
            If the inputs are empty, mismatched, or contain a single class.
        """
        features = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y).ravel()
        if features.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got {features.shape}")
        if features.shape[0] != labels.size:
            raise ValueError(
                f"X has {features.shape[0]} sample(s) but y has {labels.size} label(s)"
            )
        if features.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")
        if np.unique(labels).size < 2:
            raise ValueError(
                f"training needs at least two classes, found {np.unique(labels).tolist()}"
            )

        self.n_features_in_ = int(features.shape[1])
        self.scaler_ = MinMaxScaler().fit(features)
        scaled = np.clip(self.scaler_.transform(features), 0.0, 1.0)

        self.layers_: list[LayerWeights] = []
        if self.pretrain:
            self.layers_ = pretrain_dbn(
                scaled,
                self.hidden_layer_sizes,
                n_epochs=self.pretrain_epochs,
                learning_rate=self.pretrain_learning_rate,
                batch_size=self.pretrain_batch_size,
                cd_steps=self.cd_steps,
                momentum=self.momentum,
                update=self.update,
                seed=self.random_state,
            )

        network = self._build_network()
        self._seed_network(network, scaled, labels, self.layers_)
        network.fit(scaled, labels)

        self.mlp_ = network
        self.classes_ = network.classes_
        _LOGGER.info(
            "fine-tuned a %s network on %d sample(s) (%s initialisation)",
            tuple(int(size) for size in self.hidden_layer_sizes),
            features.shape[0],
            "pre-trained" if self.pretrain else "random",
        )
        return self

    def _prepare(self, X: object) -> np.ndarray:
        """Validate and scale features for inference."""
        check_is_fitted(self, "mlp_")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected a matrix with {self.n_features_in_} column(s), got shape "
                f"{features.shape}"
            )
        return np.clip(self.scaler_.transform(features), 0.0, 1.0)

    def predict(self, X: object) -> np.ndarray:
        """Predict the class of every sample.

        Parameters
        ----------
        X : array_like
            Features of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predicted labels of shape ``(n_samples,)``.
        """
        return np.asarray(self.mlp_.predict(self._prepare(X)))

    def predict_proba(self, X: object) -> np.ndarray:
        """Predict class posteriors.

        Parameters
        ----------
        X : array_like
            Features of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Posteriors of shape ``(n_samples, n_classes)``.
        """
        return np.asarray(self.mlp_.predict_proba(self._prepare(X)))
