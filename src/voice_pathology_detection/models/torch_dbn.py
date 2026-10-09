"""Optional PyTorch implementation of the deep belief network.

The NumPy implementation in :mod:`voice_pathology_detection.models.dbn` is the
reference: it needs only the core dependencies, and the tests always run it.
This module trains the same architecture (CD-1 pre-training of each layer, then
supervised fine-tuning) on PyTorch, for corpora large enough that a GPU is
worth the dependency: hundreds of thousands of frames and a five-layer network,
where the mini-batch loop is the bottleneck.

It has fewer options than
:class:`~voice_pathology_detection.models.dbn.DBNClassifier`.  Fine-tuning
runs for exactly ``max_iter`` epochs of plain SGD with momentum, with no early
stopping and no L2 penalty, and pre-training always uses one Gibbs step and the
contrastive divergence update.  ``cd_steps``, ``update``, ``alpha``,
``early_stopping`` and the other ``DBNClassifier`` options are not accepted, so
a ``dbn`` configuration has to be cut down to the parameters listed on
:class:`TorchDBNClassifier` before it can run here.

PyTorch is an optional extra.  It is only imported inside functions, so this
module imports and the model can be built without it; ``fit`` and
``predict_proba`` need it.  From a clone of the repository::

    pip install -e ".[dl]"

or ``pip install "torch>=2.2"`` on its own.

Examples
--------
>>> torch_is_available() in (True, False)
True

References
----------
.. [1] G. E. Hinton, S. Osindero, and Y.-W. Teh, "A fast learning algorithm for
       deep belief nets", *Neural Computation*, 18(7):1527-1554, 2006.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler

from .mlp import DEFAULT_HIDDEN_LAYERS

_LOGGER = logging.getLogger(__name__)

#: Message shown when the optional extra is missing.
_MISSING = (
    'PyTorch is not installed. Install it with `pip install -e ".[dl]"` from a clone of '
    'the repository, or `pip install "torch>=2.2"`; or use model name "dbn", which is '
    "the NumPy implementation and needs nothing beyond the core dependencies."
)


class TorchNotAvailableError(ImportError):
    """Raised when a PyTorch-only code path is reached without PyTorch."""


def torch_is_available() -> bool:
    """Whether the optional PyTorch extra is installed.

    Returns
    -------
    bool
        ``True`` when :mod:`torch` can be imported.

    Examples
    --------
    >>> isinstance(torch_is_available(), bool)
    True
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch() -> Any:
    """Import and return :mod:`torch`, or raise an error naming the extra."""
    try:
        import torch
    except ImportError as exc:
        raise TorchNotAvailableError(_MISSING) from exc
    return torch


def train_rbm_torch(
    X: object,
    n_hidden: int,
    *,
    n_epochs: int = 10,
    learning_rate: float = 0.05,
    batch_size: int = 128,
    cd_steps: int = 1,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one restricted Boltzmann machine with PyTorch tensors.

    Parameters
    ----------
    X : array_like
        Visible data of shape ``(n_samples, n_visible)``, values in ``[0, 1]``.
    n_hidden : int
        Number of hidden units.
    n_epochs : int, optional
        Passes over the data, default 10.
    learning_rate : float, optional
        Step size, default 0.05.
    batch_size : int, optional
        Mini-batch size, default 128.
    cd_steps : int, optional
        Gibbs sweeps in the negative phase, default 1.
    device : str, optional
        Torch device, default ``"cpu"``.
    seed : int, optional
        Seed, default 0.

    Returns
    -------
    weights : numpy.ndarray
        Matrix of shape ``(n_visible, n_hidden)``.
    bias : numpy.ndarray
        Hidden bias of shape ``(n_hidden,)``.

    Raises
    ------
    TorchNotAvailableError
        If PyTorch is not installed.
    ValueError
        If the data is not a two-dimensional matrix in ``[0, 1]``.
    """
    torch = _require_torch()

    data = np.asarray(X, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_visible), got shape {data.shape}")
    if data.size and (data.min() < -1e-6 or data.max() > 1.0 + 1e-6):
        raise ValueError("Bernoulli visible units need inputs in [0, 1]")

    torch.manual_seed(seed)
    tensor = torch.as_tensor(data, device=device)
    n_visible = tensor.shape[1]

    weights = torch.randn(n_visible, n_hidden, device=device) * 0.01
    visible_bias = torch.zeros(n_visible, device=device)
    hidden_bias = torch.zeros(n_hidden, device=device)

    n_samples = tensor.shape[0]
    batch_size = min(batch_size, n_samples)
    for epoch in range(n_epochs):
        order = torch.randperm(n_samples, device=device)
        error = 0.0
        for start in range(0, n_samples, batch_size):
            batch = tensor[order[start : start + batch_size]]
            positive_hidden = torch.sigmoid(batch @ weights + hidden_bias)

            negative_visible, negative_hidden = batch, positive_hidden
            for _ in range(cd_steps):
                sample = torch.bernoulli(negative_hidden)
                negative_visible = torch.sigmoid(sample @ weights.T + visible_bias)
                negative_hidden = torch.sigmoid(negative_visible @ weights + hidden_bias)

            scale = learning_rate / batch.shape[0]
            weights += scale * (batch.T @ positive_hidden - negative_visible.T @ negative_hidden)
            visible_bias += scale * (batch - negative_visible).sum(dim=0)
            hidden_bias += scale * (positive_hidden - negative_hidden).sum(dim=0)
            error += float(((batch - negative_visible) ** 2).sum())
        _LOGGER.debug(
            "torch RBM epoch %d/%d: reconstruction error %.5f",
            epoch + 1,
            n_epochs,
            error / float(data.size),
        )

    return (
        weights.detach().cpu().numpy().astype(np.float64),
        hidden_bias.detach().cpu().numpy().astype(np.float64),
    )


class TorchDBNClassifier(BaseEstimator, ClassifierMixin):
    """Deep belief network trained with PyTorch.

    It fits and predicts like
    :class:`voice_pathology_detection.models.dbn.DBNClassifier` and can stand in
    for it in any route, but it takes only the parameters below (see the module
    docstring for what is left out).

    Parameters
    ----------
    hidden_layer_sizes : sequence of int, optional
        Topology of the hidden layers, default five layers of two hundred.
    pretrain : bool, optional
        Run the layer-wise pre-training, default ``True``.
    pretrain_epochs : int, optional
        Pre-training epochs per layer, default 5.
    pretrain_learning_rate : float, optional
        Step size of the pre-training, default 0.05.
    max_iter : int, optional
        Fine-tuning epochs, default 30.
    batch_size : int, optional
        Mini-batch size, default 90.
    learning_rate_init : float, optional
        Fine-tuning step size, default 0.001.
    momentum : float, optional
        Momentum of the fine-tuning optimiser, default 0.5.
    device : str, optional
        Torch device, default ``"cpu"``.
    random_state : int, optional
        Seed, default 0.

    Raises
    ------
    TorchNotAvailableError
        On ``fit``, when PyTorch is not installed.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = DEFAULT_HIDDEN_LAYERS,
        *,
        pretrain: bool = True,
        pretrain_epochs: int = 5,
        pretrain_learning_rate: float = 0.05,
        max_iter: int = 30,
        batch_size: int = 90,
        learning_rate_init: float = 0.001,
        momentum: float = 0.5,
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.pretrain = pretrain
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_learning_rate = pretrain_learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.momentum = momentum
        self.device = device
        self.random_state = random_state

    def fit(self, X: object, y: object) -> TorchDBNClassifier:
        """Pre-train and fine-tune the network.

        Parameters
        ----------
        X : array_like
            Training features of shape ``(n_samples, n_features)``.
        y : array_like
            Training labels.

        Returns
        -------
        TorchDBNClassifier
            ``self``, fitted.

        Raises
        ------
        TorchNotAvailableError
            If PyTorch is not installed.
        ValueError
            If the inputs are mismatched or contain a single class.
        """
        torch = _require_torch()
        from torch import nn

        features = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y).ravel()
        if features.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got {features.shape}")
        if features.shape[0] != labels.size:
            raise ValueError(
                f"X has {features.shape[0]} sample(s) but y has {labels.size} label(s)"
            )
        self.classes_ = np.unique(labels)
        if self.classes_.size < 2:
            raise ValueError("training needs at least two classes")
        self.n_features_in_ = int(features.shape[1])

        torch.manual_seed(self.random_state)
        self.scaler_ = MinMaxScaler().fit(features)
        scaled = np.clip(self.scaler_.transform(features), 0.0, 1.0)
        targets = np.searchsorted(self.classes_, labels)

        sizes = [int(size) for size in self.hidden_layer_sizes]
        layers: list[nn.Module] = []
        in_features = self.n_features_in_
        pretrained: list[tuple[np.ndarray, np.ndarray]] = []
        activations = scaled
        for depth, size in enumerate(sizes):
            if self.pretrain:
                weights, bias = train_rbm_torch(
                    activations,
                    size,
                    n_epochs=self.pretrain_epochs,
                    learning_rate=self.pretrain_learning_rate,
                    batch_size=self.batch_size,
                    device=self.device,
                    seed=self.random_state + depth,
                )
                pretrained.append((weights, bias))
                activations = 1.0 / (1.0 + np.exp(-(activations @ weights + bias)))
            layers.extend([nn.Linear(in_features, size), nn.Sigmoid()])
            in_features = size
        layers.append(nn.Linear(in_features, int(self.classes_.size)))
        network = nn.Sequential(*layers).to(self.device)

        with torch.no_grad():
            for depth, (weights, bias) in enumerate(pretrained):
                linear = network[2 * depth]
                linear.weight.copy_(torch.as_tensor(weights.T, dtype=linear.weight.dtype))
                linear.bias.copy_(torch.as_tensor(bias, dtype=linear.bias.dtype))

        optimiser = torch.optim.SGD(
            network.parameters(), lr=self.learning_rate_init, momentum=self.momentum
        )
        criterion = nn.CrossEntropyLoss()
        tensor_X = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        tensor_y = torch.as_tensor(targets, dtype=torch.long, device=self.device)

        n_samples = tensor_X.shape[0]
        batch_size = min(self.batch_size, n_samples)
        for epoch in range(self.max_iter):
            order = torch.randperm(n_samples, device=self.device)
            total = 0.0
            for start in range(0, n_samples, batch_size):
                rows = order[start : start + batch_size]
                optimiser.zero_grad()
                loss = criterion(network(tensor_X[rows]), tensor_y[rows])
                loss.backward()
                optimiser.step()
                total += float(loss.item())
            _LOGGER.debug(
                "torch fine-tuning epoch %d/%d: loss %.5f", epoch + 1, self.max_iter, total
            )

        self.network_ = network
        return self

    def predict_proba(self, X: object) -> np.ndarray:
        """Class posteriors.

        Parameters
        ----------
        X : array_like
            Features of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Posteriors of shape ``(n_samples, n_classes)``.
        """
        torch = _require_torch()
        if not hasattr(self, "network_"):
            raise ValueError("this model must be fitted before it can predict")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected a matrix with {self.n_features_in_} column(s), got shape "
                f"{features.shape}"
            )
        scaled = np.clip(self.scaler_.transform(features), 0.0, 1.0)
        with torch.no_grad():
            logits = self.network_(torch.as_tensor(scaled, dtype=torch.float32, device=self.device))
            return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)

    def predict(self, X: object) -> np.ndarray:
        """Predict the most probable class of every sample.

        Parameters
        ----------
        X : array_like
            Features of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predicted labels.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
