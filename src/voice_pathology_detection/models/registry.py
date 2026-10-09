"""One place that turns a model name into an estimator.

Every route gets its classifier from here, so swapping the five-layer network
for a support vector machine means editing ``model.name`` in the configuration,
not the code.  Each factory receives the run's seed as its ``random_state``, so
a run can be repeated from its recorded configuration.

Examples
--------
>>> sorted(available_models())
['dbn', 'gmm', 'linear_svm', 'mlp', 'svm', 'torch_dbn']
>>> build_model("mlp", {"hidden_layer_sizes": [16]}, seed=1).hidden_layer_sizes
(16,)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from sklearn.base import BaseEstimator

from .dbn import DBNClassifier
from .gmm import GMMClassifier
from .mlp import build_mlp
from .svm import build_linear_svm, build_svm
from .torch_dbn import TorchDBNClassifier

_LOGGER = logging.getLogger(__name__)


def _mlp(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Build the sigmoid multi-layer perceptron."""
    return build_mlp(random_state=seed, **params)


def _dbn(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Build the NumPy deep belief network."""
    return DBNClassifier(random_state=seed, **params)


def _torch_dbn(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Build the PyTorch deep belief network (optional extra)."""
    return TorchDBNClassifier(random_state=seed, **params)


def _svm(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Build the kernel support vector baseline."""
    return build_svm(random_state=seed, **params)


def _linear_svm(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Build the linear support vector baseline."""
    return build_linear_svm(random_state=seed, **params)


def _gmm(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Build the Gaussian mixture baseline."""
    return GMMClassifier(random_state=seed, **params)


#: Name -> factory.  Factories take the parameter mapping and the run seed.
MODEL_FACTORIES: dict[str, Callable[[dict[str, Any], int], BaseEstimator]] = {
    "mlp": _mlp,
    "dbn": _dbn,
    "torch_dbn": _torch_dbn,
    "svm": _svm,
    "linear_svm": _linear_svm,
    "gmm": _gmm,
}

#: Short description of every model, for the CLI listing.
MODEL_DESCRIPTIONS: dict[str, str] = {
    "mlp": "Sigmoid multi-layer perceptron trained by SGD on cross-entropy.",
    "dbn": "Deep belief network: RBM pre-training, then supervised fine-tuning.",
    "torch_dbn": "Deep belief network on PyTorch, fewer options than dbn; needs the 'dl' extra.",
    "svm": "Kernel support vector machine on standardised features.",
    "linear_svm": "Linear support vector machine; no predict_proba.",
    "gmm": "One Gaussian mixture per class, combined through Bayes' rule.",
}

#: Parameters that name a network topology and must become tuples of ints.
_TOPOLOGY_KEYS: tuple[str, ...] = ("hidden_layer_sizes",)


def available_models() -> list[str]:
    """Return the names the registry knows.

    Returns
    -------
    list of str
        Sorted model names.
    """
    return sorted(MODEL_FACTORIES)


def _normalise_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Coerce YAML-friendly values into what the factories expect."""
    prepared = dict(params or {})
    for key in _TOPOLOGY_KEYS:
        if key in prepared and isinstance(prepared[key], list | tuple):
            prepared[key] = tuple(int(size) for size in prepared[key])
    return prepared


def build_model(
    name: str,
    params: Mapping[str, Any] | None = None,
    *,
    seed: int = 0,
) -> BaseEstimator:
    """Instantiate a registered estimator.

    Parameters
    ----------
    name : str
        A key of :data:`MODEL_FACTORIES`.
    params : mapping, optional
        Keyword arguments for the factory.  Lists naming a topology are
        converted to tuples, so a YAML ``[200, 200]`` works unchanged.
    seed : int, optional
        Run seed, passed to the factory as its ``random_state``.

    Returns
    -------
    sklearn.base.BaseEstimator
        The unfitted estimator.

    Raises
    ------
    KeyError
        If the name is not registered; the message lists the valid names.
    TypeError
        If a parameter is not accepted by the chosen factory; the message names
        the model and the offending keyword.

    Examples
    --------
    >>> build_model("gmm", {"n_components": 2}).n_components
    2
    """
    if name not in MODEL_FACTORIES:
        raise KeyError(f"unknown model {name!r}; available models are {available_models()}")
    prepared = _normalise_params(params)
    try:
        model = MODEL_FACTORIES[name](prepared, int(seed))
    except TypeError as exc:
        raise TypeError(
            f"model {name!r} does not accept these parameters {sorted(prepared)}: {exc}"
        ) from exc
    _LOGGER.debug("built model %r with parameters %s", name, prepared)
    return model


def supports_proba(model: BaseEstimator) -> bool:
    """Whether an estimator can produce class posteriors.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The estimator to inspect.

    Returns
    -------
    bool
        ``True`` when ``predict_proba`` is available, which the
        probability-averaging aggregation rule requires.

    Examples
    --------
    >>> supports_proba(build_model("linear_svm"))
    False
    """
    return hasattr(model, "predict_proba")
