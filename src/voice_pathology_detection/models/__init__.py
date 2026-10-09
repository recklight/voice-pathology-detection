"""Classifiers: the network, its pre-training and the baselines."""

from __future__ import annotations

from .dbn import DBNClassifier, LayerWeights, pretrain_dbn
from .gmm import GMMClassifier
from .maxent import MaxEntSolution, maxent_gamma, maxent_residual, sigmoid, solve_maxent
from .mlp import MLPClassifierAutoValidation, build_mlp
from .rbm import BernoulliRBM
from .registry import MODEL_DESCRIPTIONS, available_models, build_model, supports_proba
from .svm import build_linear_svm, build_svm
from .torch_dbn import TorchDBNClassifier, TorchNotAvailableError, torch_is_available

__all__ = [
    "MODEL_DESCRIPTIONS",
    "BernoulliRBM",
    "DBNClassifier",
    "GMMClassifier",
    "LayerWeights",
    "MLPClassifierAutoValidation",
    "MaxEntSolution",
    "TorchDBNClassifier",
    "TorchNotAvailableError",
    "available_models",
    "build_linear_svm",
    "build_mlp",
    "build_model",
    "build_svm",
    "maxent_gamma",
    "maxent_residual",
    "pretrain_dbn",
    "sigmoid",
    "solve_maxent",
    "supports_proba",
    "torch_is_available",
]
