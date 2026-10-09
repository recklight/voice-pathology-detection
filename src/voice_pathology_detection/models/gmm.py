"""Gaussian mixture baseline, the classic method for this task.

Before neural networks, pathological voice quality was usually assessed by
fitting one Gaussian mixture per class to short-term cepstral parameters and
comparing likelihoods [1].  I keep it as the reference the networks have to
beat.  It is generative and needs no frame labels beyond the class of the
recording, and it holds up on a small corpus.

:class:`GMMClassifier` fits one :class:`sklearn.mixture.GaussianMixture` per
class and classifies by the posterior

.. math::

    \\log p(k \\mid x) \\propto \\log p(x \\mid k) + \\log \\pi_k,

with the class priors :math:`\\pi_k` estimated from the training frequencies
unless they are set to uniform.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> X = np.vstack([rng.normal(-2, 1, (60, 3)), rng.normal(2, 1, (60, 3))])
>>> y = np.array([0] * 60 + [1] * 60)
>>> model = GMMClassifier(n_components=2, random_state=0).fit(X, y)
>>> float((model.predict(X) == y).mean()) > 0.9
True

References
----------
.. [1] J. I. Godino-Llorente, P. Gomez-Vilda, and M. Blanco-Velasco,
       "Dimensionality reduction of a pathological voice quality assessment
       system based on Gaussian mixture models and short-term cepstral
       parameters", *IEEE Transactions on Biomedical Engineering*,
       53(10):1943-1953, 2006.
.. [2] D. A. Reynolds and R. C. Rose, "Robust text-independent speaker
       identification using Gaussian mixture speaker models", *IEEE
       Transactions on Speech and Audio Processing*, 3(1):72-83, 1995.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

_LOGGER = logging.getLogger(__name__)


class GMMClassifier(BaseEstimator, ClassifierMixin):
    """One Gaussian mixture per class, combined through Bayes' rule.

    Parameters
    ----------
    n_components : int, optional
        Mixture components per class, default 8.
    covariance_type : str, optional
        Covariance structure, default ``"diag"``.  Diagonal covariances are the
        standard choice for cepstral features: they have far fewer parameters
        than full ones, and the discrete cosine transform has already removed
        most of the correlation between coefficients.
    reg_covar : float, optional
        Floor added to the covariance diagonal, default 1e-6.
    max_iter : int, optional
        EM iterations, default 100.
    priors : {'empirical', 'uniform'}, optional
        Whether the class priors follow the training frequencies (default) or
        are flat, which suits an imbalanced corpus evaluated by UAR.
    standardize : bool, optional
        Standardise the features before fitting, default ``True``.
    random_state : int, optional
        Seed of the EM initialisation, default 0.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Class labels seen during ``fit``.
    models_ : dict
        One fitted mixture per class label.
    log_priors_ : numpy.ndarray
        Logarithm of the class priors, aligned with :attr:`classes_`.
    scaler_ : sklearn.preprocessing.StandardScaler or None
        The fitted scaler, or ``None`` when ``standardize`` is ``False``.
    n_features_in_ : int
        Number of input features.
    """

    def __init__(
        self,
        n_components: int = 8,
        *,
        covariance_type: str = "diag",
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        priors: str = "empirical",
        standardize: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.priors = priors
        self.standardize = standardize
        self.random_state = random_state

    def fit(self, X: object, y: object) -> GMMClassifier:
        """Fit one mixture per class.

        Parameters
        ----------
        X : array_like
            Training features of shape ``(n_samples, n_features)``.
        y : array_like
            Training labels; at least two classes must be present.

        Returns
        -------
        GMMClassifier
            ``self``, fitted.

        Raises
        ------
        ValueError
            If the inputs are mismatched or single-class, or if a class has
            fewer frames than the requested number of components.
        """
        if self.n_components < 1:
            raise ValueError(f"n_components must be positive, got {self.n_components}")
        if self.priors not in {"empirical", "uniform"}:
            raise ValueError(f"priors must be 'empirical' or 'uniform', got {self.priors!r}")

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
            raise ValueError(f"training needs at least two classes, found {self.classes_.tolist()}")
        self.n_features_in_ = int(features.shape[1])

        self.scaler_ = StandardScaler().fit(features) if self.standardize else None
        scaled = features if self.scaler_ is None else self.scaler_.transform(features)

        self.models_ = {}
        counts = np.zeros(self.classes_.size, dtype=np.float64)
        for position, label in enumerate(self.classes_):
            subset = scaled[labels == label]
            counts[position] = subset.shape[0]
            components = min(self.n_components, subset.shape[0])
            if components < self.n_components:
                _LOGGER.warning(
                    "class %s has only %d frame(s); reducing its mixture from %d to %d "
                    "component(s)",
                    label,
                    subset.shape[0],
                    self.n_components,
                    components,
                )
            if components < 1:
                raise ValueError(f"class {label} has no training frames")
            self.models_[label] = GaussianMixture(
                n_components=components,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                random_state=self.random_state,
            ).fit(subset)

        if self.priors == "uniform":
            self.log_priors_ = np.full(self.classes_.size, -np.log(self.classes_.size))
        else:
            self.log_priors_ = np.log(counts / counts.sum())
        _LOGGER.info(
            "fitted %d class mixture(s) on %d frame(s)", self.classes_.size, features.shape[0]
        )
        return self

    def _log_joint(self, X: object) -> np.ndarray:
        """Log joint probability of every sample and class."""
        check_is_fitted(self, "models_")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected a matrix with {self.n_features_in_} column(s), got shape "
                f"{features.shape}"
            )
        scaled = features if self.scaler_ is None else self.scaler_.transform(features)
        likelihoods = np.column_stack(
            [self.models_[label].score_samples(scaled) for label in self.classes_]
        )
        return likelihoods + self.log_priors_[None, :]

    def predict_log_proba(self, X: object) -> np.ndarray:
        """Log posterior of every class.

        Parameters
        ----------
        X : array_like
            Features of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Log posteriors of shape ``(n_samples, n_classes)``.
        """
        joint = self._log_joint(X)
        return joint - logsumexp(joint, axis=1, keepdims=True)

    def predict_proba(self, X: object) -> np.ndarray:
        """Class posteriors.

        Parameters
        ----------
        X : array_like
            Features of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Posteriors of shape ``(n_samples, n_classes)``, rows summing to one.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: object) -> np.ndarray:
        """Predict the most probable class of every sample.

        Parameters
        ----------
        X : array_like
            Features of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predicted labels of shape ``(n_samples,)``.
        """
        return self.classes_[np.argmax(self._log_joint(X), axis=1)]
