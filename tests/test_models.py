"""Estimators: the network, its pre-training, the baselines and the registry."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from voice_pathology_detection.models.dbn import DBNClassifier, pretrain_dbn
from voice_pathology_detection.models.gmm import GMMClassifier
from voice_pathology_detection.models.maxent import (
    maxent_gamma,
    maxent_residual,
    sigmoid,
    solve_maxent,
)
from voice_pathology_detection.models.mlp import build_mlp
from voice_pathology_detection.models.rbm import BernoulliRBM
from voice_pathology_detection.models.registry import (
    MODEL_DESCRIPTIONS,
    available_models,
    build_model,
    supports_proba,
)
from voice_pathology_detection.models.torch_dbn import TorchNotAvailableError, torch_is_available


@pytest.fixture
def separable() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated Gaussian clusters."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2.0, 0.6, (80, 5)), rng.normal(2.0, 0.6, (80, 5))])
    y = np.array([0] * 80 + [1] * 80)
    return X, y


# -- the logistic function ---------------------------------------------
def test_the_logistic_function_is_stable_at_both_extremes() -> None:
    assert float(sigmoid(0.0)) == pytest.approx(0.5)
    assert float(sigmoid(800.0)) == pytest.approx(1.0)
    assert float(sigmoid(-800.0)) == pytest.approx(0.0)
    assert np.all(np.isfinite(sigmoid(np.array([-1e4, 0.0, 1e4]))))


def test_hidden_posteriors_are_one_half_for_a_blank_machine() -> None:
    rbm = BernoulliRBM(n_hidden=2)
    rbm.weights_ = np.zeros((4, 2))
    rbm.hidden_bias_ = np.zeros(2)
    assert np.allclose(rbm.hidden_probabilities(np.zeros((3, 4))), 0.5)


# -- the maximum-entropy solve -----------------------------------------
def test_newton_drives_the_residual_to_zero() -> None:
    rng = np.random.default_rng(0)
    f_model = rng.random(20)
    mu_data = rng.random(20)
    lam = rng.normal(0.0, 0.1, 20)

    solution = solve_maxent(f_model, mu_data, lam, var=0.1, tol=1e-10, max_iter=200)
    assert solution.converged
    residual = maxent_residual(solution.gamma, f_model, mu_data, lam, var=0.1)
    assert np.all(np.abs(residual) < 1e-8)


def test_the_solution_matches_an_independent_root_finder() -> None:
    f_model = np.array([0.4, 0.25])
    mu_data = np.array([0.7, 0.1])
    lam = np.array([0.05, -0.05])
    total = float(f_model.sum())
    gamma = maxent_gamma(f_model, mu_data, lam, var=0.1, tol=1e-12, max_iter=200)

    for i in range(f_model.size):

        def scalar(g: float, i: int = i) -> float:
            return float(f_model[i] * 0.1 * np.exp(g * total) + g + lam[i] - mu_data[i] * 0.1)

        assert gamma[i] == pytest.approx(brentq(scalar, -5.0, 5.0), abs=1e-8)


def test_matching_statistics_and_zero_parameters_give_a_zero_update() -> None:
    gamma = maxent_gamma(np.full(4, 0.5), np.full(4, 0.5), np.zeros(4), var=1.0)
    assert np.allclose(gamma, 0.0, atol=1e-9)


def test_the_solver_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="tol must be positive"):
        solve_maxent(np.ones(2), np.ones(2), np.zeros(2), tol=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        solve_maxent(np.ones(2), np.ones(2), np.zeros(2), max_iter=0)
    with pytest.raises(ValueError, match="damping"):
        solve_maxent(np.ones(2), np.ones(2), np.zeros(2), damping=0.0)
    with pytest.raises(ValueError, match="var must be positive"):
        maxent_residual(np.zeros(2), np.ones(2), np.ones(2), np.zeros(2), var=0.0)
    with pytest.raises(ValueError, match="must share one shape"):
        maxent_residual(np.zeros(2), np.ones(3), np.ones(2), np.zeros(2))


# -- the restricted Boltzmann machine ----------------------------------
def test_training_lowers_the_reconstruction_error() -> None:
    rng = np.random.default_rng(0)
    # Each row is the same five values twice, so the ten inputs fit in six
    # hidden units.
    half = rng.random((120, 5))
    X = np.hstack([half, half])

    machine = BernoulliRBM(n_hidden=6, n_epochs=25, learning_rate=0.1, seed=0).fit(X)
    assert len(machine.reconstruction_error_) == 25
    assert machine.reconstruction_error_[-1] < machine.reconstruction_error_[0]


def test_the_machine_projects_and_reconstructs_with_the_right_shapes() -> None:
    rng = np.random.default_rng(1)
    X = rng.random((30, 8))
    machine = BernoulliRBM(n_hidden=4, n_epochs=2, seed=0).fit(X)
    assert machine.transform(X).shape == (30, 4)
    assert machine.reconstruct(X).shape == (30, 8)
    assert np.all((machine.transform(X) >= 0) & (machine.transform(X) <= 1))


def test_the_maximum_entropy_update_also_trains() -> None:
    rng = np.random.default_rng(2)
    X = rng.random((60, 6))
    machine = BernoulliRBM(n_hidden=4, n_epochs=5, update="maxent", learning_rate=0.5, seed=0).fit(
        X
    )
    assert np.all(np.isfinite(machine.weights_))
    assert len(machine.reconstruction_error_) == 5


def test_training_is_reproducible_given_a_seed() -> None:
    rng = np.random.default_rng(3)
    X = rng.random((40, 5))
    first = BernoulliRBM(n_hidden=3, n_epochs=3, seed=7).fit(X)
    second = BernoulliRBM(n_hidden=3, n_epochs=3, seed=7).fit(X)
    assert np.allclose(first.weights_, second.weights_)


def test_visible_units_outside_the_unit_interval_are_refused() -> None:
    with pytest.raises(ValueError, match=r"inputs in \[0, 1\]"):
        BernoulliRBM(n_hidden=2, n_epochs=1).fit(np.array([[5.0, -3.0]]))


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"n_hidden": 0}, "n_hidden"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"n_epochs": 0}, "n_epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"cd_steps": 0}, "cd_steps"),
        ({"momentum": 1.0}, "momentum"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"update": "sgd"}, "update must be"),
    ],
)
def test_the_machine_validates_its_hyper_parameters(kwargs: dict, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        BernoulliRBM(**kwargs).fit(np.random.default_rng(0).random((10, 3)))


def test_an_unfitted_machine_cannot_transform() -> None:
    with pytest.raises(ValueError, match="must be fitted"):
        BernoulliRBM().transform(np.zeros((2, 3)))


# -- the deep belief network -------------------------------------------
def test_pre_training_produces_one_matrix_per_layer() -> None:
    rng = np.random.default_rng(0)
    layers = pretrain_dbn(rng.random((60, 8)), (5, 3), n_epochs=2, seed=0)
    assert [layer.shape for layer in layers] == [(8, 5), (5, 3)]
    assert all(layer.bias.shape == (layer.shape[1],) for layer in layers)
    assert all(np.isfinite(layer.reconstruction_error) for layer in layers)


def test_pre_training_validates_the_topology() -> None:
    X = np.random.default_rng(0).random((10, 3))
    with pytest.raises(ValueError, match="at least one layer"):
        pretrain_dbn(X, ())
    with pytest.raises(ValueError, match="at least one unit"):
        pretrain_dbn(X, (4, 0))


def test_fine_tuning_starts_from_the_pre_trained_weights(separable) -> None:
    X, y = separable
    model = DBNClassifier(
        hidden_layer_sizes=(6,), pretrain_epochs=3, max_iter=1, early_stopping=False
    ).fit(X, y)
    assert len(model.layers_) == 1
    # One epoch of fine-tuning cannot move far from the initialisation, so the
    # fitted first layer must still resemble the pre-trained one.
    assert np.corrcoef(model.mlp_.coefs_[0].ravel(), model.layers_[0].weights.ravel())[0, 1] > 0.9


def test_the_network_fits_with_early_stopping_left_on(separable) -> None:
    # Seeding the weights allocates the parameters through partial_fit, which
    # cannot coexist with early stopping; the fit must still honour it.
    X, y = separable
    model = DBNClassifier(
        hidden_layer_sizes=(8,),
        pretrain_epochs=2,
        max_iter=40,
        learning_rate_init=0.05,
        early_stopping=True,
        validation_fraction=0.2,
    ).fit(X, y)
    assert model.mlp_.early_stopping is True
    assert model.mlp_.validation_scores_
    assert model.predict(X).shape == (X.shape[0],)


def test_pre_training_can_be_switched_off(separable) -> None:
    X, y = separable
    model = DBNClassifier(
        hidden_layer_sizes=(6,), pretrain=False, max_iter=5, early_stopping=False
    ).fit(X, y)
    assert model.layers_ == []
    assert model.predict(X).shape == (X.shape[0],)


def test_the_network_learns_a_separable_problem(separable) -> None:
    X, y = separable
    model = DBNClassifier(
        hidden_layer_sizes=(16,),
        pretrain_epochs=3,
        max_iter=200,
        learning_rate_init=0.05,
        early_stopping=False,
    ).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_the_network_validates_its_training_data() -> None:
    with pytest.raises(ValueError, match="at least two classes"):
        DBNClassifier(hidden_layer_sizes=(4,)).fit(np.zeros((5, 2)), np.zeros(5))
    with pytest.raises(ValueError, match="sample"):
        DBNClassifier(hidden_layer_sizes=(4,)).fit(np.zeros((5, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="must be 2-D"):
        DBNClassifier(hidden_layer_sizes=(4,)).fit(np.zeros(5), np.zeros(5))


def test_the_network_refuses_a_dimension_change(separable) -> None:
    X, y = separable
    model = DBNClassifier(hidden_layer_sizes=(4,), pretrain_epochs=1, max_iter=2).fit(X, y)
    with pytest.raises(ValueError, match="expected a matrix with 5 column"):
        model.predict(np.zeros((3, 2)))


# -- the perceptron -----------------------------------------------------
def test_the_perceptron_learns_a_separable_problem(separable) -> None:
    X, y = separable
    model = build_mlp(
        hidden_layer_sizes=(16,), max_iter=300, learning_rate_init=0.05, early_stopping=False
    ).fit(X, y)
    assert model.score(X, y) > 0.95


def test_early_stopping_is_relaxed_when_the_data_is_too_small(caplog) -> None:
    X = np.vstack([np.zeros((4, 3)), np.ones((4, 3))])
    y = np.array([0] * 4 + [1] * 4)
    model = build_mlp(hidden_layer_sizes=(4,), max_iter=5, early_stopping=True)
    model.fit(X, y)  # would raise inside scikit-learn without the guard
    assert model.early_stopping is True  # the configuration is left intact


def test_the_topology_is_validated() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        build_mlp(hidden_layer_sizes=())
    with pytest.raises(ValueError, match="at least one unit"):
        build_mlp(hidden_layer_sizes=(10, 0))


# -- the Gaussian mixture baseline -------------------------------------
def test_the_mixture_classifies_a_separable_problem(separable) -> None:
    X, y = separable
    model = GMMClassifier(n_components=2, random_state=0).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(model.predict_log_proba(X) <= 1e-9)


def test_uniform_priors_differ_from_empirical_ones() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (90, 3)), rng.normal(1.0, 1, (10, 3))])
    y = np.array([0] * 90 + [1] * 10)
    empirical = GMMClassifier(n_components=1, priors="empirical").fit(X, y)
    uniform = GMMClassifier(n_components=1, priors="uniform").fit(X, y)
    assert not np.allclose(empirical.log_priors_, uniform.log_priors_)
    assert np.allclose(uniform.log_priors_, -np.log(2))


def test_the_mixture_shrinks_to_fit_a_tiny_class(caplog) -> None:
    X = np.vstack([np.random.default_rng(0).normal(0, 1, (20, 2)), np.ones((2, 2))])
    y = np.array([0] * 20 + [1] * 2)
    model = GMMClassifier(n_components=8, random_state=0).fit(X, y)
    assert model.models_[1].n_components == 2


def test_the_mixture_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="n_components"):
        GMMClassifier(n_components=0).fit(np.zeros((4, 2)), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="priors must be"):
        GMMClassifier(priors="flat").fit(np.zeros((4, 2)), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="at least two classes"):
        GMMClassifier(n_components=1).fit(np.zeros((4, 2)), np.zeros(4))


# -- the registry -------------------------------------------------------
def test_every_registered_model_is_documented() -> None:
    assert set(available_models()) == set(MODEL_DESCRIPTIONS)
    assert "mlp" in available_models()
    assert "dbn" in available_models()


@pytest.mark.parametrize("name", ["mlp", "dbn", "svm", "linear_svm", "gmm", "torch_dbn"])
def test_every_registered_model_can_be_built(name: str) -> None:
    assert build_model(name, seed=1) is not None


def test_an_unknown_model_lists_the_alternatives() -> None:
    with pytest.raises(KeyError, match="available models"):
        build_model("transformer")


def test_an_unsupported_parameter_names_the_model() -> None:
    with pytest.raises(TypeError, match="model 'gmm' does not accept"):
        build_model("gmm", {"hidden_layer_sizes": [10]})


def test_a_yaml_topology_list_becomes_a_tuple() -> None:
    model = build_model("mlp", {"hidden_layer_sizes": [16, 8]})
    assert model.hidden_layer_sizes == (16, 8)


def test_probability_support_is_reported() -> None:
    assert supports_proba(build_model("mlp"))
    assert supports_proba(build_model("gmm"))
    assert not supports_proba(build_model("linear_svm"))


def test_the_seed_reaches_the_estimator() -> None:
    assert build_model("mlp", seed=11).random_state == 11
    assert build_model("gmm", seed=11).random_state == 11


# -- the optional PyTorch route ----------------------------------------
def test_the_torch_route_reports_its_availability() -> None:
    assert isinstance(torch_is_available(), bool)


@pytest.mark.parametrize("name", ["cd_steps", "update", "alpha", "early_stopping"])
def test_the_torch_network_refuses_the_options_it_does_not_have(name: str) -> None:
    # The README lists these as dbn-only; building fails before torch is needed.
    with pytest.raises(TypeError, match="torch_dbn"):
        build_model("torch_dbn", {name: 1})


@pytest.mark.skipif(torch_is_available(), reason="PyTorch is installed")
def test_without_torch_the_error_explains_the_extra(separable) -> None:
    from voice_pathology_detection.models.torch_dbn import TorchDBNClassifier

    X, y = separable
    with pytest.raises(TorchNotAvailableError, match=r'pip install -e "\.\[dl\]"'):
        TorchDBNClassifier(hidden_layer_sizes=(4,)).fit(X, y)


def test_the_torch_network_matches_the_numpy_interface(separable) -> None:
    pytest.importorskip("torch")
    from voice_pathology_detection.models.torch_dbn import TorchDBNClassifier

    X, y = separable
    model = TorchDBNClassifier(hidden_layer_sizes=(8,), pretrain_epochs=2, max_iter=30).fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)
    assert model.predict_proba(X).shape == (X.shape[0], 2)
