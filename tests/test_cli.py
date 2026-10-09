"""The command line interface, exercised through Typer's test runner."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from voice_pathology_detection.cli import _parse_overrides, app

runner = CliRunner()


@pytest.fixture
def cli_config(corpus_config, tmp_path):
    """Write the synthetic-corpus configuration to a file for the CLI."""
    return corpus_config.to_yaml(tmp_path / "config.yaml")


# -- option parsing -----------------------------------------------------
def test_overrides_are_parsed_with_their_types() -> None:
    parsed = _parse_overrides(["split.n_splits=3", "seed=7", "mfcc.use_deltas=false"])
    assert parsed == {"split.n_splits": 3, "seed": 7, "mfcc.use_deltas": False}


def test_an_override_without_a_value_is_rejected() -> None:
    from typer import BadParameter

    with pytest.raises(BadParameter, match="key=value"):
        _parse_overrides(["split.n_splits"])


def test_no_overrides_is_an_empty_mapping() -> None:
    assert _parse_overrides(None) == {}


# -- commands -----------------------------------------------------------
def test_the_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("info", "prepare", "features", "train", "predict", "demo"):
        assert command in result.output


def test_info_prints_the_configuration_and_the_models() -> None:
    result = runner.invoke(app, ["info", "--log-level", "ERROR"])
    assert result.exit_code == 0
    assert "Available models:" in result.output
    assert "mlp" in result.output
    assert "Vocal palsy" in result.output


def test_the_log_level_of_the_file_holds_unless_the_option_is_given(tmp_path) -> None:
    from voice_pathology_detection.config import Config

    config_file = Config(log_level="WARNING").to_yaml(tmp_path / "quiet.yaml")
    from_file = runner.invoke(app, ["info", "-c", str(config_file)])
    assert from_file.exit_code == 0, from_file.output
    assert "log_level: WARNING" in from_file.output

    from_option = runner.invoke(app, ["info", "-c", str(config_file), "--log-level", "error"])
    assert from_option.exit_code == 0, from_option.output
    assert "log_level: ERROR" in from_option.output


def test_info_reports_a_bad_override() -> None:
    result = runner.invoke(app, ["info", "--set", "split.n_splits=1"])
    assert result.exit_code == 2
    assert "Configuration error" in result.output


def test_info_reports_an_unknown_key() -> None:
    result = runner.invoke(app, ["info", "--set", "mfcc.nonsense=1"])
    assert result.exit_code == 2
    assert "unknown configuration key" in result.output


def test_prepare_writes_the_index(cli_config, tmp_path) -> None:
    destination = tmp_path / "index.csv"
    result = runner.invoke(
        app,
        ["prepare", "-c", str(cli_config), "-o", str(destination), "--log-level", "ERROR"],
    )
    assert result.exit_code == 0, result.output
    assert destination.is_file()
    assert "Indexed 12 recording(s)" in result.output
    assert "Phonotrauma" in result.output


def test_prepare_reports_a_missing_corpus(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["prepare", "--data-root", str(tmp_path / "absent"), "--log-level", "ERROR"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_features_writes_the_frame_archive(cli_config, tmp_path) -> None:
    destination = tmp_path / "features.npz"
    result = runner.invoke(
        app,
        ["features", "-c", str(cli_config), "-o", str(destination), "--log-level", "ERROR"],
    )
    assert result.exit_code == 0, result.output
    assert destination.is_file()
    assert destination.with_suffix(".index.csv").is_file()
    assert "Extracted" in result.output


@pytest.mark.parametrize("route", ["detection", "disorder", "flat", "cascade"])
def test_every_route_trains_and_writes_its_artefacts(cli_config, tmp_path, route: str) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "--route",
            route,
            "-c",
            str(cli_config),
            "--run-id",
            f"run-{route}",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code == 0, result.output

    run_dir = tmp_path / "results" / f"run-{route}"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert any((run_dir / "figures").glob("*.png"))

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["seed"]["seed"] == 0
    assert metrics["n_files"] == 12
    assert "subject UAR" in result.output


def test_the_detection_figures_name_the_binary_classes(cli_config, monkeypatch) -> None:
    from voice_pathology_detection import cli

    drawn: dict[str, dict[int, str]] = {}

    def record(report, path, *, names=None, **kwargs):
        drawn[path.name] = dict(names)
        return path

    monkeypatch.setattr(cli, "plot_confusion_matrix", record)
    result = runner.invoke(
        app,
        [
            "train",
            "--route",
            "detection",
            "-c",
            str(cli_config),
            "--run-id",
            "names",
            "--no-save-models",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code == 0, result.output
    assert drawn["confusion_subject.png"] == {0: "Normal", 1: "Pathological"}
    assert drawn["confusion_frame.png"][1] == "Pathological"


def test_the_threshold_sweep_is_written_without_figures(cli_config, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "--route",
            "detection",
            "-c",
            str(cli_config),
            "--run-id",
            "bare",
            "--no-figures",
            "--no-save-models",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "results" / "bare"
    assert (run_dir / "threshold_sweep.csv").is_file()
    assert not (run_dir / "figures").exists()


def test_an_unknown_route_is_rejected(cli_config) -> None:
    result = runner.invoke(app, ["train", "--route", "magic", "-c", str(cli_config)])
    assert result.exit_code == 1
    assert "unknown route" in result.output


def test_training_saves_models_that_predict_can_use(cli_config, tmp_path) -> None:
    trained = runner.invoke(
        app,
        [
            "train",
            "--route",
            "cascade",
            "-c",
            str(cli_config),
            "--run-id",
            "c1",
            "--no-figures",
            "--log-level",
            "ERROR",
        ],
    )
    assert trained.exit_code == 0, trained.output

    run_dir = tmp_path / "results" / "c1"
    assert (run_dir / "detection.joblib").is_file()
    assert (run_dir / "disorder.joblib").is_file()

    # The synthetic corpus has no separate test split, so the training material
    # is scored again; this exercises the inference path end to end.
    predicted = runner.invoke(
        app,
        [
            "predict",
            str(run_dir),
            "-c",
            str(cli_config),
            "--split",
            "train",
            "-o",
            str(tmp_path / "pred.csv"),
            "--log-level",
            "ERROR",
        ],
    )
    assert predicted.exit_code == 0, predicted.output

    from voice_pathology_detection.prediction_table import PREDICTION_COLUMN, read_prediction_table

    table = read_prediction_table(tmp_path / "pred.csv")
    assert len(table) == 12
    assert set(table[PREDICTION_COLUMN].unique()) <= {0, 1, 2, 3}

    # A detector alone cannot name a disorder, so the folder is refused before
    # scoring, whether or not any recording would have been flagged.
    (run_dir / "disorder.joblib").unlink()
    refused = runner.invoke(
        app,
        [
            "predict",
            str(run_dir),
            "-c",
            str(cli_config),
            "--split",
            "train",
            "--log-level",
            "ERROR",
        ],
    )
    assert refused.exit_code == 1
    assert "disorder.joblib" in refused.output


def test_predict_starts_from_the_configuration_of_the_run(cli_config, tmp_path) -> None:
    trained = runner.invoke(
        app,
        [
            "train",
            "--route",
            "flat",
            "-c",
            str(cli_config),
            "--run-id",
            "f2",
            "--no-figures",
            "--log-level",
            "ERROR",
        ],
    )
    assert trained.exit_code == 0, trained.output
    run_dir = tmp_path / "results" / "f2"

    from voice_pathology_detection.prediction_table import read_prediction_table

    # No --config: the 25 ms front end the models were trained on comes from
    # the run folder, not the 30 ms built-in default.
    common = ["--split", "train", "--log-level", "ERROR"]
    implicit = runner.invoke(app, ["predict", str(run_dir), "-o", str(tmp_path / "a.csv"), *common])
    explicit = runner.invoke(
        app,
        ["predict", str(run_dir), "-c", str(cli_config), "-o", str(tmp_path / "b.csv"), *common],
    )
    assert implicit.exit_code == 0, implicit.output
    assert explicit.exit_code == 0, explicit.output
    assert read_prediction_table(tmp_path / "a.csv").equals(
        read_prediction_table(tmp_path / "b.csv")
    )

    mismatched = runner.invoke(app, ["predict", str(run_dir), "--set", "mfcc.frame_ms=30", *common])
    assert mismatched.exit_code == 1
    assert "mfcc.frame_ms" in mismatched.output


def test_predict_reports_a_run_without_models(cli_config, tmp_path) -> None:
    empty = tmp_path / "empty-run"
    empty.mkdir()
    result = runner.invoke(app, ["predict", str(empty), "-c", str(cli_config)])
    assert result.exit_code == 1
    assert "--save-models" in result.output


def test_the_flat_route_can_be_used_for_inference(cli_config, tmp_path) -> None:
    trained = runner.invoke(
        app,
        [
            "train",
            "--route",
            "flat",
            "-c",
            str(cli_config),
            "--run-id",
            "f1",
            "--no-figures",
            "--log-level",
            "ERROR",
        ],
    )
    assert trained.exit_code == 0, trained.output
    run_dir = tmp_path / "results" / "f1"
    assert (run_dir / "flat.joblib").is_file()

    predicted = runner.invoke(
        app,
        [
            "predict",
            str(run_dir),
            "-c",
            str(cli_config),
            "--split",
            "train",
            "-o",
            str(tmp_path / "flat.csv"),
            "--log-level",
            "ERROR",
        ],
    )
    assert predicted.exit_code == 0, predicted.output
    assert (tmp_path / "flat.csv").is_file()


def test_command_line_options_override_the_file(cli_config, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "--route",
            "detection",
            "-c",
            str(cli_config),
            "--run-id",
            "o1",
            "--folds",
            "3",
            "--normalization",
            "cvn_block",
            "--model",
            "gmm",
            "--no-figures",
            "--no-save-models",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code == 0, result.output

    from voice_pathology_detection.config import Config

    saved = Config.from_yaml(tmp_path / "results" / "o1" / "config.yaml")
    assert saved.split.n_splits == 3
    assert saved.normalization.method == "cvn_block"
    assert saved.detection.model.name == "gmm"


def test_switching_model_drops_the_old_model_parameters(corpus_config, tmp_path) -> None:
    """``--model`` must not hand the new estimator the old one's hyper-parameters."""
    from voice_pathology_detection.config import Config

    # A network topology, which a support vector machine cannot accept.
    config_file = corpus_config.override(
        {
            "detection.model.name": "mlp",
            "detection.model.params": {"hidden_layer_sizes": [4], "max_iter": 5},
        }
    ).to_yaml(tmp_path / "mlp.yaml")

    result = runner.invoke(
        app,
        [
            "train",
            "--route",
            "detection",
            "-c",
            str(config_file),
            "--run-id",
            "swap",
            "--model",
            "svm",
            "--no-figures",
            "--no-save-models",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code == 0, result.output

    saved = Config.from_yaml(tmp_path / "results" / "swap" / "config.yaml")
    assert saved.detection.model.name == "svm"
    assert saved.detection.model.params == {}


def test_switching_model_keeps_parameters_given_on_the_command_line(
    corpus_config, tmp_path
) -> None:
    """Parameters the caller supplies explicitly survive the model swap."""
    from voice_pathology_detection.config import Config

    config_file = corpus_config.override(
        {
            "detection.model.name": "mlp",
            "detection.model.params": {"hidden_layer_sizes": [4], "max_iter": 5},
        }
    ).to_yaml(tmp_path / "mlp.yaml")

    result = runner.invoke(
        app,
        [
            "train",
            "--route",
            "detection",
            "-c",
            str(config_file),
            "--run-id",
            "kept",
            "--model",
            "gmm",
            "--set",
            "detection.model.params={n_components: 1}",
            "--no-figures",
            "--no-save-models",
            "--log-level",
            "INFO",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--model=gmm replaces detection.model.name" in result.output

    saved = Config.from_yaml(tmp_path / "results" / "kept" / "config.yaml")
    assert saved.detection.model.name == "gmm"
    assert saved.detection.model.params == {"n_components": 1}


def test_single_parameters_can_be_given_for_the_new_model(corpus_config, tmp_path) -> None:
    from voice_pathology_detection.config import Config

    config_file = corpus_config.override(
        {
            "detection.model.name": "mlp",
            "detection.model.params": {"hidden_layer_sizes": [4], "max_iter": 5},
        }
    ).to_yaml(tmp_path / "mlp.yaml")

    result = runner.invoke(
        app,
        [
            "train",
            "--route",
            "detection",
            "-c",
            str(config_file),
            "--run-id",
            "single",
            "--model",
            "svm",
            "--set",
            "detection.model.params.C=2",
            "--no-figures",
            "--no-save-models",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code == 0, result.output

    saved = Config.from_yaml(tmp_path / "results" / "single" / "config.yaml")
    assert saved.detection.model.name == "svm"
    assert saved.detection.model.params == {"C": 2}


def test_spectrograms_are_drawn(synthetic_corpus, tmp_path) -> None:
    recordings = sorted((synthetic_corpus / "train").rglob("*.wav"))[:2]
    destination = tmp_path / "spec.png"
    result = runner.invoke(
        app,
        [
            "spectrograms",
            *[str(path) for path in recordings],
            "-o",
            str(destination),
            "--n-fft",
            "256",
            "--noverlap",
            "200",
            "--max-freq",
            "3500",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code == 0, result.output
    assert destination.is_file()


def test_spectrograms_report_a_missing_recording(tmp_path) -> None:
    result = runner.invoke(app, ["spectrograms", str(tmp_path / "absent.wav")])
    assert result.exit_code == 1
    assert "no such recording" in result.output


def test_the_demonstration_runs_without_any_corpus(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "demo",
            "--out-dir",
            str(tmp_path / "demo"),
            "--per-class",
            "4",
            "--folds",
            "2",
            "--duration",
            "0.5",
            "--sample-rate",
            "8000",
            "--log-level",
            "ERROR",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Demonstration finished" in result.output

    runs = list((tmp_path / "demo").glob("demo-*"))
    assert len(runs) == 1
    metrics = json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["seed"]["seed"] == 0
    assert any((runs[0] / "figures").glob("*.png"))
