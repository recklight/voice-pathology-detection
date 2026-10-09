"""Command line interface.

An experiment is one command plus a YAML file.  ``--set`` and the other options
override the file, and the configuration that actually ran is written into the
run folder next to its results.

Commands:

``info``
    Show the resolved configuration and the models the registry offers.
``prepare``
    Index a corpus and write the file list, with the per-class counts checked.
``features``
    Extract and cache the frame matrix of an indexed corpus.
``train``
    Cross-validate one route (detection, disorders, the four-way cascade or
    the single-stage four-way classifier) and write metrics and figures.
``predict``
    Apply saved models to new recordings and write a prediction table.
``spectrograms``
    Draw narrow-band spectrograms of chosen recordings.
``demo``
    Run the whole pipeline end to end on synthetic audio, with no corpus.

Examples
--------
The demo needs no data::

    voice-pathology-detection demo --out-dir results/demo

Cross-validate the detector on a real corpus::

    voice-pathology-detection train --route detection --config configs/detection_dnn.yaml \\
        --set paths.data_root=/data/corpus
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
import yaml

from .config import Config, ConfigError
from .data.dataset import DatasetError, build_index, summarise_index
from .data.labels import BINARY_NAMES, CLASS_NAMES
from .data.synthetic import write_synthetic_corpus
from .eval.metrics import folds_to_frame
from .eval.plots import (
    plot_confusion_matrix,
    plot_fold_errorbar,
    plot_roc,
    plot_spectrogram_grid,
    plot_threshold_curve,
)
from .eval.report import new_run_id, save_run
from .eval.threshold import threshold_sweep
from .features.pipeline import FeatureSet, extract_corpus
from .logging_utils import configure_logging
from .models.registry import MODEL_DESCRIPTIONS, available_models
from .prediction_table import write_prediction_table
from .routes.cascade import CascadeResult, run_cascade
from .routes.detection import DETECTION_LABELS, fit_detection, run_detection
from .routes.disorder import fit_disorder, run_disorder
from .routes.flat import fit_flat, run_flat
from .routes.stage import StageModel, StageResult
from .seed import seed_record

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    name="voice-pathology-detection",
    help="Pathological voice detection and classification from sustained vowels.",
    no_args_is_help=True,
    add_completion=False,
)

#: Routes the ``train`` command understands.
ROUTES: tuple[str, ...] = ("detection", "disorder", "flat", "cascade")

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="YAML configuration file to start from."),
]
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override one setting, as key=value, e.g. --set split.n_splits=3. Repeatable.",
    ),
]
SeedOption = Annotated[int | None, typer.Option("--seed", help="Override the run seed.")]
LogOption = Annotated[
    str | None,
    typer.Option(
        "--log-level",
        help="DEBUG, INFO, WARNING, ERROR or CRITICAL; default: log_level of the "
        "configuration, or INFO.",
    ),
]
# spectrograms and demo read no configuration, so INFO is their only default.
StandaloneLogOption = Annotated[
    str | None,
    typer.Option("--log-level", help="DEBUG, INFO, WARNING, ERROR or CRITICAL; default INFO."),
]
DataRootOption = Annotated[
    Path | None, typer.Option("--data-root", help="Override paths.data_root.")
]


def _parse_overrides(entries: list[str] | None) -> dict[str, Any]:
    """Turn ``key=value`` strings into a dotted-key override mapping.

    Values are parsed as YAML, so numbers, booleans and lists are typed rather
    than left as strings.
    """
    overrides: dict[str, Any] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise typer.BadParameter(
                f"expected key=value but got {entry!r}, for example split.n_splits=3"
            )
        key, _, raw = entry.partition("=")
        try:
            overrides[key.strip()] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(f"cannot parse the value of {key!r}: {exc}") from exc
    return overrides


def _load_config(
    config: Path | None,
    overrides: list[str] | None,
    *,
    seed: int | None = None,
    data_root: Path | None = None,
    log_level: str | None = None,
) -> Config:
    """Build the configuration for a command and set up logging."""
    mapping = _parse_overrides(overrides)
    if seed is not None:
        mapping["seed"] = int(seed)
    if data_root is not None:
        mapping["paths.data_root"] = str(data_root)
    # Without --log-level the file decides; --set log_level=... beats both.
    if log_level is not None:
        mapping.setdefault("log_level", log_level.upper())

    try:
        cfg = (
            Config.from_yaml(config, overrides=mapping)
            if config is not None
            else Config().override(mapping)
        )
    except (ConfigError, FileNotFoundError) as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    configure_logging(cfg.log_level)
    return cfg


def _fail(message: str, exc: Exception | None = None) -> None:
    """Report a user-facing error and exit with a non-zero status."""
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


def _load_features(
    cfg: Config,
    *,
    split: str = "train",
    strict: bool = False,
    expected_counts: dict[int, int] | None = None,
) -> FeatureSet:
    """Index the corpus and extract its features, or exit with the reason."""
    try:
        index = build_index(cfg, split=split, strict_counts=strict, expected_counts=expected_counts)
        return extract_corpus(index, cfg, cache_dir=cfg.paths.cache_dir)
    except (DatasetError, ValueError) as exc:
        _fail(str(exc), exc)
        raise  # pragma: no cover - _fail always raises


def _model_overrides(route: str, model: str, set_: list[str] | None) -> list[str]:
    """Build the overrides that switch a route to a different estimator.

    The hyper-parameters in a configuration file belong to the model that file
    names.  Keeping them while swapping the name would hand, say, a support
    vector machine the layer sizes of a network, and the fit would fail.  So
    they are cleared together with the name and the new estimator starts from
    its own defaults, unless a whole mapping is given with
    ``--set <section>.model.params``.  Single keys such as
    ``--set detection.model.params.C=2`` land in the cleared mapping, because
    ``train`` applies the ``--set`` entries after these overrides.

    Parameters
    ----------
    route : str
        Route being trained, which decides the stage or stages affected.
    model : str
        Name of the estimator to use instead.
    set_ : list of str or None
        Raw ``--set`` entries, inspected for explicitly given parameters.

    Returns
    -------
    list of str
        Dotted-key override strings to apply before the ``--set`` entries.

    Examples
    --------
    >>> _model_overrides("detection", "svm", None)
    ['detection.model.name=svm', 'detection.model.params={}']
    >>> _model_overrides("detection", "svm", ["detection.model.params={C: 2}"])
    ['detection.model.name=svm']
    >>> _model_overrides("detection", "svm", ["detection.model.params.C=2"])
    ['detection.model.name=svm', 'detection.model.params={}']
    >>> _model_overrides("flat", "svm", None)
    ['flat.model.name=svm', 'flat.model.params={}']
    >>> _model_overrides("cascade", "gmm", None)[::2]
    ['detection.model.name=gmm', 'disorder.model.name=gmm']
    """
    given = {entry.partition("=")[0].strip() for entry in set_ or []}
    overrides: list[str] = []
    for section in _route_sections(route):
        prefix = f"{section}.model.params"
        overrides.append(f"{section}.model.name={model}")
        if prefix not in given:
            overrides.append(f"{prefix}={{}}")
    return overrides


def _route_sections(route: str) -> tuple[str, ...]:
    """Return the configuration sections whose model a route trains."""
    return ("detection", "disorder") if route == "cascade" else (route,)


@app.command()
def info(
    config: ConfigOption = None,
    set_: SetOption = None,
    log_level: LogOption = None,
) -> None:
    """Show the resolved configuration and the available models."""
    cfg = _load_config(config, set_, log_level=log_level)
    typer.echo("Resolved configuration:\n")
    typer.echo(yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False))
    typer.echo("Available models:")
    for name in available_models():
        typer.echo(f"  {name:<12} {MODEL_DESCRIPTIONS.get(name, '')}")
    typer.echo("\nClasses:")
    for code, name in CLASS_NAMES.items():
        typer.echo(f"  {code} = {name}")


@app.command()
def prepare(
    config: ConfigOption = None,
    set_: SetOption = None,
    data_root: DataRootOption = None,
    split: Annotated[str, typer.Option(help="Which split to index: train or test.")] = "train",
    strict: Annotated[
        bool, typer.Option("--strict/--no-strict", help="Fail if the class counts differ.")
    ] = False,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Where to write the index CSV.")
    ] = None,
    log_level: LogOption = None,
) -> None:
    """Index a corpus and write the file list as CSV."""
    cfg = _load_config(config, set_, data_root=data_root, log_level=log_level)
    try:
        index = build_index(cfg, split=split, strict_counts=strict)
    except (DatasetError, ValueError) as exc:
        _fail(str(exc), exc)
        return

    destination = output or (cfg.paths.results_dir / f"index_{split}.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(destination, index=False)

    typer.echo(f"Indexed {len(index)} recording(s) -> {destination}")
    if split == "train":
        for row in summarise_index(index).itertuples(index=False):
            typer.echo(f"  {row.label_name:<14} {int(row.n_files)}")


@app.command()
def features(
    config: ConfigOption = None,
    set_: SetOption = None,
    data_root: DataRootOption = None,
    split: Annotated[str, typer.Option(help="Which split to analyse.")] = "train",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Where to write the .npz archive.")
    ] = None,
    log_level: LogOption = None,
) -> None:
    """Extract the frame matrix of a corpus and cache it."""
    cfg = _load_config(config, set_, data_root=data_root, log_level=log_level)
    feature_set = _load_features(cfg, split=split)
    destination = output or (cfg.paths.cache_dir / f"features_{split}.npz")
    feature_set.save(destination)
    typer.echo(
        f"Extracted {feature_set.n_frames} frame(s) of {feature_set.n_dim} dimension(s) "
        f"from {feature_set.n_files} recording(s) -> {destination}"
    )


def _display_names(labels: tuple[int, ...]) -> dict[int, str]:
    """Axis labels for a confusion matrix over ``labels``.

    Examples
    --------
    >>> _display_names((0, 1))[1]
    'Pathological'
    >>> _display_names((1, 2, 3))[1]
    'Neoplasm'
    """
    table = BINARY_NAMES if tuple(labels) == DETECTION_LABELS else CLASS_NAMES
    return {int(code): name for code, name in table.items()}


def _write_threshold_sweep(
    result: StageResult | CascadeResult, run_dir: Path
) -> pd.DataFrame | None:
    """Write ``threshold_sweep.csv`` for a two-class decision and return the table.

    The detection score of every subject is swept over the default grid.  A
    multi-class stage has no such score, and ``None`` is returned.
    """
    if isinstance(result, CascadeResult):
        truth = result.predictions["y_true_binary"].to_numpy()
    elif len(result.labels) == 2:
        truth = result.predictions["y_true"].to_numpy()
    else:
        return None
    sweep = threshold_sweep(truth, result.predictions["score"].to_numpy())
    sweep.to_csv(run_dir / "threshold_sweep.csv", index=False)
    return sweep


def _mean_threshold(thresholds: list[float]) -> float | None:
    """Mean of the per-fold thresholds, for the marker in the threshold figure."""
    return float(sum(thresholds) / len(thresholds)) if thresholds else None


def _save_stage_figures(
    result: StageResult, run_dir: Path, sweep: pd.DataFrame | None = None
) -> None:
    """Draw the standard figures of a single-stage result."""
    figures = run_dir / "figures"
    names = _display_names(result.labels)
    plot_confusion_matrix(
        result.pooled_subject,
        figures / "confusion_subject.png",
        names=names,
        title=f"{result.task}: subject level",
    )
    plot_confusion_matrix(
        result.pooled_frame,
        figures / "confusion_frame.png",
        names=names,
        title=f"{result.task}: frame level",
    )
    plot_fold_errorbar(
        folds_to_frame([outcome.subject_metrics for outcome in result.folds]),
        figures / "folds_subject.png",
        title=f"{result.task}: subject level across folds",
    )
    if sweep is not None:
        plot_threshold_curve(
            sweep,
            figures / "threshold.png",
            chosen=_mean_threshold(result.thresholds),
            title=f"{result.task}: operating point",
        )
        plot_roc(
            result.predictions["y_true"].to_numpy(),
            result.predictions["score"].to_numpy(),
            figures / "roc.png",
            title=f"{result.task}: ROC",
        )


def _save_cascade_figures(result: CascadeResult, run_dir: Path, sweep: pd.DataFrame) -> None:
    """Draw the standard figures of a cascade result."""
    figures = run_dir / "figures"
    plot_confusion_matrix(
        result.pooled_overall,
        figures / "confusion_four_way.png",
        names={int(code): name for code, name in CLASS_NAMES.items()},
        title="Cascade: merged four-way decision",
    )
    plot_confusion_matrix(
        result.pooled_detection,
        figures / "confusion_detection.png",
        names=_display_names(DETECTION_LABELS),
        title="Cascade: detection stage",
    )
    plot_threshold_curve(
        sweep,
        figures / "threshold.png",
        chosen=_mean_threshold(result.thresholds),
        title="Cascade: detection threshold",
    )
    plot_roc(
        result.predictions["y_true_binary"].to_numpy(),
        result.predictions["score"].to_numpy(),
        figures / "roc.png",
        title="Cascade: detection ROC",
    )


@app.command()
def train(
    route: Annotated[
        str, typer.Option("--route", "-r", help=f"One of {', '.join(ROUTES)}.")
    ] = "cascade",
    config: ConfigOption = None,
    set_: SetOption = None,
    data_root: DataRootOption = None,
    seed: SeedOption = None,
    folds: Annotated[int | None, typer.Option("--folds", help="Override split.n_splits.")] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the estimator of the route.")
    ] = None,
    normalization: Annotated[
        str | None, typer.Option("--normalization", help="Override normalization.method.")
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Name of the run folder.")] = None,
    save_models: Annotated[
        bool,
        typer.Option(
            "--save-models/--no-save-models",
            help="Also fit and store the models that predict uses.",
        ),
    ] = True,
    figures: Annotated[
        bool, typer.Option("--figures/--no-figures", help="Draw the standard figures.")
    ] = True,
    log_level: LogOption = None,
) -> None:
    """Cross-validate one route and write its metrics, figures and models."""
    if route not in ROUTES:
        _fail(f"unknown route {route!r}; choose one of {list(ROUTES)}")

    extra = [] if model is None else _model_overrides(route, model, set_)
    extra.extend(set_ or [])
    if folds is not None:
        extra.append(f"split.n_splits={folds}")
    if normalization is not None:
        extra.append(f"normalization.method={normalization}")

    cfg = _load_config(config, extra, seed=seed, data_root=data_root, log_level=log_level)
    if model is not None:
        # Logged here because _load_config is what sets up logging.
        for section in _route_sections(route):
            if f"{section}.model.params={{}}" not in extra:
                _LOGGER.info(
                    "--model=%s replaces %s.model.name; the parameters given with --set are kept",
                    model,
                    section,
                )
    record = seed_record(cfg.seed)
    feature_set = _load_features(cfg)

    try:
        if route == "detection":
            result: StageResult | CascadeResult = run_detection(feature_set, cfg)
        elif route == "disorder":
            result = run_disorder(feature_set, cfg)
        elif route == "flat":
            result = run_flat(feature_set, cfg)
        else:
            result = run_cascade(feature_set, cfg)
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc), exc)
        return

    payload = dict(result.to_dict())
    payload["seed"] = record.to_dict()
    payload["n_files"] = feature_set.n_files
    payload["n_frames"] = feature_set.n_frames

    tables = {"predictions": result.predictions}
    if isinstance(result, StageResult):
        tables["folds_subject"] = folds_to_frame(
            [outcome.subject_metrics for outcome in result.folds]
        )
        tables["folds_frame"] = folds_to_frame([outcome.frame_metrics for outcome in result.folds])

    run_dir = save_run(
        cfg.paths.results_dir,
        cfg,
        payload,
        run_id=run_id or new_run_id(route, cfg.paths.results_dir),
        tables=tables,
        title=f"{route} report",
    )
    sweep = _write_threshold_sweep(result, run_dir)

    if figures:
        if isinstance(result, CascadeResult):
            _save_cascade_figures(result, run_dir, sweep)
        else:
            _save_stage_figures(result, run_dir, sweep)

    if save_models:
        try:
            if route in {"detection", "cascade"}:
                fit_detection(feature_set, cfg).save(run_dir / "detection.joblib")
            if route in {"disorder", "cascade"}:
                fit_disorder(feature_set, cfg).save(run_dir / "disorder.joblib")
            if route == "flat":
                fit_flat(feature_set, cfg).save(run_dir / "flat.joblib")
        except (RuntimeError, ValueError) as exc:
            _LOGGER.warning("could not fit the model(s) for predict: %s", exc)

    summary = (
        result.overall_summary if isinstance(result, CascadeResult) else result.subject_summary
    )
    typer.echo(f"\nRoute '{route}' finished. Artefacts in {run_dir}")
    typer.echo(f"  subject UAR      {summary['uar']['mean']:.4f} +/- {summary['uar']['std']:.4f}")
    typer.echo(
        f"  subject accuracy {summary['accuracy']['mean']:.4f} +/- {summary['accuracy']['std']:.4f}"
    )


def _recording_ids(feature_set: FeatureSet) -> list[str]:
    """Pick identifiers for a prediction table.

    A test set is usually one flat folder, so the file name identifies the
    recording.  A corpus organised one folder per diagnosis repeats names such
    as ``001.wav`` across folders, and a table keyed on them would collide; the
    subject identifier, which carries the folder, is used instead.
    """
    names = [str(name) for name in feature_set.files["file_name"]]
    stems = [Path(name).stem for name in names]
    if len(set(stems)) == len(stems):
        return names
    _LOGGER.info(
        "file names repeat across folders; using the subject identifiers in the prediction table"
    )
    # Flattened, because the prediction table strips directories from its keys.
    return [
        str(subject).replace("/", "_").replace("\\", "_")
        for subject in feature_set.files["subject_id"]
    ]


def _predict_config(
    run_dir: Path,
    config: Path | None,
    overrides: list[str] | None,
    *,
    data_root: Path | None,
    log_level: str | None,
) -> Config:
    """Build the configuration of ``predict`` from the run folder.

    The run's own ``config.yaml`` is the starting point unless ``--config``
    names another file.  The saved models only make sense on features computed
    the way they were trained on, so any ``mfcc`` or ``trim`` setting that
    differs from that file is refused instead of being scored silently.
    """
    trained_path = run_dir / "config.yaml"
    if config is None and trained_path.is_file():
        config = trained_path
    cfg = _load_config(config, overrides, data_root=data_root, log_level=log_level)
    if not trained_path.is_file():
        _LOGGER.warning(
            "%s has no config.yaml; the front end of the saved models cannot be checked",
            run_dir,
        )
        return cfg
    try:
        trained = Config.from_yaml(trained_path).to_dict()
    except (ConfigError, FileNotFoundError) as exc:
        _fail(f"cannot read {trained_path}: {exc}", exc)
    current = cfg.to_dict()
    differing = [
        f"{section}.{key}"
        for section in ("mfcc", "trim")
        for key, value in current[section].items()
        if trained[section].get(key) != value
    ]
    if differing:
        _fail(
            f"{differing} differ from {trained_path}, which the saved models were trained "
            "with; drop --config and the mfcc/trim overrides, or train again"
        )
    return cfg


@app.command()
def predict(
    run_dir: Annotated[Path, typer.Argument(help="Run folder holding the saved models.")],
    config: ConfigOption = None,
    set_: SetOption = None,
    data_root: DataRootOption = None,
    split: Annotated[str, typer.Option(help="Which split to score.")] = "test",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Prediction table, .xlsx or .csv."),
    ] = None,
    log_level: LogOption = None,
) -> None:
    """Apply saved models to new recordings and write a prediction table.

    The configuration starts from the run folder's config.yaml, so only the
    data location usually needs to be given.
    """
    cfg = _predict_config(run_dir, config, set_, data_root=data_root, log_level=log_level)

    flat_path = run_dir / "flat.joblib"
    detection_path = run_dir / "detection.joblib"
    disorder_path = run_dir / "disorder.joblib"
    if not flat_path.is_file():
        missing = [path.name for path in (detection_path, disorder_path) if not path.is_file()]
        if missing:
            _fail(
                f"{run_dir} holds no flat.joblib and lacks {' and '.join(missing)}; point "
                "predict at a run of --route flat or --route cascade trained with --save-models"
            )

    try:
        index = build_index(cfg, split=split)
        feature_set = extract_corpus(index, cfg, cache_dir=cfg.paths.cache_dir)
    except (DatasetError, ValueError) as exc:
        _fail(str(exc), exc)
        return

    if flat_path.is_file():
        predictions = StageModel.load(flat_path).predict_subjects(feature_set).pred
    else:
        detector = StageModel.load(detection_path)
        forwarded = detector.predict_subjects(feature_set).pred == detector.positive_label
        # Everyone starts as Normal; only the forwarded subjects get a disorder.
        predictions = np.zeros(feature_set.n_files, dtype=int)

        if forwarded.any():
            disorder = StageModel.load(disorder_path)
            subset = feature_set.select(forwarded)
            predictions[forwarded] = disorder.predict_subjects(subset).pred

    destination = output or (run_dir / f"predictions_{split}.xlsx")
    write_prediction_table(destination, _recording_ids(feature_set), predictions)
    typer.echo(f"Wrote {feature_set.n_files} prediction(s) -> {destination}")


@app.command()
def spectrograms(
    files: Annotated[list[Path], typer.Argument(help="Recordings to draw.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Image file to write.")] = Path(
        "results/figures/spectrograms.png"
    ),
    n_fft: Annotated[int, typer.Option(help="Analysis window length in samples.")] = 1024,
    noverlap: Annotated[int, typer.Option(help="Window overlap in samples.")] = 1000,
    max_freq: Annotated[
        float, typer.Option(help="Upper limit of the frequency axis, Hz.")
    ] = 8000.0,
    log_level: StandaloneLogOption = None,
) -> None:
    """Draw narrow-band spectrograms of the given recordings."""
    configure_logging(log_level or "INFO")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        _fail(f"no such recording(s): {missing}")
    try:
        written = plot_spectrogram_grid(
            files, output, n_fft=n_fft, noverlap=noverlap, max_freq=max_freq
        )
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc), exc)
        return
    typer.echo(f"Wrote {written}")


@app.command()
def demo(
    out_dir: Annotated[
        Path, typer.Option("--out-dir", "-o", help="Where to write the run artefacts.")
    ] = Path("results"),
    per_class: Annotated[int, typer.Option(help="Synthetic recordings per class.")] = 8,
    duration: Annotated[float, typer.Option(help="Duration of each recording, seconds.")] = 1.0,
    sample_rate: Annotated[int, typer.Option(help="Sampling rate of the synthetic audio.")] = 16000,
    folds: Annotated[int, typer.Option(help="Cross-validation folds.")] = 2,
    seed: Annotated[int, typer.Option(help="Run seed.")] = 0,
    log_level: StandaloneLogOption = None,
) -> None:
    """Run the whole pipeline on synthetic audio; no corpus needed."""
    configure_logging(log_level or "INFO")
    with tempfile.TemporaryDirectory(prefix="voice-pathology-detection-demo-") as workspace:
        root = Path(workspace)
        write_synthetic_corpus(
            root,
            dict.fromkeys(CLASS_NAMES, per_class),
            sr=sample_rate,
            duration=duration,
            seed=seed,
        )
        cfg = Config().override(
            {
                "paths.data_root": str(root),
                "paths.results_dir": str(out_dir),
                "paths.cache_dir": str(root / "cache"),
                "seed": seed,
                "split.n_splits": folds,
                "split.val_fraction": 0.5,
                "mfcc.sample_rate": sample_rate,
                "mfcc.frame_ms": 25.0,
                "mfcc.hop_ms": 10.0,
                "mfcc.high_freq": min(7800.0, sample_rate / 2.0 - 100.0),
                "trim.mode": "none",
                "normalization.method": "cmvn",
                # A few hundred frames need a larger step and more epochs than
                # the hundreds of thousands a real corpus provides; early
                # stopping is off because there is nothing to hold out.
                "detection.model.name": "mlp",
                "detection.model.params": {
                    "hidden_layer_sizes": [64, 64],
                    "max_iter": 300,
                    "batch_size": 64,
                    "learning_rate_init": 0.01,
                    "momentum": 0.9,
                    "early_stopping": False,
                },
                "disorder.model.name": "mlp",
                "disorder.model.params": {
                    "hidden_layer_sizes": [64, 64],
                    "max_iter": 300,
                    "batch_size": 64,
                    "learning_rate_init": 0.01,
                    "momentum": 0.9,
                    "early_stopping": False,
                },
            }
        )
        record = seed_record(cfg.seed)
        feature_set = _load_features(cfg, expected_counts=dict.fromkeys(CLASS_NAMES, per_class))
        result = run_cascade(feature_set, cfg)

        run_dir = save_run(
            cfg.paths.results_dir,
            cfg,
            {**result.to_dict(), "n_files": feature_set.n_files, "seed": record.to_dict()},
            run_id=new_run_id("demo", cfg.paths.results_dir),
            tables={"predictions": result.predictions},
            title="Synthetic demonstration",
        )
        _save_cascade_figures(result, run_dir, _write_threshold_sweep(result, run_dir))

    typer.echo(f"Demonstration finished. Artefacts in {run_dir}")
    typer.echo(
        f"  four-way UAR {result.overall_summary['uar']['mean']:.4f} "
        f"+/- {result.overall_summary['uar']['std']:.4f}"
    )
    typer.echo(f"  detection sensitivity {result.detection_summary['sensitivity']['mean']:.4f}")


def main() -> None:
    """Entry point of the ``voice-pathology-detection`` command."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        typer.secho("Interrupted.", fg=typer.colors.YELLOW, err=True)
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
