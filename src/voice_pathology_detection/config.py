"""Configuration dataclasses for the whole project.

Paths, front-end parameters, the normalisation strategy, model
hyper-parameters, the cross-validation layout and the decision thresholds are
all fields of the dataclasses below, so an experiment is changed in a YAML file
and not in the code.  A configuration is loaded from YAML, overridden with
dotted keys from the command line, and saved next to the results of a run, so
every run folder holds the settings needed to repeat it.
``configs/default.yaml`` writes out the defaults of these classes in full.

Examples
--------
>>> cfg = Config()
>>> cfg.mfcc.n_dim
39
>>> cfg.override({"split.n_splits": 3}).split.n_splits
3
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

#: Normalisation strategies accepted by :class:`NormalizationConfig`.
NORMALIZATION_METHODS: tuple[str, ...] = (
    "none",
    "cvn_block",
    "cmn",
    "cvn",
    "cmvn",
    "cmvn_per_utt",
)

#: Waveform trimming strategies accepted by :class:`TrimConfig`.
TRIM_MODES: tuple[str, ...] = ("none", "fixed_edges", "energy_vad")

#: Frame-to-subject aggregation rules accepted by the pipeline configurations.
AGGREGATION_MODES: tuple[str, ...] = ("mean_proba", "majority_vote", "ratio_threshold")

#: Threshold selection rules accepted by :class:`DetectionConfig`.
THRESHOLD_METHODS: tuple[str, ...] = (
    "fixed",
    "youden",
    "target_sensitivity",
    "target_specificity",
)

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigError(ValueError):
    """Raised when a configuration value is missing, unknown or out of range."""


def _check_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Raise :class:`ConfigError`, listing ``allowed``, unless ``value`` is one of them."""
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; choose one of {list(allowed)}")


def _check_positive(name: str, value: float) -> None:
    """Validate that a numeric configuration entry is strictly positive."""
    if not value > 0:
        raise ConfigError(f"{name} must be positive, got {value!r}")


def _check_multiclass_aggregation(section: str, aggregation: str) -> None:
    """Validate the frame-to-subject rule of a stage with more than two classes."""
    _check_choice(f"{section}.aggregation", aggregation, AGGREGATION_MODES)
    if aggregation == "ratio_threshold":
        raise ConfigError(
            f"{section}.aggregation='ratio_threshold' is only defined for two classes"
        )


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Filesystem locations used by the pipeline.

    The defaults are placeholders.  No recording ships with the package, so
    ``data_root`` has to point at a local copy of the corpus before anything
    can be read.

    Attributes
    ----------
    data_root : pathlib.Path
        Root folder holding the corpus.
    train_dir, test_dir : str
        Sub-folders of ``data_root`` with the labelled and unlabelled material.
    demographics : str or None
        Optional spreadsheet with one row per training file, used to read the
        diagnosis of every recording instead of inferring it from the folder.
    cache_dir, results_dir : pathlib.Path
        Where cached feature matrices and per-run artefacts are written.
    """

    data_root: Path = Path("data/corpus")
    train_dir: str = "train"
    test_dir: str = "test"
    demographics: str | None = None
    cache_dir: Path = Path("cache")
    results_dir: Path = Path("results")

    def __post_init__(self) -> None:
        for name in ("data_root", "cache_dir", "results_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

    @property
    def train_root(self) -> Path:
        """Path of the training folder."""
        return self.data_root / self.train_dir

    @property
    def test_root(self) -> Path:
        """Path of the test folder."""
        return self.data_root / self.test_dir

    @property
    def demographics_path(self) -> Path | None:
        """Path of the demographics spreadsheet, or ``None`` when unused."""
        return None if self.demographics is None else self.data_root / self.demographics


@dataclass(frozen=True, slots=True)
class MFCCConfig:
    """Parameters of the cepstral front end.

    The defaults describe a 30 ms / 15 ms analysis of 44.1 kHz sustained vowels
    with 23 mel filters and 12 cepstral coefficients plus the zeroth term,
    expanded with first and second order regression deltas to 39 dimensions.

    Attributes
    ----------
    sample_rate : int
        Sampling rate the recordings are expected to have, in Hz.
    frame_ms, hop_ms : float
        Analysis frame length and advance in milliseconds.
    n_mels : int
        Number of triangular mel filters.
    n_ceps : int
        Number of cepstral coefficients kept besides ``c0``.
    lifter : int
        Sinusoidal cepstral lifter parameter; ``0`` disables liftering.
    preemph : float
        Pre-emphasis coefficient; ``0`` disables pre-emphasis.
    low_freq, high_freq : float or None
        Band limits in Hz applied to the waveform before the front end;
        ``None`` disables that side of the band limit.  The default is a 64 Hz
        high-pass and no low-pass.
    n_fft : int or None
        FFT size; ``None`` selects the smallest power of two not shorter than
        the frame.
    use_deltas : bool
        Whether the static coefficients are expanded with deltas.
    delta_width, accel_width : int
        Regression window lengths, in frames, of the first and second order
        differences.  Both must be odd and at least three.
    """

    sample_rate: int = 44100
    frame_ms: float = 30.0
    hop_ms: float = 15.0
    n_mels: int = 23
    n_ceps: int = 12
    lifter: int = 22
    preemph: float = 0.97
    low_freq: float | None = 64.0
    high_freq: float | None = None
    n_fft: int | None = None
    use_deltas: bool = True
    delta_width: int = 7
    accel_width: int = 5

    def __post_init__(self) -> None:
        _check_positive("mfcc.sample_rate", self.sample_rate)
        _check_positive("mfcc.frame_ms", self.frame_ms)
        _check_positive("mfcc.hop_ms", self.hop_ms)
        _check_positive("mfcc.n_mels", self.n_mels)
        _check_positive("mfcc.n_ceps", self.n_ceps)
        if self.n_ceps >= self.n_mels:
            raise ConfigError(
                f"mfcc.n_ceps={self.n_ceps} must be smaller than mfcc.n_mels={self.n_mels}"
            )
        if self.hop_ms > self.frame_ms:
            raise ConfigError(
                f"mfcc.hop_ms={self.hop_ms} must not exceed mfcc.frame_ms={self.frame_ms}"
            )
        if not 0.0 <= self.preemph < 1.0:
            raise ConfigError(f"mfcc.preemph must lie in [0, 1), got {self.preemph}")
        for name in ("delta_width", "accel_width"):
            width = getattr(self, name)
            if width < 3 or width % 2 == 0:
                raise ConfigError(f"mfcc.{name} must be an odd integer >= 3, got {width}")
        nyquist = self.sample_rate / 2.0
        low = 0.0 if self.low_freq is None else float(self.low_freq)
        high = nyquist if self.high_freq is None else float(self.high_freq)
        if not 0.0 <= low < high <= nyquist:
            raise ConfigError(
                "require 0 <= mfcc.low_freq < mfcc.high_freq <= sample_rate / 2, got "
                f"low_freq={self.low_freq}, high_freq={self.high_freq}"
            )
        if self.n_fft is not None and self.n_fft < self.frame_length:
            raise ConfigError(
                f"mfcc.n_fft={self.n_fft} is smaller than the frame of "
                f"{self.frame_length} samples, which would discard signal"
            )

    @property
    def frame_length(self) -> int:
        """Analysis frame length in samples."""
        return round(self.sample_rate * self.frame_ms / 1000.0)

    @property
    def resolved_n_fft(self) -> int:
        """FFT size used at :attr:`sample_rate`, defaulting to the next power of two."""
        # Imported here because the features package imports this module.
        from .features.mfcc import n_fft_for

        return n_fft_for(self.sample_rate, self)

    @property
    def n_static(self) -> int:
        """Number of static coefficients, ``n_ceps`` plus the zeroth term."""
        return self.n_ceps + 1

    @property
    def n_dim(self) -> int:
        """Total feature dimension after the optional delta expansion."""
        return self.n_static * (3 if self.use_deltas else 1)

    @property
    def band(self) -> tuple[float, float] | None:
        """Band limits as a pair, or ``None`` when the full band is kept."""
        if self.low_freq is None and self.high_freq is None:
            return None
        nyquist = self.sample_rate / 2.0
        low = 0.0 if self.low_freq is None else float(self.low_freq)
        high = nyquist if self.high_freq is None else float(self.high_freq)
        if low <= 0.0 and high >= nyquist:
            return None
        return low, high


@dataclass(frozen=True, slots=True)
class TrimConfig:
    """Waveform trimming applied before the front end.

    Attributes
    ----------
    mode : str
        One of ``none``, ``fixed_edges`` or ``energy_vad``.
    edge_seconds : float
        Amount removed from each end in ``fixed_edges`` mode.
    min_seconds : float
        Recordings of this length or shorter are left untouched by
        ``fixed_edges``, so a short prompt is never trimmed away.
    min_output_seconds : float
        Lower bound on the trimmed result; a trim that would leave less audio
        than this is discarded and the original waveform kept.
    frame_ms, hop_ms : float
        Analysis grid of the energy detector used by ``energy_vad``.
    """

    mode: str = "fixed_edges"
    edge_seconds: float = 2.0
    min_seconds: float = 4.5
    min_output_seconds: float = 0.5
    frame_ms: float = 25.0
    hop_ms: float = 10.0

    def __post_init__(self) -> None:
        _check_choice("trim.mode", self.mode, TRIM_MODES)
        if self.edge_seconds < 0:
            raise ConfigError(f"trim.edge_seconds must not be negative, got {self.edge_seconds}")
        _check_positive("trim.min_output_seconds", self.min_output_seconds)
        _check_positive("trim.frame_ms", self.frame_ms)
        _check_positive("trim.hop_ms", self.hop_ms)


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Cepstral normalisation strategy.

    Attributes
    ----------
    method : str
        One of :data:`NORMALIZATION_METHODS`.
    block_size : int
        Width of one static/delta/acceleration block, used by ``cvn_block``.
    divisor : str
        ``std`` or ``var``: whether ``cvn_block`` divides each block by its
        standard deviation or by its variance.
    eps : float
        Floor added to the divisor to keep constant blocks finite.
    """

    method: str = "cmvn"
    block_size: int = 13
    divisor: str = "std"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        _check_choice("normalization.method", self.method, NORMALIZATION_METHODS)
        _check_choice("normalization.divisor", self.divisor, ("std", "var"))
        _check_positive("normalization.block_size", self.block_size)
        _check_positive("normalization.eps", self.eps)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Cross-validation layout.

    Attributes
    ----------
    n_splits : int
        Number of outer folds.
    seed : int
        Seed of the fold shuffling, so a split can be reproduced exactly.
    group_by : str
        Column of the file index whose values must never straddle a fold
        boundary; grouping by subject keeps the frames of one speaker from
        appearing on both sides of a split.
    val_fraction : float
        Share of every training fold held out to tune the decision threshold,
        so that the evaluation fold is never used for model selection.  The
        held-out part is one fold of a grouped split into
        ``k = round(1 / val_fraction)`` parts, so the share is about ``1 / k``
        (0.2 holds out a fifth, 0.3 a third, 0.4 and 0.5 a half); values above
        0.5 are refused.  When a class has fewer than ``k`` groups on the
        training side, ``k`` drops to that count and more is held out.
    """

    n_splits: int = 5
    seed: int = 0
    group_by: str = "subject_id"
    val_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ConfigError(f"split.n_splits must be at least 2, got {self.n_splits}")
        if not 0.0 < self.val_fraction <= 0.5:
            raise ConfigError(f"split.val_fraction must lie in (0, 0.5], got {self.val_fraction}")


def _mlp_params(hidden_layer_sizes: list[int], batch_size: int) -> dict[str, Any]:
    """Default network of a stage: sigmoid layers trained for 30 epochs of SGD."""
    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "max_iter": 30,
        "batch_size": batch_size,
        "learning_rate_init": 0.001,
        "momentum": 0.5,
    }


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Name and hyper-parameters of an estimator.

    Attributes
    ----------
    name : str
        Key registered in :mod:`voice_pathology_detection.models.registry`.
    params : dict
        Keyword arguments forwarded to the factory of that estimator.
    """

    name: str = "mlp"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("model.name must not be empty")
        if not isinstance(self.params, dict):
            raise ConfigError(f"model.params must be a mapping, got {type(self.params).__name__}")


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Stage one: normal versus pathological detection.

    Attributes
    ----------
    model : ModelConfig
        Estimator trained on frame-level features.
    aggregation : str
        How frame scores become one decision per subject.
    threshold_method : str
        How the decision threshold is chosen on held-out training material.
    threshold : float or None
        Threshold used when ``threshold_method`` is ``fixed``.
    target : float or None
        Operating point requested by the ``target_sensitivity`` and
        ``target_specificity`` rules.
    """

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="mlp", params=_mlp_params([200] * 5, 90))
    )
    aggregation: str = "mean_proba"
    threshold_method: str = "youden"
    threshold: float | None = None
    target: float | None = 0.9

    def __post_init__(self) -> None:
        _check_choice("detection.aggregation", self.aggregation, AGGREGATION_MODES)
        _check_choice("detection.threshold_method", self.threshold_method, THRESHOLD_METHODS)
        if self.threshold_method == "fixed" and self.threshold is None:
            raise ConfigError("detection.threshold must be given when threshold_method='fixed'")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ConfigError(f"detection.threshold must lie in [0, 1], got {self.threshold}")
        if self.threshold_method.startswith("target") and self.target is None:
            raise ConfigError(
                f"detection.target must be given when threshold_method={self.threshold_method!r}"
            )


@dataclass(frozen=True, slots=True)
class DisorderConfig:
    """Stage two: which disorder, among the subjects flagged as pathological.

    Attributes
    ----------
    model : ModelConfig
        Estimator trained on the pathological subset.
    aggregation : str
        Frame-to-subject rule; the three-way decision never uses a threshold.
    """

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="mlp", params=_mlp_params([100] * 4, 80))
    )
    aggregation: str = "mean_proba"

    def __post_init__(self) -> None:
        _check_multiclass_aggregation("disorder", self.aggregation)


@dataclass(frozen=True, slots=True)
class FlatConfig:
    """The single-stage route: one classifier over all four classes.

    Attributes
    ----------
    model : ModelConfig
        Estimator trained on every recording with its four-way label.
    aggregation : str
        Frame-to-subject rule; like the disorder stage, the four-way decision
        never uses a threshold.
    """

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="mlp", params=_mlp_params([100] * 4, 80))
    )
    aggregation: str = "mean_proba"

    def __post_init__(self) -> None:
        _check_multiclass_aggregation("flat", self.aggregation)


@dataclass(frozen=True, slots=True)
class Config:
    """Root configuration object.

    Attributes
    ----------
    paths : PathsConfig
        Filesystem locations.
    mfcc : MFCCConfig
        Front-end parameters.
    trim : TrimConfig
        Waveform trimming.
    normalization : NormalizationConfig
        Cepstral normalisation.
    split : SplitConfig
        Cross-validation layout.
    detection : DetectionConfig
        Stage-one settings.
    disorder : DisorderConfig
        Stage-two settings.
    flat : FlatConfig
        Settings of the single-stage route.
    seed : int
        Global seed applied to every random number generator of a run.
    log_level : str
        Logging level name applied by the command line interface.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    mfcc: MFCCConfig = field(default_factory=MFCCConfig)
    trim: TrimConfig = field(default_factory=TrimConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    disorder: DisorderConfig = field(default_factory=DisorderConfig)
    flat: FlatConfig = field(default_factory=FlatConfig)
    seed: int = 0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.log_level.upper() not in _LOG_LEVELS:
            raise ConfigError(f"log_level={self.log_level!r} is not a logging level name")

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a configuration from a nested mapping.

        Parameters
        ----------
        data : dict
            Mapping whose keys mirror the dataclass fields.

        Returns
        -------
        Config
            The validated configuration.

        Raises
        ------
        ConfigError
            If an unknown key is present anywhere in the mapping.
        """
        return _from_mapping(cls, data, prefix="")

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
        """Load a configuration from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path
            YAML document with the same structure as :meth:`from_dict`.
        overrides : dict, optional
            Dotted-key overrides applied after loading, e.g.
            ``{"split.n_splits": 3}``.  Entries whose value is ``None`` are
            ignored, which lets the command line pass unset options straight
            through.

        Returns
        -------
        Config
            The validated configuration.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ConfigError
            If the document is not a mapping or contains unknown keys.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no such configuration file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a mapping at the top level")
        cfg = cls.from_dict(raw)
        _LOGGER.debug("loaded configuration from %s", path)
        return cfg.override(overrides) if overrides else cfg

    def override(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with dotted-key entries replaced.

        Parameters
        ----------
        overrides : dict
            Mapping from dotted paths such as ``"mfcc.frame_ms"`` to values.
            ``None`` values are skipped.

        Returns
        -------
        Config
            A new, validated configuration.

        Examples
        --------
        >>> Config().override({"seed": 7}).seed
        7
        """
        data = self.to_dict()
        for dotted, value in overrides.items():
            if value is None:
                continue
            node: dict[str, Any] = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    raise ConfigError(f"override key {dotted!r} does not address a section")
                node = child
            node[parts[-1]] = value
        return Config.from_dict(data)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain, YAML-serialisable mapping of the configuration."""
        return _to_plain(dataclasses.asdict(self))

    def to_yaml(self, path: str | Path) -> Path:
        """Write the configuration to ``path`` and return that path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, allow_unicode=True, sort_keys=False)
        return path


def _to_plain(value: Any) -> Any:
    """Convert paths and tuples inside a nested structure to YAML-safe types."""
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _from_mapping(cls: type[T], data: Any, prefix: str) -> T:
    """Recursively instantiate a configuration dataclass from a mapping."""
    if not isinstance(data, dict):
        raise ConfigError(f"section {prefix.rstrip('.') or '<root>'} must be a mapping")
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(
            f"unknown configuration key(s): {[prefix + key for key in unknown]}; "
            f"valid keys in this section are {sorted(names)}"
        )
    nested = _NESTED_TYPES.get(cls, {})
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        child = nested.get(key)
        kwargs[key] = _from_mapping(child, value, f"{prefix}{key}.") if child else value
    return cls(**kwargs)  # type: ignore[call-arg]


#: Which fields of which configuration class hold another configuration object.
_NESTED_TYPES: dict[type, dict[str, type]] = {
    Config: {
        "paths": PathsConfig,
        "mfcc": MFCCConfig,
        "trim": TrimConfig,
        "normalization": NormalizationConfig,
        "split": SplitConfig,
        "detection": DetectionConfig,
        "disorder": DisorderConfig,
        "flat": FlatConfig,
    },
    DetectionConfig: {"model": ModelConfig},
    DisorderConfig: {"model": ModelConfig},
    FlatConfig: {"model": ModelConfig},
}
