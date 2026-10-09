"""Configuration objects: defaults, validation, overrides and round trips."""

from __future__ import annotations

import pytest
import yaml

from voice_pathology_detection.config import (
    Config,
    ConfigError,
    DetectionConfig,
    DisorderConfig,
    FlatConfig,
    MFCCConfig,
    NormalizationConfig,
    SplitConfig,
    TrimConfig,
)


def test_defaults_describe_the_reference_front_end() -> None:
    cfg = MFCCConfig()
    assert cfg.n_static == 13
    assert cfg.n_dim == 39
    assert cfg.frame_length == 1323
    assert cfg.resolved_n_fft == 2048


def test_n_fft_is_the_next_power_of_two_above_the_frame() -> None:
    assert MFCCConfig(sample_rate=16000, frame_ms=25.0).resolved_n_fft == 512
    assert MFCCConfig(sample_rate=16000, frame_ms=25.0, n_fft=1024).resolved_n_fft == 1024


def test_band_collapses_to_none_when_the_full_spectrum_is_kept() -> None:
    assert MFCCConfig(low_freq=None, high_freq=None).band is None
    assert MFCCConfig(low_freq=0.0, high_freq=None).band is None
    assert MFCCConfig(low_freq=64.0).band == (64.0, 22050.0)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"n_ceps": 30}, "smaller than"),
        ({"hop_ms": 40.0}, "must not exceed"),
        ({"preemph": 1.5}, "preemph"),
        ({"delta_width": 4}, "odd integer"),
        ({"low_freq": 30000.0}, "low_freq"),
        ({"n_fft": 64}, "smaller than the"),
    ],
)
def test_invalid_front_end_settings_are_rejected(kwargs: dict, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        MFCCConfig(**kwargs)


@pytest.mark.parametrize(
    ("factory", "kwargs", "fragment"),
    [
        (TrimConfig, {"mode": "magic"}, "not supported"),
        (TrimConfig, {"edge_seconds": -1.0}, "not be negative"),
        (NormalizationConfig, {"method": "zscore"}, "not supported"),
        (NormalizationConfig, {"divisor": "median"}, "not supported"),
        (SplitConfig, {"n_splits": 1}, "at least 2"),
        (SplitConfig, {"val_fraction": 0.95}, "val_fraction"),
        (SplitConfig, {"val_fraction": 0.6}, r"\(0, 0\.5\]"),
        (DetectionConfig, {"threshold_method": "fixed"}, "must be given"),
        (DetectionConfig, {"threshold": 1.5}, "must lie in"),
        (DetectionConfig, {"aggregation": "vote"}, "not supported"),
        (DisorderConfig, {"aggregation": "ratio_threshold"}, "only defined for two classes"),
        (FlatConfig, {"aggregation": "ratio_threshold"}, "only defined for two classes"),
        (FlatConfig, {"aggregation": "vote"}, "not supported"),
    ],
)
def test_invalid_section_settings_are_rejected(factory, kwargs: dict, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        factory(**kwargs)


def test_unknown_keys_are_reported_with_their_path() -> None:
    with pytest.raises(ConfigError, match=r"mfcc\.window_ms"):
        Config.from_dict({"mfcc": {"window_ms": 30.0}})


def test_unknown_top_level_key_lists_the_valid_ones() -> None:
    with pytest.raises(ConfigError, match="valid keys in this section"):
        Config.from_dict({"mdel": {}})


def test_override_replaces_nested_values_and_revalidates() -> None:
    cfg = Config().override({"split.n_splits": 3, "mfcc.frame_ms": 20.0})
    assert cfg.split.n_splits == 3
    assert cfg.mfcc.frame_ms == 20.0
    with pytest.raises(ConfigError):
        cfg.override({"split.n_splits": 1})


def test_override_ignores_none_so_unset_cli_options_pass_through() -> None:
    cfg = Config().override({"seed": None})
    assert cfg.seed == 0


def test_override_rejects_a_key_that_is_not_a_section() -> None:
    with pytest.raises(ConfigError, match="does not address a section"):
        Config().override({"seed.value": 3})


def test_yaml_round_trip_preserves_every_setting(tmp_path) -> None:
    original = Config().override(
        {"seed": 7, "normalization.method": "cvn_block", "detection.model.name": "svm"}
    )
    path = original.to_yaml(tmp_path / "config.yaml")
    restored = Config.from_yaml(path)
    assert restored.to_dict() == original.to_dict()
    assert restored.seed == 7
    assert restored.detection.model.name == "svm"


def test_yaml_paths_are_written_as_posix_strings(tmp_path) -> None:
    path = Config().to_yaml(tmp_path / "c.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["paths"]["data_root"] == "data/corpus"


def test_missing_config_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no such configuration file"):
        Config.from_yaml(tmp_path / "absent.yaml")


def test_a_yaml_document_that_is_not_a_mapping_is_rejected(tmp_path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        Config.from_yaml(path)


def test_shipped_configs_all_load() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(root.glob("*.yaml")):
        assert isinstance(Config.from_yaml(path), Config), path.name


def test_default_yaml_holds_exactly_the_built_in_defaults() -> None:
    # A command run without --config uses Config(); the file must not drift.
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert Config.from_yaml(path) == Config()


def test_paths_helpers_compose_the_expected_locations() -> None:
    cfg = Config().override({"paths.data_root": "/data", "paths.demographics": "meta.xlsx"})
    assert cfg.paths.train_root.as_posix().endswith("/data/train")
    assert cfg.paths.test_root.as_posix().endswith("/data/test")
    assert cfg.paths.demographics_path is not None
    assert cfg.paths.demographics_path.name == "meta.xlsx"


def test_demographics_path_is_none_when_unset() -> None:
    assert Config().paths.demographics_path is None


def test_log_level_is_validated() -> None:
    with pytest.raises(ConfigError, match="logging level"):
        Config(log_level="LOUD")
