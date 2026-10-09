"""Shared fixtures.

The shared fixtures are generated here: a two-harmonic tone, synthesised
vowels written to a temporary folder, and Gaussian frame clusters with known
class means.  No test needs a real corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from voice_pathology_detection.config import Config, MFCCConfig
from voice_pathology_detection.data.labels import CLASS_NAMES
from voice_pathology_detection.data.synthetic import make_frame_dataset, write_synthetic_corpus
from voice_pathology_detection.features.pipeline import FeatureSet

#: Sampling rate used by the audio fixtures; low enough to keep tests quick.
SR = 8000


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so every test is reproducible."""
    return np.random.default_rng(0)


@pytest.fixture
def tone() -> np.ndarray:
    """One second of a 220 Hz tone with a second harmonic."""
    t = np.arange(SR) / SR
    return np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def mfcc_cfg() -> MFCCConfig:
    """A front end matched to the fixture sampling rate."""
    return MFCCConfig(
        sample_rate=SR,
        frame_ms=25.0,
        hop_ms=10.0,
        low_freq=64.0,
        high_freq=3800.0,
    )


@pytest.fixture
def frame_data() -> tuple[np.ndarray, np.ndarray, object]:
    """A four-class frame matrix with separable class means."""
    return make_frame_dataset(n_files_per_class=6, n_frames=10, n_dim=12, separation=2.0, seed=0)


@pytest.fixture
def feature_set(frame_data) -> FeatureSet:
    """The frame matrix wrapped in a :class:`FeatureSet`."""
    X, frame_file, files = frame_data
    return FeatureSet(X, frame_file, files)


@pytest.fixture
def small_config() -> Config:
    """A configuration sized for the fixtures: two folds, cheap models."""
    return Config().override(
        {
            "split.n_splits": 2,
            "split.val_fraction": 0.4,
            "normalization.method": "cmvn",
            "detection.model.name": "gmm",
            "detection.model.params": {"n_components": 1},
            "disorder.model.name": "gmm",
            "disorder.model.params": {"n_components": 1},
            "flat.model.name": "gmm",
            "flat.model.params": {"n_components": 1},
        }
    )


@pytest.fixture
def synthetic_corpus(tmp_path):
    """A tiny synthetic corpus on disk, three recordings per class."""
    write_synthetic_corpus(tmp_path, dict.fromkeys(CLASS_NAMES, 3), sr=SR, duration=0.6, seed=0)
    return tmp_path


@pytest.fixture
def corpus_config(synthetic_corpus, tmp_path) -> Config:
    """A configuration pointing at the synthetic corpus."""
    return Config().override(
        {
            "paths.data_root": str(synthetic_corpus),
            "paths.cache_dir": str(tmp_path / "cache"),
            "paths.results_dir": str(tmp_path / "results"),
            "mfcc.sample_rate": SR,
            "mfcc.frame_ms": 25.0,
            "mfcc.hop_ms": 10.0,
            "mfcc.high_freq": 3800.0,
            "trim.mode": "none",
            "normalization.method": "cmvn",
            "split.n_splits": 2,
            "split.val_fraction": 0.4,
            "detection.model.name": "gmm",
            "detection.model.params": {"n_components": 1},
            "disorder.model.name": "gmm",
            "disorder.model.params": {"n_components": 1},
            "flat.model.name": "gmm",
            "flat.model.params": {"n_components": 1},
        }
    )
