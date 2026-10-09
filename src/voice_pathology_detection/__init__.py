"""Pathological voice detection and classification from sustained vowels.

Two questions about a recording of a sustained ``/a:/``: is the voice normal
or pathological, and if it is pathological, does it look most like Neoplasm,
Phonotrauma or Vocal palsy.  This is research code, not a medical device: it
has not been validated clinically, and its output is not a diagnosis.

From a WAV file to a decision:

1. trim the recording to the phonated segment
   (:mod:`voice_pathology_detection.data.trim`);
2. analyse it into 13 cepstral coefficients per 30 ms frame, expanded with
   velocity and acceleration to 39 dimensions
   (:mod:`voice_pathology_detection.features.mfcc`);
3. normalise the frames with one of the strategies in
   :mod:`voice_pathology_detection.features.normalize`;
4. classify every frame with a network, a deep belief network or a
   baseline (:mod:`voice_pathology_detection.models`);
5. aggregate the frame scores into one decision per subject, at a threshold
   chosen on held-out data (:mod:`voice_pathology_detection.eval.aggregate`,
   :mod:`voice_pathology_detection.eval.threshold`);
6. write cross-validated metrics, figures and predictions
   (:mod:`voice_pathology_detection.eval.report`).

All settings live in one :class:`~voice_pathology_detection.config.Config`
object, and every split is grouped by subject.  The signal processing comes
from the ``speechdsp`` package.

Examples
--------
>>> from voice_pathology_detection import Config, extract_features
>>> import numpy as np
>>> cfg = Config().override({"mfcc.sample_rate": 16000, "mfcc.low_freq": None})
>>> x = np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)
>>> extract_features(x, 16000, cfg.mfcc).shape[1]
39
"""

from __future__ import annotations

from .config import (
    Config,
    ConfigError,
    DetectionConfig,
    DisorderConfig,
    FlatConfig,
    MFCCConfig,
    ModelConfig,
    NormalizationConfig,
    PathsConfig,
    SplitConfig,
    TrimConfig,
)
from .data.dataset import build_index, scan_class_directories
from .data.labels import CLASS_NAMES, BinaryClass, VoiceClass, class_name, to_binary
from .data.splits import subject_folds, train_val_split
from .data.synthetic import make_frame_dataset, synthesize_voice, write_synthetic_corpus
from .data.trim import trim_audio
from .eval.aggregate import aggregate_frames
from .eval.metrics import evaluate, summarise_folds
from .eval.report import save_run
from .eval.threshold import select_threshold, threshold_sweep
from .features.mfcc import extract_features, extract_static
from .features.normalize import make_normalizer
from .features.pipeline import FeatureSet, extract_corpus, extract_file
from .logging_utils import configure_logging
from .models.registry import available_models, build_model
from .prediction_table import write_prediction_table
from .routes.cascade import CascadeResult, run_cascade
from .routes.detection import fit_detection, run_detection
from .routes.disorder import fit_disorder, run_disorder
from .routes.flat import fit_flat, run_flat
from .routes.stage import StageModel, StageResult
from .seed import set_global_seed

__version__ = "0.1.0"

__all__ = [
    "CLASS_NAMES",
    "BinaryClass",
    "CascadeResult",
    "Config",
    "ConfigError",
    "DetectionConfig",
    "DisorderConfig",
    "FeatureSet",
    "FlatConfig",
    "MFCCConfig",
    "ModelConfig",
    "NormalizationConfig",
    "PathsConfig",
    "SplitConfig",
    "StageModel",
    "StageResult",
    "TrimConfig",
    "VoiceClass",
    "__version__",
    "aggregate_frames",
    "available_models",
    "build_index",
    "build_model",
    "class_name",
    "configure_logging",
    "evaluate",
    "extract_corpus",
    "extract_features",
    "extract_file",
    "extract_static",
    "fit_detection",
    "fit_disorder",
    "fit_flat",
    "make_frame_dataset",
    "make_normalizer",
    "run_cascade",
    "run_detection",
    "run_disorder",
    "run_flat",
    "save_run",
    "scan_class_directories",
    "select_threshold",
    "set_global_seed",
    "subject_folds",
    "summarise_folds",
    "synthesize_voice",
    "threshold_sweep",
    "to_binary",
    "train_val_split",
    "trim_audio",
    "write_prediction_table",
    "write_synthetic_corpus",
]
