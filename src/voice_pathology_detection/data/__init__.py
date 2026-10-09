"""Corpus indexing, labels, trimming, splits and synthetic material."""

from __future__ import annotations

from .dataset import DatasetError, build_index, read_demographics, scan_class_directories
from .labels import (
    ALL_LABELS,
    BINARY_NAMES,
    CLASS_NAMES,
    DISORDER_LABELS,
    PATHOLOGICAL_LABELS,
    BinaryClass,
    VoiceClass,
    class_name,
    is_pathological,
    parse_label,
    to_binary,
)
from .splits import Fold, SplitError, check_no_group_leakage, subject_folds, train_val_split
from .synthetic import make_frame_dataset, synthesize_voice, write_synthetic_corpus
from .trim import TrimResult, trim_audio, trim_energy_vad, trim_fixed_edges

__all__ = [
    "ALL_LABELS",
    "BINARY_NAMES",
    "CLASS_NAMES",
    "DISORDER_LABELS",
    "PATHOLOGICAL_LABELS",
    "BinaryClass",
    "DatasetError",
    "Fold",
    "SplitError",
    "TrimResult",
    "VoiceClass",
    "build_index",
    "check_no_group_leakage",
    "class_name",
    "is_pathological",
    "make_frame_dataset",
    "parse_label",
    "read_demographics",
    "scan_class_directories",
    "subject_folds",
    "synthesize_voice",
    "to_binary",
    "train_val_split",
    "trim_audio",
    "trim_energy_vad",
    "trim_fixed_edges",
    "write_synthetic_corpus",
]
