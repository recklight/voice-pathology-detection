"""The training and evaluation routes: detection, disorder, cascade and flat."""

from __future__ import annotations

from .cascade import CascadeResult, run_cascade
from .detection import binary_view, fit_detection, run_detection
from .disorder import fit_disorder, pathological_subset, run_disorder
from .flat import fit_flat, run_flat
from .stage import (
    FoldOutcome,
    StageError,
    StageModel,
    StageResult,
    fit_stage,
    frame_posteriors,
    run_stage,
)

__all__ = [
    "CascadeResult",
    "FoldOutcome",
    "StageError",
    "StageModel",
    "StageResult",
    "binary_view",
    "fit_detection",
    "fit_disorder",
    "fit_flat",
    "fit_stage",
    "frame_posteriors",
    "pathological_subset",
    "run_cascade",
    "run_detection",
    "run_disorder",
    "run_flat",
    "run_stage",
]
