"""Aggregation, thresholds, metrics, figures and run artefacts."""

from __future__ import annotations

from .aggregate import AggregateResult, aggregate_frames, frame_offsets, split_by_group
from .metrics import (
    confusion_frame,
    evaluate,
    folds_to_frame,
    predictive_values,
    summarise_folds,
)
from .plots import (
    plot_confusion_matrix,
    plot_fold_errorbar,
    plot_roc,
    plot_spectrogram_grid,
    plot_threshold_curve,
)
from .report import NumpyJSONEncoder, new_run_id, render_markdown, save_run, write_json
from .threshold import ThresholdChoice, roc_table, select_threshold, threshold_sweep

__all__ = [
    "AggregateResult",
    "NumpyJSONEncoder",
    "ThresholdChoice",
    "aggregate_frames",
    "confusion_frame",
    "evaluate",
    "folds_to_frame",
    "frame_offsets",
    "new_run_id",
    "plot_confusion_matrix",
    "plot_fold_errorbar",
    "plot_roc",
    "plot_spectrogram_grid",
    "plot_threshold_curve",
    "predictive_values",
    "render_markdown",
    "roc_table",
    "save_run",
    "select_threshold",
    "split_by_group",
    "summarise_folds",
    "threshold_sweep",
    "write_json",
]
