"""Score-level fusion of several systems."""

from __future__ import annotations

from .score_fusion import (
    FusionResult,
    StackingFuser,
    fuse_scores,
    grid_search_weights,
    simplex_grid,
)

__all__ = [
    "FusionResult",
    "StackingFuser",
    "fuse_scores",
    "grid_search_weights",
    "simplex_grid",
]
