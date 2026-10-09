"""Seeding the random number generators of a run.

A run can only be repeated if its seed is set in one place and stored with the
results.  :func:`set_global_seed` seeds the standard library, NumPy's global
random state and, when it is installed, PyTorch, and returns a
:class:`numpy.random.Generator` that callers should pass along instead of going
back to the global state.

Examples
--------
>>> rng = set_global_seed(0)
>>> float(rng.random()) == float(set_global_seed(0).random())
True
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """What was seeded, for storage next to the results of a run.

    Attributes
    ----------
    seed : int
        The seed that was applied.
    numpy_seeded, random_seeded : bool
        Whether the respective generator was reseeded.
    torch_seeded : bool
        ``True`` only when PyTorch was importable and therefore seeded too.
    """

    seed: int
    numpy_seeded: bool = True
    random_seeded: bool = True
    torch_seeded: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        """Return a JSON-serialisable view of the record."""
        return {
            "seed": self.seed,
            "numpy_seeded": self.numpy_seeded,
            "random_seeded": self.random_seeded,
            "torch_seeded": self.torch_seeded,
        }


def _seed_torch(seed: int) -> bool:
    """Seed PyTorch if it is installed; report whether that happened.

    The import stays inside this function so the package works without the
    optional ``dl`` extra.
    """
    try:  # pragma: no cover - exercised only when the optional extra is present
        import torch
    except ImportError:
        return False
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(seed)
    return True


def set_global_seed(seed: int = 0) -> np.random.Generator:
    """Seed every random number generator the project can touch.

    Parameters
    ----------
    seed : int, optional
        Non-negative seed, default 0.

    Returns
    -------
    numpy.random.Generator
        A fresh generator built from ``seed``.  Prefer passing this object
        around over relying on the global NumPy state.

    Raises
    ------
    ValueError
        If ``seed`` is negative.

    Examples
    --------
    >>> rng = set_global_seed(42)
    >>> rng.integers(0, 10).item() == set_global_seed(42).integers(0, 10).item()
    True
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch_seeded = _seed_torch(seed)
    _LOGGER.debug("global seed set to %d (torch seeded: %s)", seed, torch_seeded)
    return np.random.default_rng(seed)


def seed_record(seed: int = 0) -> SeedRecord:
    """Seed everything and return the record describing what was seeded.

    Parameters
    ----------
    seed : int, optional
        Non-negative seed, default 0.

    Returns
    -------
    SeedRecord
        Description of the generators that were reseeded.
    """
    set_global_seed(seed)
    try:  # pragma: no cover - depends on the optional extra
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False
    return SeedRecord(seed=int(seed), torch_seeded=torch_available)
