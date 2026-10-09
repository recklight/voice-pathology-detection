"""Package metadata that other files repeat and must agree with."""

from __future__ import annotations

from pathlib import Path

import voice_pathology_detection

PYPROJECT = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")


def test_version_matches_the_release() -> None:
    # pyproject.toml, CITATION.cff and CHANGELOG.md carry the same number.
    assert voice_pathology_detection.__version__ == "0.1.0"


def test_the_speechdsp_floor_is_the_version_ci_pins() -> None:
    # The oldest-dependency job in ci.yml installs speechdsp==0.2.*; move both together.
    assert '"speechdsp>=0.2.0"' in PYPROJECT
