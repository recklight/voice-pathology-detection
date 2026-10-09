"""Saving each run into a folder of its own.

Each run gets its own directory under ``results/`` holding the configuration it
used, its metrics as JSON and as CSV, its predictions and its figures.  The
configuration sits next to the numbers it produced, so rerunning a past
experiment means pointing the CLI at that directory's ``config.yaml``.

Examples
--------
>>> import tempfile, pathlib
>>> from voice_pathology_detection.config import Config
>>> with tempfile.TemporaryDirectory() as tmp:
...     run = save_run(pathlib.Path(tmp), Config(), {"uar": 0.62}, run_id="demo")
...     sorted(p.name for p in run.iterdir())
['config.yaml', 'metrics.json', 'report.md']
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config

_LOGGER = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder for NumPy scalars and arrays, paths and data frames.

    Metrics come out of NumPy and scikit-learn as ``np.float64`` and small
    arrays, which the standard encoder rejects.  Converting them here saves a
    ``float(...)`` cast at every call site.
    """

    def default(self, o: Any) -> Any:
        """Convert an unsupported object into something JSON can hold."""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return o.as_posix()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        return super().default(o)


def new_run_id(prefix: str = "run", results_dir: str | Path | None = None) -> str:
    """Build a time-stamped identifier for a run.

    Parameters
    ----------
    prefix : str, optional
        Leading word of the identifier, default ``"run"``.
    results_dir : str or pathlib.Path, optional
        Parent directory of the runs.  When given, the run directory is
        created here and the identifier gets a ``-2``, ``-3`` ... suffix if a
        run of the same second already took the plain name, so two runs never
        share a directory.

    Returns
    -------
    str
        Something like ``"run-20261008-101500"``, sortable by time.

    Examples
    --------
    >>> new_run_id("detection").startswith("detection-")
    True
    """
    stamp = f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    if results_dir is None:
        return stamp
    parent = Path(results_dir)
    parent.mkdir(parents=True, exist_ok=True)
    candidate, counter = stamp, 1
    while True:
        try:
            # mkdir is atomic, so two processes cannot claim the same name.
            (parent / candidate).mkdir()
        except FileExistsError:
            counter += 1
            candidate = f"{stamp}-{counter}"
        else:
            return candidate


def _json_safe(value: Any) -> Any:
    """Return ``value`` with NumPy types made native and non-finite floats as ``None``."""
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray | pd.DataFrame):
        return _json_safe(NumpyJSONEncoder().default(value))
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a mapping as indented JSON.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file; parent directories are created.
    payload : mapping
        Content to serialise.  Undefined numbers (``nan``, ``inf``) are
        written as ``null``: the ``NaN`` token Python would emit is not JSON,
        and strict parsers such as ``jq`` or JavaScript reject the file.

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            cls=NumpyJSONEncoder,
            ensure_ascii=False,
            allow_nan=False,
        )
    return path


def _format_summary(summary: Mapping[str, Any]) -> list[str]:
    """Render a mean/standard-deviation summary as Markdown table rows."""
    rows = ["| Metric | Mean | Std | Folds |", "| --- | --- | --- | --- |"]
    for metric, values in summary.items():
        if not isinstance(values, Mapping):
            continue
        mean, deviation = values.get("mean"), values.get("std")
        mean_text = "n/a" if mean is None or not np.isfinite(mean) else f"{mean:.4f}"
        std_text = "n/a" if deviation is None or not np.isfinite(deviation) else f"{deviation:.4f}"
        rows.append(
            f"| {metric.replace('_', ' ')} | {mean_text} | {std_text} | {values.get('n', 0)} |"
        )
    return rows


def render_markdown(payload: Mapping[str, Any], title: str = "Run report") -> str:
    """Render a metrics payload as a short Markdown report.

    Parameters
    ----------
    payload : mapping
        Metrics, typically the ``to_dict`` of a stage or cascade result.
    title : str, optional
        Heading of the document.

    Returns
    -------
    str
        The Markdown source.
    """
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"# {title}", "", f"Generated {stamp}", ""]

    task = payload.get("task")
    if task:
        lines += [f"**Task**: `{task}`", ""]
    if "labels" in payload:
        lines += [f"**Labels**: {payload['labels']}", ""]

    for key, heading in (
        ("subject_summary", "Subject level, across folds"),
        ("frame_summary", "Frame level, across folds"),
        ("detection_summary", "Detection stage, across folds"),
        ("disorder_summary", "Disorder stage, across folds"),
        ("overall_summary", "Merged four-way decision, across folds"),
    ):
        summary = payload.get(key)
        if isinstance(summary, Mapping) and summary:
            lines += [f"## {heading}", "", *_format_summary(summary), ""]

    for key, heading in (
        ("pooled_subject", "Pooled subject-level metrics"),
        ("pooled_overall", "Pooled four-way metrics"),
    ):
        pooled = payload.get(key)
        if isinstance(pooled, Mapping) and pooled.get("confusion_matrix"):
            lines += [
                f"## {heading}",
                "",
                f"- n = {pooled.get('n')}",
                f"- UAR = {float(pooled.get('uar', float('nan'))):.4f}",
                f"- accuracy = {float(pooled.get('accuracy', float('nan'))):.4f}",
                "",
                "Confusion matrix (rows: true, columns: predicted):",
                "",
                "```",
                *[str(row) for row in pooled["confusion_matrix"]],
                "```",
                "",
            ]

    if payload.get("thresholds"):
        values = [float(v) for v in payload["thresholds"]]
        lines += [
            "## Decision threshold",
            "",
            f"Chosen per fold: {[round(v, 4) for v in values]}",
            f"Mean {np.mean(values):.4f}, standard deviation {np.std(values):.4f}",
            "",
        ]
    return "\n".join(lines) + "\n"


def save_run(
    results_dir: str | Path,
    cfg: Config,
    payload: Mapping[str, Any],
    *,
    run_id: str | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    title: str = "Run report",
) -> Path:
    """Persist everything one run produced into its own directory.

    Parameters
    ----------
    results_dir : str or pathlib.Path
        Parent directory of all runs.
    cfg : Config
        The configuration that was used; written as ``config.yaml``.
    payload : mapping
        Metrics, written as ``metrics.json`` and rendered into ``report.md``.
    run_id : str, optional
        Name of the run directory; a fresh time-stamped one from
        :func:`new_run_id` when omitted.  A directory that already exists under
        an explicit name is written into.
    tables : mapping, optional
        Extra tables to write as CSV, keyed by file stem.
    title : str, optional
        Heading of the Markdown report.

    Returns
    -------
    pathlib.Path
        The run directory.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from voice_pathology_detection.config import Config
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     run = save_run(tmp, Config(), {"task": "demo"}, run_id="r1")
    ...     (run / "metrics.json").is_file()
    True
    """
    run_dir = Path(results_dir) / (run_id or new_run_id(results_dir=results_dir))
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.to_yaml(run_dir / "config.yaml")
    write_json(run_dir / "metrics.json", payload)
    (run_dir / "report.md").write_text(render_markdown(payload, title), encoding="utf-8")

    for stem, table in (tables or {}).items():
        table.to_csv(run_dir / f"{stem}.csv", index=False)

    _LOGGER.info("wrote the run artefacts to %s", run_dir)
    return run_dir
