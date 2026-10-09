"""Figures drawn from the recorded metrics.

Every figure is drawn from a report or a table that the run also writes to
disk, so a plot can be redrawn from the run folder and always shows the same
numbers as the files next to it.

The Agg backend is selected on import, because these functions run inside a
command line tool and on machines with no display.

Examples
--------
>>> import tempfile, pathlib
>>> from voice_pathology_detection.eval.metrics import evaluate
>>> report = evaluate([0, 1, 1], [0, 1, 0], labels=[0, 1])
>>> with tempfile.TemporaryDirectory() as tmp:
...     out = plot_confusion_matrix(report, pathlib.Path(tmp) / "cm.png")
...     out.is_file()
True
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from speechdsp import read_wav, spectrogram_db

_LOGGER = logging.getLogger(__name__)

#: Resolution of every saved figure, in dots per inch.
DPI: int = 150


def _prepare(path: str | Path) -> Path:
    """Create the parent directory of a figure and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(figure: plt.Figure, path: Path) -> Path:
    """Write and close a figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    _LOGGER.debug("wrote figure %s", path)
    return path


def plot_confusion_matrix(
    report: Mapping[str, object],
    path: str | Path,
    *,
    names: Mapping[int, str] | None = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> Path:
    """Draw the confusion matrix of a report as a heat map.

    Parameters
    ----------
    report : mapping
        A report from :func:`voice_pathology_detection.eval.metrics.evaluate`.
    path : str or pathlib.Path
        Destination image file.
    names : mapping, optional
        Display name of every label code.
    normalize : bool, optional
        Colour the cells by per-class recall rather than by raw counts,
        default ``True``.  With 50 normal and 150 pathological recordings, raw
        counts make the larger classes look uniformly darker and say nothing
        about accuracy.
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Raises
    ------
    KeyError
        If the report has no confusion matrix.
    """
    matrix = np.asarray(report["confusion_matrix"], dtype=float)
    labels = [int(label) for label in report["labels"]]  # type: ignore[union-attr]
    titles = [names.get(label, str(label)) if names else str(label) for label in labels]

    display = matrix.copy()
    if normalize:
        totals = display.sum(axis=1, keepdims=True)
        display = np.divide(display, totals, out=np.zeros_like(display), where=totals > 0)

    figure, axes = plt.subplots(figsize=(1.6 + 1.1 * len(labels), 1.4 + 1.0 * len(labels)))
    image = axes.imshow(display, cmap="Blues", vmin=0.0, vmax=display.max() or 1.0)
    axes.set_xticks(range(len(labels)), titles, rotation=30, ha="right")
    axes.set_yticks(range(len(labels)), titles)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title(title)

    threshold = (display.max() or 1.0) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            text = (
                f"{display[row, column]:.2f}\n({int(matrix[row, column])})"
                if normalize
                else f"{int(matrix[row, column])}"
            )
            axes.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if display[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axes, shrink=0.8)
    return _save(figure, _prepare(path))


def plot_threshold_curve(
    sweep: pd.DataFrame,
    path: str | Path,
    *,
    chosen: float | None = None,
    title: str = "Operating point",
) -> Path:
    """Plot sensitivity and specificity against the decision threshold.

    Parameters
    ----------
    sweep : pandas.DataFrame
        Table from :func:`voice_pathology_detection.eval.threshold.threshold_sweep`.
    path : str or pathlib.Path
        Destination image file.
    chosen : float, optional
        Threshold to mark with a vertical line.
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    required = {"threshold", "sensitivity", "specificity"}
    missing = required - set(sweep.columns)
    if missing:
        raise KeyError(f"the sweep table lacks the column(s) {sorted(missing)}")

    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(sweep["threshold"], sweep["sensitivity"], label="Sensitivity", linewidth=2)
    axes.plot(sweep["threshold"], sweep["specificity"], label="Specificity", linewidth=2)
    if "youden" in sweep:
        axes.plot(
            sweep["threshold"], sweep["youden"], label="Youden's J", linestyle="--", alpha=0.7
        )
    if chosen is not None:
        axes.axvline(
            chosen, color="crimson", linestyle=":", linewidth=2, label=f"chosen = {chosen:.3f}"
        )
    axes.set_xlabel("Threshold on the subject score")
    axes.set_ylabel("Rate")
    axes.set_ylim(-0.02, 1.02)
    axes.set_title(title)
    axes.grid(alpha=0.3)
    axes.legend(loc="best")
    return _save(figure, _prepare(path))


def plot_fold_errorbar(
    table: pd.DataFrame,
    path: str | Path,
    *,
    metrics: Sequence[str] = ("uar", "accuracy", "sensitivity", "specificity"),
    title: str = "Cross-validated metrics",
) -> Path:
    """Plot each metric's mean across folds with its standard deviation.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-fold table from :func:`voice_pathology_detection.eval.metrics.folds_to_frame`.
    path : str or pathlib.Path
        Destination image file.
    metrics : sequence of str, optional
        Which columns to draw.
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Notes
    -----
    The error bars are there because a mean of 0.62 from folds ranging over
    0.45 to 0.80 is a different result from the same mean with every fold
    near 0.62.
    """
    available = [metric for metric in metrics if metric in table.columns]
    if not available:
        raise KeyError(
            f"none of the metrics {list(metrics)} are columns of the table {sorted(table.columns)}"
        )

    means, deviations = [], []
    for metric in available:
        values = pd.to_numeric(table[metric], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        means.append(float(finite.mean()) if finite.size else np.nan)
        deviations.append(float(finite.std()) if finite.size else np.nan)

    positions = np.arange(len(available))
    figure, axes = plt.subplots(figsize=(1.6 + 1.4 * len(available), 4.5))
    axes.bar(positions, means, yerr=deviations, capsize=6, color="#4c72b0", alpha=0.85)
    for position, (mean, deviation) in enumerate(zip(means, deviations, strict=True)):
        if np.isfinite(mean):
            axes.text(
                position,
                min(mean + (deviation if np.isfinite(deviation) else 0.0) + 0.03, 1.02),
                f"{mean:.3f}",
                ha="center",
                fontsize=9,
            )
    axes.set_xticks(positions, [metric.replace("_", " ") for metric in available])
    axes.set_ylim(0.0, 1.08)
    axes.set_ylabel("Value")
    axes.set_title(title)
    axes.grid(axis="y", alpha=0.3)
    return _save(figure, _prepare(path))


def plot_roc(
    y_true: object,
    score: object,
    path: str | Path,
    *,
    positive_label: int = 1,
    title: str = "ROC curve",
) -> Path:
    """Draw the ROC curve of a set of subject scores.

    Parameters
    ----------
    y_true : array_like
        True labels.
    score : array_like
        Continuous score of the positive class.
    path : str or pathlib.Path
        Destination image file.
    positive_label : int, optional
        Label treated as positive, default 1.
    title : str, optional
        Figure title.

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    from sklearn.metrics import auc, roc_curve

    truth = np.asarray(y_true).ravel()
    values = np.asarray(score, dtype=np.float64).ravel()
    if np.unique(truth).size < 2:
        raise ValueError("an ROC curve needs both classes to be present")

    false_positive, true_positive, _ = roc_curve(truth, values, pos_label=positive_label)
    area = auc(false_positive, true_positive)

    figure, axes = plt.subplots(figsize=(5.5, 5))
    axes.plot(false_positive, true_positive, linewidth=2, label=f"AUC = {area:.3f}")
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.7, label="chance")
    axes.set_xlabel("1 - specificity")
    axes.set_ylabel("Sensitivity")
    axes.set_title(title)
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right")
    return _save(figure, _prepare(path))


def plot_spectrogram_grid(
    files: Sequence[str | Path],
    path: str | Path,
    *,
    titles: Sequence[str] | None = None,
    n_fft: int = 1024,
    noverlap: int = 1000,
    max_freq: float = 8000.0,
    top_db: float = 80.0,
) -> Path:
    """Draw narrow-band spectrograms of several recordings side by side.

    The default window of 1024 samples (23 ms at 44.1 kHz, 64 ms at 16 kHz)
    spans several pitch periods, so the individual harmonics show up as
    horizontal lines.  A clean harmonic structure and the noise that fills the
    gaps between harmonics in a disordered voice can then be told apart at a
    glance.  The spectrogram itself comes from :func:`speechdsp.spectrogram_db`
    (Hann window, levels relative to the peak of each recording).

    Parameters
    ----------
    files : sequence of str or pathlib.Path
        Recordings to draw, one panel each.
    path : str or pathlib.Path
        Destination image file.
    titles : sequence of str, optional
        Panel titles; the file stems are used when omitted.
    n_fft : int, optional
        Window and FFT length in samples, default 1024.  A recording shorter
        than that is analysed with one window of its own length.
    noverlap : int, optional
        Overlap in samples, default 1000, i.e. a hop of 24 samples.
    max_freq : float, optional
        Upper limit of the frequency axis in Hz, default 8000.
    top_db : float, optional
        Dynamic range below the peak, default 80.  A silent recording is drawn
        at ``-top_db`` throughout.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Raises
    ------
    ValueError
        If no file is given or ``noverlap`` is not smaller than ``n_fft``.
    FileNotFoundError
        If a recording does not exist.
    """
    paths = [Path(item) for item in files]
    if not paths:
        raise ValueError("at least one recording is required")
    if noverlap >= n_fft:
        raise ValueError(f"noverlap={noverlap} must be smaller than n_fft={n_fft}")

    columns = min(2, len(paths))
    rows = int(np.ceil(len(paths) / columns))
    figure, axes_grid = plt.subplots(
        rows, columns, figsize=(6.5 * columns, 3.6 * rows), squeeze=False
    )

    for position, source in enumerate(paths):
        axes = axes_grid[position // columns][position % columns]
        audio, sr = read_wav(source)
        window = min(n_fft, audio.size)
        hop = window - min(noverlap, window - 1)
        decibels = spectrogram_db(audio, sr, n_fft=window, hop=hop, top_db=top_db)
        frequencies = np.fft.rfftfreq(window, d=1.0 / sr)
        # Frames are not centred: frame t covers [t * hop, t * hop + window),
        # so it is drawn at the middle of that span.
        times = (np.arange(decibels.shape[0]) * hop + window / 2.0) / sr

        keep = frequencies <= max_freq
        axes.pcolormesh(times, frequencies[keep], decibels[:, keep].T, shading="auto", cmap="magma")
        axes.set_title(
            titles[position] if titles is not None and position < len(titles) else source.stem
        )
        axes.set_xlabel("Time (s)")
        axes.set_ylabel("Frequency (Hz)")

    for empty in range(len(paths), rows * columns):
        axes_grid[empty // columns][empty % columns].axis("off")
    return _save(figure, _prepare(path))
