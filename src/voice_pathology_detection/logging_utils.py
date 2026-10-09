"""Logging setup for the CLI and the example scripts.

Apart from the CLI, no module in the package prints anything.  They log
through ``logging.getLogger(__name__)`` and leave the handlers to whoever runs
them.  :func:`configure_logging` is the one place that installs a handler; the
CLI and the scripts in ``examples/`` call it at start-up.

Examples
--------
>>> configure_logging("WARNING")
>>> import logging
>>> logging.getLogger("voice_pathology_detection").level
30
"""

from __future__ import annotations

import logging
import sys

#: Record format of the installed handler.
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

#: Timestamp format of the installed handler, time of day only.
DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: str = "INFO", *, stream: object | None = None) -> None:
    """Install a stream handler on the package logger.

    Calling this twice replaces the handler rather than adding a second one, so
    repeated CLI invocations inside one process do not duplicate every line.

    Parameters
    ----------
    level : str, optional
        Logging level name, default ``"INFO"``.
    stream : file-like, optional
        Destination of the log records, default :data:`sys.stderr`.

    Raises
    ------
    ValueError
        If ``level`` is not a logging level name.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown logging level: {level!r}")
    logger = logging.getLogger("voice_pathology_detection")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(numeric)
    logger.propagate = False
