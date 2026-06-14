"""Centralized logging configuration built on `loguru`.

Call :func:`configure_logging` once at process startup (the FastAPI lifespan
handler and every CLI entrypoint do this). Library modules should simply do
``from loguru import logger`` and log; they must not configure sinks.
"""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED: bool = False

_LOG_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logging(level: str = "INFO") -> None:
    """Configure the global loguru sink (idempotent).

    Parameters
    ----------
    level:
        Minimum log level, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=_LOG_FORMAT,
        backtrace=False,
        diagnose=False,  # keep tracebacks free of variable values in prod logs
        enqueue=True,  # safe across asyncio / threads
    )
    _CONFIGURED = True
    logger.debug("Logging configured at level {}", level.upper())
