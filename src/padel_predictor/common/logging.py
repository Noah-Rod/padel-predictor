"""Uniform logging for every pipeline entrypoint."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, at the level given by ``PADEL_LOG_LEVEL``."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.getenv("PADEL_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    setup_logging()
    return logging.getLogger(name)
