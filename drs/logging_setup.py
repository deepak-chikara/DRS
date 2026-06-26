"""Structured application logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from drs.paths import user_logs_dir


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("drs")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    log_file = user_logs_dir() / "drs.log"
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.WARNING)
    logger.addHandler(console)

    logger.info("Logging initialized at %s", log_file)
    return logger
