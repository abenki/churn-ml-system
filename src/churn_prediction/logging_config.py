"""Logging configuration for the churn prediction system."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up structured logging for the application.

    Args:
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("churn_prediction")

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name for the logger.

    Returns:
        Logger instance for the module.
    """
    return logging.getLogger(f"churn_prediction.{name}")
