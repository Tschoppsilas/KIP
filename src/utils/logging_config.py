"""Logging setup for UniVision2Board."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple global logging format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
