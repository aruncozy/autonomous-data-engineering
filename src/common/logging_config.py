"""Structured Cloud Logging setup for all agents."""

from __future__ import annotations

import logging
import os
import sys

import google.cloud.logging as cloud_logging


def setup_logging(agent_id: str = "unknown", default_level: str = "INFO") -> None:
    """Configure structured logging.

    In GCP environments the Cloud Logging handler is attached so that logs
    appear in Cloud Logging with JSON payloads.  Locally, a human-readable
    ``StreamHandler`` is used instead.
    """
    level = getattr(logging, os.environ.get("LOG_LEVEL", default_level).upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicate output.
    root.handlers.clear()

    if os.environ.get("K_SERVICE"):
        # Running on Cloud Run / GCP — use Cloud Logging.
        client = cloud_logging.Client()
        handler = client.get_default_handler()
        handler.setLevel(level)
        root.addHandler(handler)
    else:
        # Local development — structured but human-readable.
        fmt = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            f"agent={agent_id} | %(message)s"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
        handler.setLevel(level)
        root.addHandler(handler)

    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
