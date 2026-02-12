"""Source-specific handlers that normalize data to IngestedRecord format."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from google.cloud import pubsub_v1, storage

from src.common.models import CDCOperation, IngestedRecord

logger = logging.getLogger(__name__)


class BaseSourceHandler(ABC):
    """Common interface for all source handlers."""

    @abstractmethod
    def handle(self, payload: dict[str, Any]) -> IngestedRecord:
        """Process the raw event payload and return an IngestedRecord."""


class BatchFileHandler(BaseSourceHandler):
    """Handle GCS file-arrival events for batch sources (CSV/JSON/Parquet/Avro)."""

    def __init__(self, project_id: str) -> None:
        self._storage = storage.Client(project=project_id)

    def handle(self, payload: dict[str, Any]) -> IngestedRecord:
        bucket_name = payload["bucket"]
        object_name = payload["name"]
        gcs_path = f"gs://{bucket_name}/{object_name}"

        fmt = self._detect_format(object_name)
        row_count = self._estimate_rows(bucket_name, object_name, fmt)

        logger.info("BatchFileHandler: %s (%s, ~%d rows)", gcs_path, fmt, row_count)
        return IngestedRecord(
            source_id=payload.get("source_id", "unknown"),
            gcs_path=gcs_path,
            format=fmt,
            row_count=row_count,
            metadata={
                "size_bytes": payload.get("size", 0),
                "content_type": payload.get("contentType", ""),
            },
        )

    @staticmethod
    def _detect_format(name: str) -> str:
        lower = name.lower()
        for ext in ("parquet", "avro", "json", "jsonl", "csv"):
            if lower.endswith(f".{ext}"):
                return "json" if ext == "jsonl" else ext
        return "csv"

    def _estimate_rows(self, bucket: str, blob_name: str, fmt: str) -> int:
        blob = self._storage.bucket(bucket).blob(blob_name)
        blob.reload()
        size = blob.size or 0
        avg_row = {"csv": 200, "json": 500, "parquet": 100, "avro": 100}.get(fmt, 200)
        return max(1, size // avg_row) if size else 0


class StreamHandler(BaseSourceHandler):
    """Handle Pub/Sub streaming messages."""

    def handle(self, payload: dict[str, Any]) -> IngestedRecord:
        data = payload.get("data", {})
        source_id = payload.get("source_id", "unknown")

        if isinstance(data, str):
            data = json.loads(data)

        row_count = len(data) if isinstance(data, list) else 1

        logger.info("StreamHandler: source=%s, rows=%d", source_id, row_count)
        return IngestedRecord(
            source_id=source_id,
            gcs_path="",  # will be set by landing manager after write
            format="json",
            row_count=row_count,
            metadata={"stream_data": data},
        )


class CDCHandler(BaseSourceHandler):
    """Handle Datastream CDC events."""

    def handle(self, payload: dict[str, Any]) -> IngestedRecord:
        source_id = payload.get("source_id", "unknown")
        operation = payload.get("operation", "INSERT")
        change_data = payload.get("data", {})

        try:
            cdc_op = CDCOperation(operation)
        except ValueError:
            cdc_op = CDCOperation.INSERT

        logger.info("CDCHandler: source=%s, op=%s", source_id, cdc_op.value)
        return IngestedRecord(
            source_id=source_id,
            gcs_path="",
            format="json",
            row_count=1,
            metadata={
                "cdc_operation": cdc_op.value,
                "change_data": change_data,
                "change_timestamp": payload.get("change_timestamp", ""),
            },
        )


def get_handler(source_type: str, project_id: str) -> BaseSourceHandler:
    """Factory: return the appropriate handler for *source_type*."""
    handlers: dict[str, type[BaseSourceHandler]] = {
        "batch": BatchFileHandler,
        "stream": StreamHandler,
        "cdc": CDCHandler,
    }
    cls = handlers.get(source_type)
    if cls is None:
        raise ValueError(f"Unknown source type: {source_type}")
    if cls is BatchFileHandler:
        return cls(project_id)
    return cls()
