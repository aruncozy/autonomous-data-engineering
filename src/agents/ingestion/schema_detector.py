"""Schema inference and evolution detection for incoming data."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from google.cloud import storage

from src.common.models import ColumnDef, EvolutionType, SchemaEvolution, SchemaVersion

logger = logging.getLogger(__name__)

# BigQuery-compatible type mapping
_PYTHON_TO_BQ: dict[str, str] = {
    "str": "STRING",
    "int": "INT64",
    "float": "FLOAT64",
    "bool": "BOOL",
    "NoneType": "STRING",
}


class SchemaDetector:
    """Infer schemas from raw files and detect changes."""

    def __init__(self, project_id: str, sample_rows: int = 1000) -> None:
        self._storage = storage.Client(project=project_id)
        self._sample_rows = sample_rows

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer_schema(self, gcs_path: str, fmt: str) -> SchemaVersion:
        """Sample up to *sample_rows* from *gcs_path* and infer a schema.

        Supported formats: csv, json, parquet, avro.
        """
        data = self._read_sample(gcs_path)

        if fmt == "csv":
            columns = self._infer_csv(data)
        elif fmt == "json":
            columns = self._infer_json(data)
        elif fmt in ("parquet", "avro"):
            columns = self._infer_columnar(gcs_path, fmt)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        schema = SchemaVersion(
            source_id="",  # caller sets this
            columns=columns,
        )
        schema.compute_fingerprint()
        return schema

    # ------------------------------------------------------------------
    # Format-specific inference
    # ------------------------------------------------------------------

    def _infer_csv(self, raw: bytes) -> list[ColumnDef]:
        text = raw.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        rows = [r for _, r in zip(range(self._sample_rows), reader)]
        if not rows:
            return []
        return self._columns_from_dicts(rows)

    def _infer_json(self, raw: bytes) -> list[ColumnDef]:
        text = raw.decode("utf-8", errors="replace").strip()
        # Handle JSON-lines or JSON array.
        if text.startswith("["):
            records = json.loads(text)[: self._sample_rows]
        else:
            records = []
            for i, line in enumerate(text.splitlines()):
                if i >= self._sample_rows:
                    break
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return []
        # Flatten nested fields.
        flat = [self._flatten(r) for r in records]
        return self._columns_from_dicts(flat)

    def _infer_columnar(self, gcs_path: str, fmt: str) -> list[ColumnDef]:
        """Infer from Parquet/Avro by reading native schema metadata."""
        try:
            import pyarrow.parquet as pq
            from google.cloud.storage import Blob

            bucket_name, blob_path = _parse_gcs(gcs_path)
            bucket = self._storage.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            local_tmp = f"/tmp/_schema_sample.{fmt}"
            blob.download_to_filename(local_tmp)

            if fmt == "parquet":
                pf = pq.read_schema(local_tmp)
                return [
                    ColumnDef(
                        name=field.name,
                        data_type=_arrow_to_bq(str(field.type)),
                        nullable=field.nullable,
                    )
                    for field in pf
                ]
        except ImportError:
            logger.warning("pyarrow not available — falling back to JSON inference")
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_sample(self, gcs_path: str) -> bytes:
        bucket_name, blob_path = _parse_gcs(gcs_path)
        bucket = self._storage.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        # Download first 10 MB at most for sampling.
        return blob.download_as_bytes(start=0, end=10 * 1024 * 1024)

    @staticmethod
    def _columns_from_dicts(rows: list[dict[str, Any]]) -> list[ColumnDef]:
        all_keys: dict[str, str] = {}
        nullable_keys: set[str] = set()

        for row in rows:
            for key, value in row.items():
                py_type = type(value).__name__
                bq_type = _PYTHON_TO_BQ.get(py_type, "STRING")
                # Promote: keep the widest type seen.
                existing = all_keys.get(key)
                if existing is None:
                    all_keys[key] = bq_type
                elif existing != bq_type and bq_type != "STRING":
                    all_keys[key] = _wider(existing, bq_type)
                if value is None:
                    nullable_keys.add(key)

        return [
            ColumnDef(name=k, data_type=v, nullable=k in nullable_keys)
            for k, v in all_keys.items()
        ]

    @staticmethod
    def _flatten(obj: dict, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in obj.items():
            full = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(SchemaDetector._flatten(v, prefix=f"{full}."))
            else:
                out[full] = v
        return out

    # ------------------------------------------------------------------
    # Schema comparison (delegates to SchemaRegistry for full diff)
    # ------------------------------------------------------------------

    @staticmethod
    def compare_schemas(old: SchemaVersion, new: SchemaVersion) -> SchemaEvolution:
        from src.common.schema_registry import SchemaRegistry
        return SchemaRegistry.detect_evolution(old, new)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_gcs(path: str) -> tuple[str, str]:
    path = path.removeprefix("gs://")
    bucket, _, blob = path.partition("/")
    return bucket, blob


_TYPE_WIDTH = {"BOOL": 0, "INT64": 1, "FLOAT64": 2, "NUMERIC": 3, "STRING": 4}


def _wider(a: str, b: str) -> str:
    return a if _TYPE_WIDTH.get(a, 4) >= _TYPE_WIDTH.get(b, 4) else b


def _arrow_to_bq(arrow_type: str) -> str:
    mapping = {
        "int8": "INT64", "int16": "INT64", "int32": "INT64", "int64": "INT64",
        "uint8": "INT64", "uint16": "INT64", "uint32": "INT64", "uint64": "INT64",
        "float": "FLOAT64", "float16": "FLOAT64", "float32": "FLOAT64",
        "float64": "FLOAT64", "double": "FLOAT64",
        "bool": "BOOL", "string": "STRING", "utf8": "STRING", "large_string": "STRING",
        "date32": "DATE", "date64": "DATE",
        "timestamp[ns]": "TIMESTAMP", "timestamp[us]": "TIMESTAMP",
        "timestamp[ms]": "TIMESTAMP", "timestamp[s]": "TIMESTAMP",
    }
    return mapping.get(arrow_type, "STRING")
