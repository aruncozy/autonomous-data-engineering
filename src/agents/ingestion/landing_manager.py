"""Bronze GCS zone management — partitioned landing, metadata sidecars."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from google.cloud import storage

logger = logging.getLogger(__name__)


class LandingManager:
    """Manage the Bronze-layer GCS bucket structure."""

    def __init__(self, project_id: str, bronze_bucket: str) -> None:
        self._storage = storage.Client(project=project_id)
        self._bucket = self._storage.bucket(bronze_bucket)

    # ------------------------------------------------------------------
    # Landing
    # ------------------------------------------------------------------

    def land_file(
        self,
        source_id: str,
        source_gcs_path: str,
        *,
        schema_version: int = 1,
        row_count: int = 0,
    ) -> str:
        """Copy a raw file into the Bronze partition structure.

        Returns the destination GCS path.
        """
        now = datetime.now(timezone.utc)
        partition = now.strftime("%Y/%m/%d")
        ts_suffix = now.strftime("%H%M%S")
        filename = source_gcs_path.rsplit("/", 1)[-1] if "/" in source_gcs_path else source_gcs_path
        dest_path = f"{source_id}/{partition}/{ts_suffix}_{filename}"

        # Copy the source blob to bronze bucket.
        src_bucket_name, src_blob_name = _parse_gcs(source_gcs_path)
        src_bucket = self._storage.bucket(src_bucket_name)
        src_blob = src_bucket.blob(src_blob_name)

        src_bucket.copy_blob(src_blob, self._bucket, new_name=dest_path)
        dest_uri = f"gs://{self._bucket.name}/{dest_path}"
        logger.info("Landed %s → %s", source_gcs_path, dest_uri)

        # Write metadata sidecar.
        self._write_sidecar(dest_path, source_gcs_path, schema_version, row_count)
        return dest_uri

    def land_stream_data(
        self,
        source_id: str,
        data: bytes,
        *,
        schema_version: int = 1,
        row_count: int = 0,
    ) -> str:
        """Write streaming/CDC data directly to Bronze GCS.

        Returns the destination GCS path.
        """
        now = datetime.now(timezone.utc)
        partition = now.strftime("%Y/%m/%d")
        ts_suffix = now.strftime("%H%M%S%f")
        dest_path = f"{source_id}/{partition}/{ts_suffix}.json"

        blob = self._bucket.blob(dest_path)
        blob.upload_from_string(data, content_type="application/json")

        dest_uri = f"gs://{self._bucket.name}/{dest_path}"
        logger.info("Landed stream data → %s (%d bytes)", dest_uri, len(data))

        self._write_sidecar(dest_path, "stream", schema_version, row_count)
        return dest_uri

    # ------------------------------------------------------------------
    # Metadata sidecar
    # ------------------------------------------------------------------

    def _write_sidecar(
        self,
        dest_path: str,
        source_path: str,
        schema_version: int,
        row_count: int,
    ) -> None:
        sidecar_path = dest_path + "._metadata.json"
        meta = {
            "source": source_path,
            "landed_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": schema_version,
            "row_count": row_count,
            "checksum": "",  # populated below if possible
        }

        # Compute checksum of the landed blob.
        blob = self._bucket.blob(dest_path)
        blob.reload()
        if blob.md5_hash:
            meta["checksum"] = blob.md5_hash

        sidecar_blob = self._bucket.blob(sidecar_path)
        sidecar_blob.upload_from_string(
            json.dumps(meta, indent=2),
            content_type="application/json",
        )

    # ------------------------------------------------------------------
    # Cleanup / archival
    # ------------------------------------------------------------------

    def archive_processed(self, gcs_path: str, archive_bucket_name: str) -> None:
        """Move a processed file from Bronze to an archive bucket."""
        _, blob_path = _parse_gcs(gcs_path)
        blob = self._bucket.blob(blob_path)
        archive_bucket = self._storage.bucket(archive_bucket_name)
        self._bucket.copy_blob(blob, archive_bucket, new_name=blob_path)
        blob.delete()
        logger.info("Archived %s → gs://%s/%s", gcs_path, archive_bucket_name, blob_path)

    def list_unprocessed(self, source_id: str, date: str | None = None) -> list[str]:
        """List Bronze blobs for *source_id*, optionally filtered by date (YYYY/MM/DD)."""
        prefix = f"{source_id}/"
        if date:
            prefix += f"{date}/"
        blobs = self._bucket.list_blobs(prefix=prefix)
        return [
            f"gs://{self._bucket.name}/{b.name}"
            for b in blobs
            if not b.name.endswith("._metadata.json")
        ]


def _parse_gcs(path: str) -> tuple[str, str]:
    path = path.removeprefix("gs://")
    bucket, _, blob = path.partition("/")
    return bucket, blob
