"""BigLake managed table operations for the Silver layer."""

from __future__ import annotations

import logging
from typing import Any

from google.cloud import bigquery

from src.common.models import ColumnDef

logger = logging.getLogger(__name__)

# BigQuery type mapping from our internal representation
_TYPE_MAP = {
    "STRING": "STRING",
    "INT64": "INT64",
    "FLOAT64": "FLOAT64",
    "BOOL": "BOOL",
    "NUMERIC": "NUMERIC",
    "DATE": "DATE",
    "TIMESTAMP": "TIMESTAMP",
    "BYTES": "BYTES",
}


class BigLakeManager:
    """Create and manage BigLake managed tables over Silver GCS data."""

    def __init__(
        self,
        project_id: str,
        dataset: str,
        location: str = "US",
        biglake_connection: str = "",
    ) -> None:
        self._bq = bigquery.Client(project=project_id, location=location)
        self._project_id = project_id
        self._dataset = dataset
        self._location = location
        self._connection = biglake_connection

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def create_or_update_table(
        self,
        source_id: str,
        entity: str,
        columns: list[ColumnDef],
        gcs_uri: str,
        *,
        fmt: str = "PARQUET",
        partition_column: str = "",
    ) -> str:
        """Create (or evolve) a BigLake managed table.

        Returns the fully-qualified table reference.
        """
        table_name = f"silver_{source_id}_{entity}"
        table_ref = f"{self._project_id}.{self._dataset}.{table_name}"

        if self._table_exists(table_ref):
            self._evolve_table(table_ref, columns)
        else:
            self._create_table(table_ref, columns, gcs_uri, fmt, partition_column)

        return table_ref

    def _create_table(
        self,
        table_ref: str,
        columns: list[ColumnDef],
        gcs_uri: str,
        fmt: str,
        partition_column: str,
    ) -> None:
        bq_schema = [
            bigquery.SchemaField(
                name=c.name,
                field_type=_TYPE_MAP.get(c.data_type, "STRING"),
                mode="NULLABLE" if c.nullable else "REQUIRED",
                description=c.description,
            )
            for c in columns
        ]

        table = bigquery.Table(table_ref, schema=bq_schema)

        # External data configuration for BigLake.
        ext = bigquery.ExternalConfig(self._format_to_source(fmt))
        ext.source_uris = [gcs_uri]
        ext.autodetect = False
        if self._connection:
            ext.connection_id = self._connection

        table.external_data_configuration = ext

        if partition_column:
            table.time_partitioning = bigquery.TimePartitioning(
                field=partition_column,
            )

        table = self._bq.create_table(table)
        logger.info("Created BigLake table %s → %s", table_ref, gcs_uri)

    # ------------------------------------------------------------------
    # Schema evolution
    # ------------------------------------------------------------------

    def _evolve_table(self, table_ref: str, columns: list[ColumnDef]) -> None:
        """Add new columns to an existing table (additive evolution only)."""
        existing = self._bq.get_table(table_ref)
        existing_names = {f.name for f in existing.schema}

        new_fields = [
            bigquery.SchemaField(
                name=c.name,
                field_type=_TYPE_MAP.get(c.data_type, "STRING"),
                mode="NULLABLE",
                description=c.description,
            )
            for c in columns
            if c.name not in existing_names
        ]

        if not new_fields:
            return

        updated_schema = list(existing.schema) + new_fields
        existing.schema = updated_schema
        self._bq.update_table(existing, ["schema"])
        logger.info(
            "Evolved table %s: added %s",
            table_ref,
            [f.name for f in new_fields],
        )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    def load_data(
        self,
        table_ref: str,
        gcs_uri: str,
        *,
        fmt: str = "PARQUET",
        write_disposition: str = "WRITE_APPEND",
    ) -> str:
        """Load data from GCS into the BigLake table.  Returns job ID."""
        job_config = bigquery.LoadJobConfig(
            source_format=self._format_to_source(fmt),
            write_disposition=write_disposition,
        )

        job = self._bq.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
        job.result()  # wait
        logger.info("Loaded %s into %s (job=%s)", gcs_uri, table_ref, job.job_id)
        return job.job_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _table_exists(self, table_ref: str) -> bool:
        try:
            self._bq.get_table(table_ref)
            return True
        except Exception:
            return False

    @staticmethod
    def _format_to_source(fmt: str) -> str:
        mapping = {
            "PARQUET": bigquery.SourceFormat.PARQUET,
            "parquet": bigquery.SourceFormat.PARQUET,
            "ORC": bigquery.SourceFormat.ORC,
            "orc": bigquery.SourceFormat.ORC,
            "AVRO": bigquery.SourceFormat.AVRO,
            "avro": bigquery.SourceFormat.AVRO,
            "CSV": bigquery.SourceFormat.CSV,
            "csv": bigquery.SourceFormat.CSV,
            "JSON": bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            "json": bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        }
        return mapping.get(fmt, bigquery.SourceFormat.PARQUET)
