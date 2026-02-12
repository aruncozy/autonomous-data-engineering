"""Business-ready Gold dataset creation with optimized DDL."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from google.cloud import bigquery
from jinja2 import Environment, FileSystemLoader

from src.agents.aggregation.optimization import (
    ClusterRecommendation,
    PartitionRecommendation,
)
from src.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "templates"


class DatasetBuilder:
    """Create Gold-layer BigQuery tables with optimal partitioning and clustering."""

    def __init__(
        self,
        project_id: str,
        dataset: str,
        location: str = "US",
        gemini_client: GeminiClient | None = None,
    ) -> None:
        self._bq = bigquery.Client(project=project_id, location=location)
        self._project_id = project_id
        self._dataset = dataset
        self._gemini = gemini_client
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_table(
        self,
        domain: str,
        entity: str,
        source_table: str,
        columns: list[dict[str, Any]],
        partition: PartitionRecommendation | None = None,
        clustering: ClusterRecommendation | None = None,
        *,
        aggregation_sql: str = "",
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> str:
        """Create a Gold table with optimized DDL + aggregation SQL.

        Returns the fully-qualified table reference.
        """
        table_name = f"gold_{domain}_{entity}"
        table_ref = f"{self._project_id}.{self._dataset}.{table_name}"

        # Generate aggregation SQL if not provided.
        if not aggregation_sql:
            aggregation_sql = self._generate_aggregation_sql(
                source_table, table_ref, columns, partition, clustering,
            )

        # Execute the DDL + insert.
        self._execute(aggregation_sql)

        # Apply metadata.
        self._apply_metadata(table_ref, description, labels or {}, columns)

        logger.info("Built Gold table %s", table_ref)
        return table_ref

    def build_scd2_dimension(
        self,
        domain: str,
        entity: str,
        source_table: str,
        primary_key: list[str],
        columns: list[dict[str, Any]],
    ) -> str:
        """Create an SCD Type 2 dimension table."""
        table_name = f"gold_{domain}_{entity}_dim"
        table_ref = f"{self._project_id}.{self._dataset}.{table_name}"

        pk_expr = ", ".join(primary_key)
        col_names = [c["name"] for c in columns if c["name"] not in primary_key]
        col_expr = ", ".join(col_names)

        sql = f"""\
CREATE OR REPLACE TABLE `{table_ref}` AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY {pk_expr} ORDER BY _loaded_at DESC) AS _rn
    FROM `{source_table}`
)
SELECT
    {pk_expr},
    {col_expr},
    _loaded_at AS effective_from,
    LEAD(_loaded_at) OVER (PARTITION BY {pk_expr} ORDER BY _loaded_at) AS effective_to,
    IF(ROW_NUMBER() OVER (PARTITION BY {pk_expr} ORDER BY _loaded_at DESC) = 1, TRUE, FALSE) AS is_current
FROM ranked
"""
        self._execute(sql)
        logger.info("Built SCD2 dimension %s", table_ref)
        return table_ref

    def build_rollup(
        self,
        domain: str,
        entity: str,
        source_table: str,
        group_columns: list[str],
        metric_expressions: list[str],
        *,
        partition: PartitionRecommendation | None = None,
    ) -> str:
        """Create a pre-aggregated rollup table."""
        table_name = f"gold_{domain}_{entity}_rollup"
        table_ref = f"{self._project_id}.{self._dataset}.{table_name}"

        group_expr = ", ".join(group_columns)
        metric_expr = ", ".join(metric_expressions)

        partition_clause = ""
        if partition and partition.column:
            partition_clause = f"PARTITION BY {partition.partition_type or 'DAY'}({partition.column})"

        sql = f"""\
CREATE OR REPLACE TABLE `{table_ref}`
{partition_clause}
AS
SELECT
    {group_expr},
    {metric_expr}
FROM `{source_table}`
GROUP BY {group_expr}
"""
        self._execute(sql)
        logger.info("Built rollup table %s", table_ref)
        return table_ref

    # ------------------------------------------------------------------
    # SQL generation
    # ------------------------------------------------------------------

    def _generate_aggregation_sql(
        self,
        source_table: str,
        target_table: str,
        columns: list[dict[str, Any]],
        partition: PartitionRecommendation | None,
        clustering: ClusterRecommendation | None,
    ) -> str:
        col_names = [c["name"] for c in columns]

        partition_clause = ""
        if partition and partition.column and partition.partition_type:
            partition_clause = f"PARTITION BY {partition.partition_type}({partition.column})"

        cluster_clause = ""
        if clustering and clustering.columns:
            cluster_clause = f"CLUSTER BY {', '.join(clustering.columns)}"

        try:
            tmpl = self._jinja.get_template("sql_aggregate.sql.j2")
            return tmpl.render(
                source_table=source_table,
                target_table=target_table,
                columns=col_names,
                partition_clause=partition_clause,
                cluster_clause=cluster_clause,
            )
        except Exception:
            logger.info("Template rendering failed — using inline SQL")
            col_expr = ", ".join(col_names) if col_names else "*"
            return (
                f"CREATE OR REPLACE TABLE `{target_table}`\n"
                f"{partition_clause}\n"
                f"{cluster_clause}\n"
                f"AS\n"
                f"SELECT {col_expr}\n"
                f"FROM `{source_table}`"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _apply_metadata(
        self,
        table_ref: str,
        description: str,
        labels: dict[str, str],
        columns: list[dict[str, Any]],
    ) -> None:
        try:
            table = self._bq.get_table(table_ref)
            if description:
                table.description = description
            if labels:
                table.labels = labels

            # Apply column descriptions.
            existing_schema = {f.name: f for f in table.schema}
            new_schema = []
            for field in table.schema:
                col_def = next((c for c in columns if c["name"] == field.name), None)
                desc = col_def.get("description", "") if col_def else ""
                new_schema.append(
                    bigquery.SchemaField(
                        name=field.name,
                        field_type=field.field_type,
                        mode=field.mode,
                        description=desc or field.description,
                    )
                )
            table.schema = new_schema
            self._bq.update_table(table, ["description", "labels", "schema"])
        except Exception:
            logger.warning("Could not apply metadata to %s", table_ref)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self, sql: str) -> None:
        job = self._bq.query(sql)
        job.result()
        logger.info("SQL executed (job=%s)", job.job_id)
