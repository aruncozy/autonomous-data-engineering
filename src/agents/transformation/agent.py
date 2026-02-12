"""Transformation Agent — Silver layer orchestrator."""

from __future__ import annotations

import json
import logging
from typing import Any

import yaml

from src.agents.base_agent import BaseAgent
from src.agents.transformation.biglake_manager import BigLakeManager
from src.agents.transformation.code_generator import CodeGenerator
from src.agents.transformation.quality_engine import QualityEngine
from src.common.config import PipelineConfig
from src.common.event_bus import EventBus
from src.common.lineage_tracker import LineageTracker
from src.common.models import (
    ColumnDef,
    EventType,
    Layer,
    PipelineEvent,
    QualityReport,
)
from src.common.schema_registry import SchemaRegistry
from src.common.state_manager import StateManager
from src.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class TransformationAgent(BaseAgent):
    agent_name = "transformation"
    layer = Layer.SILVER

    def __init__(
        self,
        config: PipelineConfig,
        event_bus: EventBus,
        state_manager: StateManager,
        lineage_tracker: LineageTracker,
        schema_registry: SchemaRegistry,
        gemini_client: GeminiClient,
        quality_rules: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, event_bus, state_manager, lineage_tracker)
        self.schema_registry = schema_registry
        self.gemini_client = gemini_client
        self.quality_engine = QualityEngine(config.gcp.project_id, config.bigquery.location)
        self.code_generator = CodeGenerator(
            gemini_client,
            strategy=config.agents.transformation.code_gen_strategy,
        )
        self.biglake_manager = BigLakeManager(
            project_id=config.gcp.project_id,
            dataset=config.bigquery.silver_dataset,
            location=config.bigquery.location,
            biglake_connection=config.agents.transformation.biglake_connection,
        )
        self._quality_rules = quality_rules or {}
        self._sources = {s.id: s for s in config.sources}

    def _subscription_name(self) -> str:
        return self.config.pubsub.transformation_subscription

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def process_event(self, event: PipelineEvent) -> None:
        if event.event_type != EventType.INGESTION_COMPLETE:
            logger.debug("Ignoring event type %s", event.event_type.value)
            return

        source_id = event.source_id
        bronze_path = event.payload.get("bronze_path", "")
        fmt = event.payload.get("format", "csv")
        schema_version = event.payload.get("schema_version", 1)

        logger.info(
            "Processing INGESTION_COMPLETE for %s (corr=%s, bronze=%s)",
            source_id,
            event.correlation_id,
            bronze_path,
        )

        source_def = self._sources.get(source_id)
        if source_def is None:
            logger.warning("Unknown source_id: %s", source_id)
            return

        # 1. Load bronze data into a temp BQ table for quality checks.
        temp_table = self._load_bronze_to_temp(source_id, bronze_path, fmt)

        # 2. Run quality checks.
        rules = self._collect_rules(source_id)
        source_cfg = {
            "source_id": source_id,
            "primary_key": source_def.schema_hints.primary_key,
            "timestamp_column": source_def.schema_hints.timestamp_column,
            "critical_columns": source_def.quality.critical_columns,
        }
        report = self.quality_engine.run_checks(temp_table, rules, source_cfg)

        # 3. Quality gate.
        threshold = self.config.agents.transformation.quality_threshold
        if report.score < threshold:
            logger.warning(
                "Quality score %.2f < threshold %.2f for %s — routing to DLQ",
                report.score,
                threshold,
                source_id,
            )
            self._emit_quality_failure(event, report)
            return

        # 4. Get current schema.
        current = self.schema_registry.get_current(source_id)
        schema_dict: dict[str, Any] = {}
        columns: list[ColumnDef] = []
        if current:
            columns = current.columns
            schema_dict = current.model_dump(mode="json")

        # 5. Generate transform code.
        target_table = f"{self.config.gcp.project_id}.{self.config.bigquery.silver_dataset}.silver_{source_id}_main"
        sql = self.code_generator.generate_sql(
            source_table=temp_table,
            target_table=target_table,
            schema=schema_dict,
            primary_key=source_def.schema_hints.primary_key,
            timestamp_column=source_def.schema_hints.timestamp_column,
        )

        # 6. Execute transform.
        self._execute_sql(sql)

        # 7. Create / update BigLake table.
        silver_gcs_uri = f"gs://{self.config.storage.silver_bucket}/{source_id}/"
        table_ref = self.biglake_manager.create_or_update_table(
            source_id=source_id,
            entity="main",
            columns=columns,
            gcs_uri=silver_gcs_uri,
            fmt=fmt.upper() if fmt in ("parquet", "avro", "orc") else "PARQUET",
            partition_column=source_def.schema_hints.partition_column,
        )

        # 8. Record lineage.
        self.lineage_tracker.record(
            source_ref=bronze_path,
            target_ref=f"bigquery:{table_ref}",
            process_name="transformation-agent",
            run_id=event.correlation_id,
        )

        # 9. Emit TRANSFORMATION_COMPLETE.
        complete_event = PipelineEvent(
            event_type=EventType.TRANSFORMATION_COMPLETE,
            source_id=source_id,
            layer=Layer.SILVER,
            correlation_id=event.correlation_id,
            payload={
                "silver_table": table_ref,
                "quality_score": report.score,
                "schema_version": schema_version,
                "row_count": event.payload.get("row_count", 0),
            },
        )
        self.event_bus.publish(self.config.pubsub.aggregation_topic, complete_event)
        logger.info(
            "Transformation complete for %s → %s (corr=%s, quality=%.2f)",
            source_id,
            table_ref,
            event.correlation_id,
            report.score,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_bronze_to_temp(self, source_id: str, gcs_path: str, fmt: str) -> str:
        """Load Bronze GCS file into a temporary BigQuery table."""
        from google.cloud import bigquery as bq

        client = bq.Client(project=self.config.gcp.project_id, location=self.config.bigquery.location)
        temp_table = f"{self.config.gcp.project_id}.{self.config.bigquery.bronze_dataset}._temp_{source_id}"

        fmt_map = {
            "csv": bq.SourceFormat.CSV,
            "json": bq.SourceFormat.NEWLINE_DELIMITED_JSON,
            "parquet": bq.SourceFormat.PARQUET,
            "avro": bq.SourceFormat.AVRO,
        }

        job_config = bq.LoadJobConfig(
            source_format=fmt_map.get(fmt, bq.SourceFormat.CSV),
            autodetect=True,
            write_disposition=bq.WriteDisposition.WRITE_TRUNCATE,
        )
        job = client.load_table_from_uri(gcs_path, temp_table, job_config=job_config)
        job.result()
        logger.info("Loaded %s → %s", gcs_path, temp_table)
        return temp_table

    def _execute_sql(self, sql: str) -> None:
        """Run a SQL transform via BigQuery."""
        from google.cloud import bigquery as bq

        client = bq.Client(project=self.config.gcp.project_id, location=self.config.bigquery.location)
        job = client.query(sql)
        job.result()
        logger.info("SQL transform executed (job=%s)", job.job_id)

    def _collect_rules(self, source_id: str) -> list[dict]:
        """Merge global + source-specific quality rules."""
        rules: list[dict] = []
        global_rules = self._quality_rules.get("global_rules", {})
        for category in ("completeness", "uniqueness", "validity", "freshness", "volume"):
            rules.extend(global_rules.get(category, []))
        source_rules = self._quality_rules.get("source_rules", {}).get(source_id, [])
        rules.extend(source_rules)
        return rules

    def _emit_quality_failure(self, event: PipelineEvent, report: QualityReport) -> None:
        fail_event = PipelineEvent(
            event_type=EventType.QUALITY_FAILURE,
            source_id=event.source_id,
            layer=Layer.SILVER,
            correlation_id=event.correlation_id,
            payload={
                "quality_score": report.score,
                "checks_passed": report.checks_passed,
                "checks_failed": report.checks_failed,
                "details": [d.model_dump() for d in report.details],
            },
        )
        self.event_bus.publish(self.config.pubsub.transformation_topic, fail_event)
