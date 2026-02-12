"""Ingestion Agent — Bronze layer orchestrator."""

from __future__ import annotations

import json
import logging

from src.agents.base_agent import BaseAgent
from src.agents.ingestion.landing_manager import LandingManager
from src.agents.ingestion.schema_detector import SchemaDetector
from src.agents.ingestion.source_handlers import get_handler
from src.common.config import PipelineConfig, SourceDefinition
from src.common.event_bus import EventBus
from src.common.lineage_tracker import LineageTracker
from src.common.models import (
    EventType,
    EvolutionType,
    Layer,
    PipelineEvent,
    SchemaVersion,
)
from src.common.schema_registry import SchemaRegistry
from src.common.state_manager import StateManager
from src.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class IngestionAgent(BaseAgent):
    agent_name = "ingestion"
    layer = Layer.BRONZE

    def __init__(
        self,
        config: PipelineConfig,
        event_bus: EventBus,
        state_manager: StateManager,
        lineage_tracker: LineageTracker,
        schema_registry: SchemaRegistry,
        gemini_client: GeminiClient,
    ) -> None:
        super().__init__(config, event_bus, state_manager, lineage_tracker)
        self.schema_registry = schema_registry
        self.gemini_client = gemini_client
        self.schema_detector = SchemaDetector(
            config.gcp.project_id,
            sample_rows=config.agents.ingestion.schema_sample_rows,
        )
        self.landing_manager = LandingManager(
            config.gcp.project_id,
            config.storage.bronze_bucket,
        )
        self._sources = {s.id: s for s in config.sources}

    def _subscription_name(self) -> str:
        return self.config.pubsub.ingestion_subscription

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def process_event(self, event: PipelineEvent) -> None:
        if event.event_type != EventType.FILE_ARRIVED:
            logger.debug("Ignoring event type %s", event.event_type.value)
            return

        source_def = self._sources.get(event.source_id)
        if source_def is None:
            logger.warning("Unknown source_id: %s", event.source_id)
            return

        logger.info(
            "Processing %s for source %s (corr=%s)",
            event.event_type.value,
            event.source_id,
            event.correlation_id,
        )

        # 1. Handle source-specific logic.
        handler = get_handler(source_def.type, self.config.gcp.project_id)
        record = handler.handle(event.payload)
        record.source_id = event.source_id

        # 2. Infer / validate schema.
        gcs_path = record.gcs_path or event.payload.get("gcs_path", "")
        if gcs_path:
            new_schema = self.schema_detector.infer_schema(gcs_path, source_def.format)
            new_schema.source_id = event.source_id
        else:
            # Stream/CDC — build schema from payload data.
            new_schema = self._schema_from_payload(event.source_id, record.metadata)

        # 3. Check schema evolution.
        evolution = self.schema_registry.register(event.source_id, new_schema)
        if evolution is not None:
            self._handle_evolution(event, source_def, evolution, new_schema)

        current_schema = self.schema_registry.get_current(event.source_id)
        schema_version = current_schema.version if current_schema else 1

        # 4. Land data in Bronze GCS zone.
        if gcs_path:
            bronze_path = self.landing_manager.land_file(
                event.source_id,
                gcs_path,
                schema_version=schema_version,
                row_count=record.row_count,
            )
        else:
            data = json.dumps(record.metadata.get("stream_data", record.metadata)).encode()
            bronze_path = self.landing_manager.land_stream_data(
                event.source_id,
                data,
                schema_version=schema_version,
                row_count=record.row_count,
            )

        # 5. Record lineage.
        source_ref = gcs_path or f"pubsub:{event.source_id}"
        self.lineage_tracker.record(
            source_ref=source_ref,
            target_ref=bronze_path,
            process_name="ingestion-agent",
            run_id=event.correlation_id,
        )

        # 6. Emit INGESTION_COMPLETE.
        complete_event = PipelineEvent(
            event_type=EventType.INGESTION_COMPLETE,
            source_id=event.source_id,
            layer=Layer.BRONZE,
            correlation_id=event.correlation_id,
            payload={
                "bronze_path": bronze_path,
                "format": source_def.format,
                "schema_version": schema_version,
                "row_count": record.row_count,
            },
        )
        self.event_bus.publish(self.config.pubsub.transformation_topic, complete_event)
        logger.info(
            "Ingestion complete for %s → %s (corr=%s)",
            event.source_id,
            bronze_path,
            event.correlation_id,
        )

    # ------------------------------------------------------------------
    # Schema evolution handling
    # ------------------------------------------------------------------

    def _handle_evolution(
        self,
        event: PipelineEvent,
        source_def: SourceDefinition,
        evolution,
        new_schema: SchemaVersion,
    ) -> None:
        if evolution.evolution_type == EvolutionType.IDENTICAL:
            return

        logger.info(
            "Schema evolution detected for %s: %s",
            event.source_id,
            evolution.evolution_type.value,
        )

        # Emit a SCHEMA_EVOLUTION event regardless of strategy.
        evo_event = PipelineEvent(
            event_type=EventType.SCHEMA_EVOLUTION,
            source_id=event.source_id,
            layer=Layer.BRONZE,
            correlation_id=event.correlation_id,
            payload={
                "evolution_type": evolution.evolution_type.value,
                "added": [c.model_dump() for c in evolution.added_columns],
                "removed": [c.model_dump() for c in evolution.removed_columns],
                "type_changes": evolution.type_changes,
            },
        )
        self.event_bus.publish(self.config.pubsub.ingestion_topic, evo_event)

        strategy = self.config.agents.ingestion.schema_evolution_strategy

        if evolution.evolution_type == EvolutionType.ADDITIVE:
            logger.info("Auto-evolving additive schema change for %s", event.source_id)
            return  # schema already registered

        if evolution.evolution_type == EvolutionType.BREAKING:
            if strategy == "auto":
                # Consult Gemini for guidance.
                recommendation = self.gemini_client.analyze_schema(evolution)
                logger.info(
                    "Gemini schema recommendation for %s: %s",
                    event.source_id,
                    recommendation.get("recommendation"),
                )
                if recommendation.get("recommendation") == "reject":
                    raise RuntimeError(
                        f"Breaking schema change rejected for {event.source_id}: "
                        f"{recommendation.get('reason')}"
                    )
            elif strategy == "pause":
                raise RuntimeError(
                    f"Breaking schema change detected for {event.source_id} — pausing ingestion"
                )
            # strategy == "alert": just log and continue

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _schema_from_payload(source_id: str, metadata: dict) -> SchemaVersion:
        """Build a schema from stream/CDC payload keys."""
        from src.common.models import ColumnDef

        data = metadata.get("stream_data", metadata.get("change_data", metadata))
        if isinstance(data, list):
            data = data[0] if data else {}

        columns = [
            ColumnDef(name=k, data_type="STRING", nullable=True)
            for k in data.keys()
        ]
        schema = SchemaVersion(source_id=source_id, columns=columns)
        schema.compute_fingerprint()
        return schema
