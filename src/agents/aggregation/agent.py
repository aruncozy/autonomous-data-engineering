"""Aggregation Agent — Gold layer orchestrator."""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.aggregation.dataset_builder import DatasetBuilder
from src.agents.aggregation.optimization import OptimizationEngine
from src.agents.aggregation.query_analyzer import QueryAnalyzer
from src.common.config import PipelineConfig
from src.common.event_bus import EventBus
from src.common.lineage_tracker import LineageTracker
from src.common.models import EventType, Layer, PipelineEvent
from src.common.schema_registry import SchemaRegistry
from src.common.state_manager import StateManager
from src.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class AggregationAgent(BaseAgent):
    agent_name = "aggregation"
    layer = Layer.GOLD

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

        self.query_analyzer = QueryAnalyzer(
            config.gcp.project_id, config.bigquery.location,
        )
        self.optimization_engine = OptimizationEngine(
            project_id=config.gcp.project_id,
            location=config.bigquery.location,
            gemini_client=gemini_client,
            max_clustering_columns=config.agents.aggregation.max_clustering_columns,
            mv_staleness_hours=config.agents.aggregation.mv_staleness_hours,
            mv_cost_threshold=config.agents.aggregation.mv_cost_benefit_threshold,
        )
        self.dataset_builder = DatasetBuilder(
            project_id=config.gcp.project_id,
            dataset=config.bigquery.gold_dataset,
            location=config.bigquery.location,
            gemini_client=gemini_client,
        )

    def _subscription_name(self) -> str:
        return self.config.pubsub.aggregation_subscription

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def process_event(self, event: PipelineEvent) -> None:
        if event.event_type != EventType.TRANSFORMATION_COMPLETE:
            logger.debug("Ignoring event type %s", event.event_type.value)
            return

        source_id = event.source_id
        silver_table = event.payload.get("silver_table", "")

        logger.info(
            "Processing TRANSFORMATION_COMPLETE for %s (corr=%s, silver=%s)",
            source_id,
            event.correlation_id,
            silver_table,
        )

        # 1. Analyze access patterns.
        dataset = self.config.bigquery.silver_dataset
        table_name = silver_table.rsplit(".", 1)[-1] if "." in silver_table else silver_table
        patterns = self.query_analyzer.analyze(
            dataset,
            table_name,
            days=self.config.agents.aggregation.query_history_days,
        )

        # Enrich with cardinality.
        if patterns:
            patterns = self.query_analyzer.enrich_cardinality(silver_table, patterns)

        # 2. Get table schema.
        current_schema = self.schema_registry.get_current(source_id)
        table_schema: list[dict[str, Any]] = []
        columns: list[dict[str, Any]] = []
        if current_schema:
            table_schema = [
                {"name": c.name, "type": c.data_type} for c in current_schema.columns
            ]
            columns = [c.model_dump() for c in current_schema.columns]

        # 3. Determine optimal partitioning + clustering.
        partition = self.optimization_engine.recommend_partition(table_schema, patterns)
        clustering = self.optimization_engine.recommend_clustering(patterns)

        logger.info(
            "Optimization for %s: partition=%s(%s), cluster=%s",
            source_id,
            partition.column,
            partition.partition_type,
            clustering.columns,
        )

        # 4. Build Gold dataset.
        gold_table = self.dataset_builder.build_table(
            domain=self._infer_domain(source_id),
            entity=source_id,
            source_table=silver_table,
            columns=columns,
            partition=partition,
            clustering=clustering,
            description=f"Gold table for {source_id}",
            labels={"source": source_id, "layer": "gold"},
        )

        # 5. Evaluate and create materialized views.
        mv_recs = self.optimization_engine.recommend_materialized_views(
            self.config.bigquery.gold_dataset, gold_table.rsplit(".", 1)[-1], patterns,
        )
        for mv in mv_recs:
            if mv.create:
                try:
                    self.optimization_engine.create_materialized_view(
                        self.config.bigquery.gold_dataset, mv,
                    )
                except Exception:
                    logger.warning("Failed to create MV %s", mv.name, exc_info=True)

        # 6. Record lineage.
        self.lineage_tracker.record(
            source_ref=f"bigquery:{silver_table}",
            target_ref=f"bigquery:{gold_table}",
            process_name="aggregation-agent",
            run_id=event.correlation_id,
        )

        # 7. Emit AGGREGATION_COMPLETE.
        complete_event = PipelineEvent(
            event_type=EventType.AGGREGATION_COMPLETE,
            source_id=source_id,
            layer=Layer.GOLD,
            correlation_id=event.correlation_id,
            payload={
                "gold_table": gold_table,
                "partition_column": partition.column,
                "clustering_columns": clustering.columns,
                "materialized_views": [mv.name for mv in mv_recs if mv.create],
            },
        )
        self.event_bus.publish(self.config.pubsub.aggregation_topic, complete_event)
        logger.info(
            "Aggregation complete for %s → %s (corr=%s)",
            source_id,
            gold_table,
            event.correlation_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_domain(source_id: str) -> str:
        """Derive a domain name from the source_id."""
        parts = source_id.split("_")
        return parts[0] if parts else "default"
