"""Unit tests for the Aggregation Agent's core logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.common.models import AccessPattern, EventType, Layer, PipelineEvent


# ---------------------------------------------------------------------------
# Query analyzer — pattern extraction
# ---------------------------------------------------------------------------

class TestQueryAnalyzer:
    def test_extract_where_columns(self):
        from src.agents.aggregation.query_analyzer import QueryAnalyzer

        cols = QueryAnalyzer._extract_where_columns(
            "SELECT * FROM t WHERE status = 'active' AND created_at > '2025-01-01'"
        )
        assert "status" in cols
        assert "created_at" in cols

    def test_extract_group_by_columns(self):
        from src.agents.aggregation.query_analyzer import QueryAnalyzer

        cols = QueryAnalyzer._extract_group_by_columns(
            "SELECT region, COUNT(*) FROM t GROUP BY region, status ORDER BY 1"
        )
        assert "region" in cols
        assert "status" in cols

    def test_extract_join_columns(self):
        from src.agents.aggregation.query_analyzer import QueryAnalyzer

        cols = QueryAnalyzer._extract_join_columns(
            "SELECT * FROM a JOIN b ON a.user_id = b.user_id"
        )
        # Should find user_id (appears in both sides).
        assert "user_id" in cols

    def test_keywords_excluded(self):
        from src.agents.aggregation.query_analyzer import QueryAnalyzer

        cols = QueryAnalyzer._extract_where_columns(
            "SELECT * FROM t WHERE id IS NOT NULL AND name LIKE '%test%'"
        )
        assert "id" in cols
        assert "name" in cols
        # SQL keywords should not appear.
        assert "is" not in cols
        assert "not" not in cols


# ---------------------------------------------------------------------------
# Optimization engine
# ---------------------------------------------------------------------------

class TestOptimizationEngine:
    def test_recommend_partition_prefers_date(self):
        from src.agents.aggregation.optimization import OptimizationEngine

        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine._max_cluster = 4

        schema = [
            {"name": "id", "type": "INT64"},
            {"name": "created_at", "type": "TIMESTAMP"},
            {"name": "name", "type": "STRING"},
        ]
        patterns = [
            AccessPattern(column_name="created_at", filter_frequency=10, cost_impact_bytes=1_000_000_000),
            AccessPattern(column_name="id", filter_frequency=5),
        ]

        rec = engine.recommend_partition(schema, patterns)
        assert rec.column == "created_at"
        assert rec.partition_type == "DAY"

    def test_recommend_partition_no_date_column(self):
        from src.agents.aggregation.optimization import OptimizationEngine

        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine._max_cluster = 4

        schema = [
            {"name": "id", "type": "INT64"},
            {"name": "name", "type": "STRING"},
        ]
        patterns = [
            AccessPattern(column_name="id", filter_frequency=10),
        ]

        rec = engine.recommend_partition(schema, patterns)
        assert rec.column is None

    def test_recommend_clustering(self):
        from src.agents.aggregation.optimization import OptimizationEngine

        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine._max_cluster = 4

        patterns = [
            AccessPattern(column_name="region", filter_frequency=20, cardinality=500),
            AccessPattern(column_name="status", filter_frequency=15, cardinality=5),
            AccessPattern(column_name="user_id", filter_frequency=10, cardinality=1_000_000),
            AccessPattern(column_name="category", filter_frequency=8, cardinality=500),
            AccessPattern(column_name="unused", filter_frequency=0),
        ]

        rec = engine.recommend_clustering(patterns)
        assert len(rec.columns) <= 4
        assert rec.columns[0] == "region"  # highest score (freq * medium cardinality boost)
        assert "unused" not in rec.columns

    def test_mv_recommendation_threshold(self):
        from src.agents.aggregation.optimization import OptimizationEngine

        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine._project_id = "test-project"
        engine._mv_threshold = 0.3

        patterns = [
            AccessPattern(column_name="region", group_by_frequency=10),
            AccessPattern(column_name="status", group_by_frequency=8),
        ]

        recs = engine.recommend_materialized_views("gold", "test_table", patterns)
        assert len(recs) >= 1
        assert recs[0].create is True
        assert "region" in recs[0].query


# ---------------------------------------------------------------------------
# Dataset builder SQL generation
# ---------------------------------------------------------------------------

class TestDatasetBuilder:
    def test_build_table_sql_generation(self, mock_gemini_client):
        from src.agents.aggregation.dataset_builder import DatasetBuilder
        from src.agents.aggregation.optimization import ClusterRecommendation, PartitionRecommendation

        builder = DatasetBuilder.__new__(DatasetBuilder)
        builder._project_id = "test-project"
        builder._dataset = "gold"
        builder._gemini = mock_gemini_client

        from jinja2 import Environment, FileSystemLoader
        from pathlib import Path

        templates_dir = Path(__file__).resolve().parents[1] / "templates"
        builder._jinja = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        partition = PartitionRecommendation(column="created_at", partition_type="DAY")
        clustering = ClusterRecommendation(columns=["region", "status"])

        sql = builder._generate_aggregation_sql(
            source_table="test-project.silver.silver_test_main",
            target_table="test-project.gold.gold_test",
            columns=[
                {"name": "id"},
                {"name": "name"},
                {"name": "region"},
                {"name": "status"},
                {"name": "created_at"},
            ],
            partition=partition,
            clustering=clustering,
        )

        assert "gold_test" in sql
        assert "DAY(created_at)" in sql
        assert "region, status" in sql


# ---------------------------------------------------------------------------
# Aggregation agent event processing
# ---------------------------------------------------------------------------

class TestAggregationAgent:
    def test_ignores_non_transformation_events(
        self, test_config, mock_event_bus, mock_state_manager,
        mock_lineage_tracker, mock_schema_registry, mock_gemini_client,
    ):
        from src.agents.aggregation.agent import AggregationAgent

        agent = AggregationAgent(
            config=test_config,
            event_bus=mock_event_bus,
            state_manager=mock_state_manager,
            lineage_tracker=mock_lineage_tracker,
            schema_registry=mock_schema_registry,
            gemini_client=mock_gemini_client,
        )

        event = PipelineEvent(
            event_type=EventType.FILE_ARRIVED,
            source_id="test_source",
            layer=Layer.BRONZE,
        )
        agent.process_event(event)
        mock_event_bus.publish.assert_not_called()

    def test_infer_domain(self):
        from src.agents.aggregation.agent import AggregationAgent

        assert AggregationAgent._infer_domain("sales_transactions") == "sales"
        assert AggregationAgent._infer_domain("customers") == "customers"
