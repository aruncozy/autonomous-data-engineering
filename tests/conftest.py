"""Shared test fixtures — mock GCP clients, sample data, test configs."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.common.config import (
    AgentsConfig,
    AggregationAgentConfig,
    BigQueryConfig,
    FirestoreConfig,
    GCPConfig,
    HealthCheckConfig,
    IngestionAgentConfig,
    LLMConfig,
    PipelineConfig,
    PubSubConfig,
    SourceDefinition,
    SourceQuality,
    SourceSchemaHints,
    StorageConfig,
    TransformationAgentConfig,
)
from src.common.models import (
    AgentState,
    ColumnDef,
    EventType,
    Layer,
    PipelineEvent,
    QualityCheckResult,
    QualityReport,
    SchemaVersion,
)


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

@pytest.fixture
def test_config() -> PipelineConfig:
    return PipelineConfig(
        gcp=GCPConfig(project_id="test-project", region="us-central1"),
        storage=StorageConfig(
            bronze_bucket="test-bronze",
            silver_bucket="test-silver",
            dlq_bucket="test-dlq",
            temp_bucket="test-temp",
        ),
        pubsub=PubSubConfig(
            ingestion_topic="ingestion-events",
            transformation_topic="transformation-events",
            aggregation_topic="aggregation-events",
            ingestion_subscription="ingestion-events-sub",
            transformation_subscription="transformation-events-sub",
            aggregation_subscription="aggregation-events-sub",
        ),
        bigquery=BigQueryConfig(bronze_dataset="bronze", silver_dataset="silver", gold_dataset="gold"),
        firestore=FirestoreConfig(),
        agents=AgentsConfig(
            ingestion=IngestionAgentConfig(),
            transformation=TransformationAgentConfig(quality_threshold=0.85),
            aggregation=AggregationAgentConfig(),
        ),
        llm=LLMConfig(model="gemini-2.0-flash"),
        health_check=HealthCheckConfig(port=0),  # port 0 = don't bind in tests
        sources=[
            SourceDefinition(
                id="test_source",
                name="Test Source",
                type="batch",
                format="csv",
                location="gs://test-landing/test/",
                schema_hints=SourceSchemaHints(
                    primary_key=["id"],
                    timestamp_column="created_at",
                    partition_column="created_at",
                ),
                quality=SourceQuality(
                    min_score=0.85,
                    critical_columns=["id", "name"],
                    max_null_ratio=0.05,
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Mock GCP services
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.publish.return_value = "msg-123"
    return bus


@pytest.fixture
def mock_state_manager():
    mgr = MagicMock()
    mgr.load_checkpoint.return_value = None
    return mgr


@pytest.fixture
def mock_lineage_tracker():
    return MagicMock()


@pytest.fixture
def mock_schema_registry():
    registry = MagicMock()
    registry.get_current.return_value = SchemaVersion(
        source_id="test_source",
        version=1,
        columns=[
            ColumnDef(name="id", data_type="INT64", nullable=False),
            ColumnDef(name="name", data_type="STRING", nullable=True),
            ColumnDef(name="created_at", data_type="TIMESTAMP", nullable=True),
        ],
    )
    registry.register.return_value = None
    return registry


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    client.analyze_schema.return_value = {
        "recommendation": "auto_evolve",
        "reason": "Additive change is safe",
        "migration_steps": [],
    }
    client.generate_transform_code.return_value = {
        "code": "SELECT * FROM source",
        "explanation": "Simple pass-through",
    }
    client.recommend_partitioning.return_value = {
        "partition_column": "created_at",
        "partition_type": "DAY",
        "clustering_columns": ["id"],
        "reason": "Date column is best for partitioning",
    }
    return client


# ---------------------------------------------------------------------------
# Sample data & events
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_file_arrived_event() -> PipelineEvent:
    return PipelineEvent(
        event_type=EventType.FILE_ARRIVED,
        source_id="test_source",
        layer=Layer.BRONZE,
        correlation_id="corr-001",
        payload={
            "bucket": "test-landing",
            "name": "test/data.csv",
            "source_id": "test_source",
            "size": 10240,
        },
    )


@pytest.fixture
def sample_ingestion_complete_event() -> PipelineEvent:
    return PipelineEvent(
        event_type=EventType.INGESTION_COMPLETE,
        source_id="test_source",
        layer=Layer.BRONZE,
        correlation_id="corr-001",
        payload={
            "bronze_path": "gs://test-bronze/test_source/2025/01/15/120000_data.csv",
            "format": "csv",
            "schema_version": 1,
            "row_count": 500,
        },
    )


@pytest.fixture
def sample_transformation_complete_event() -> PipelineEvent:
    return PipelineEvent(
        event_type=EventType.TRANSFORMATION_COMPLETE,
        source_id="test_source",
        layer=Layer.SILVER,
        correlation_id="corr-001",
        payload={
            "silver_table": "test-project.silver.silver_test_source_main",
            "quality_score": 0.95,
            "schema_version": 1,
            "row_count": 480,
        },
    )


@pytest.fixture
def sample_schema() -> SchemaVersion:
    schema = SchemaVersion(
        source_id="test_source",
        version=1,
        columns=[
            ColumnDef(name="id", data_type="INT64", nullable=False),
            ColumnDef(name="name", data_type="STRING", nullable=True),
            ColumnDef(name="amount", data_type="FLOAT64", nullable=True),
            ColumnDef(name="created_at", data_type="TIMESTAMP", nullable=True),
        ],
    )
    schema.compute_fingerprint()
    return schema


@pytest.fixture
def sample_quality_report() -> QualityReport:
    return QualityReport(
        source_id="test_source",
        checks_passed=8,
        checks_failed=2,
        score=0.88,
        details=[
            QualityCheckResult(name="null_check", category="completeness", passed=True, score=0.98),
            QualityCheckResult(name="pk_unique", category="uniqueness", passed=True, score=1.0),
            QualityCheckResult(name="date_valid", category="validity", passed=True, score=0.99),
            QualityCheckResult(name="freshness", category="freshness", passed=False, score=0.0),
        ],
    )
