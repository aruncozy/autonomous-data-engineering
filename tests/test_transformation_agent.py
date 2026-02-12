"""Unit tests for the Transformation Agent's core logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.common.models import (
    ColumnDef,
    EventType,
    Layer,
    PipelineEvent,
    QualityCheckResult,
    QualityReport,
    SchemaVersion,
)


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

class TestCodeGenerator:
    def test_sql_template_rendering(self, mock_gemini_client):
        from src.agents.transformation.code_generator import CodeGenerator

        gen = CodeGenerator(mock_gemini_client, strategy="template")

        sql = gen.generate_sql(
            source_table="project.bronze._temp_test",
            target_table="project.silver.silver_test_main",
            schema={"columns": [
                {"name": "id"},
                {"name": "name"},
                {"name": "created_at"},
            ]},
            primary_key=["id"],
            timestamp_column="created_at",
        )

        assert "MERGE" in sql
        assert "silver_test_main" in sql
        assert "ROW_NUMBER" in sql
        assert "id" in sql

    def test_sql_template_no_pk(self, mock_gemini_client):
        from src.agents.transformation.code_generator import CodeGenerator

        gen = CodeGenerator(mock_gemini_client, strategy="template")

        sql = gen.generate_sql(
            source_table="project.bronze._temp_test",
            target_table="project.silver.silver_test_main",
            schema={"columns": [{"name": "col_a"}, {"name": "col_b"}]},
            primary_key=[],
        )

        # Without PK, should do a CREATE OR REPLACE.
        assert "CREATE OR REPLACE" in sql

    def test_sql_validation_rejects_empty(self, mock_gemini_client):
        from src.agents.transformation.code_generator import CodeGenerator

        gen = CodeGenerator(mock_gemini_client, strategy="template")
        with pytest.raises(ValueError, match="empty"):
            gen._validate_sql("")

    def test_python_validation_rejects_syntax_error(self, mock_gemini_client):
        from src.agents.transformation.code_generator import CodeGenerator

        gen = CodeGenerator(mock_gemini_client, strategy="template")
        with pytest.raises(SyntaxError):
            gen._validate_python("def broken(")

    def test_beam_template_rendering(self, mock_gemini_client):
        from src.agents.transformation.code_generator import CodeGenerator

        gen = CodeGenerator(mock_gemini_client, strategy="template")

        code = gen.generate_beam_pipeline(
            source_path="gs://test-bronze/test/data.csv",
            target_table="project.silver.silver_test_main",
            schema={"columns": [{"name": "id"}, {"name": "name"}]},
            fmt="csv",
        )

        assert "ReadCSV" in code
        assert "WriteToBQ" in code


# ---------------------------------------------------------------------------
# Quality engine (basic scoring tests)
# ---------------------------------------------------------------------------

class TestQualityScoring:
    def test_perfect_score(self):
        from src.agents.transformation.quality_engine import QualityEngine

        results = [
            QualityCheckResult(name="c1", category="completeness", passed=True, score=1.0),
            QualityCheckResult(name="u1", category="uniqueness", passed=True, score=1.0),
            QualityCheckResult(name="v1", category="validity", passed=True, score=1.0),
        ]
        weights = {"completeness": 0.4, "uniqueness": 0.3, "validity": 0.3}
        score = QualityEngine._compute_score(results, weights)
        assert score == pytest.approx(1.0)

    def test_partial_score(self):
        from src.agents.transformation.quality_engine import QualityEngine

        results = [
            QualityCheckResult(name="c1", category="completeness", passed=True, score=1.0),
            QualityCheckResult(name="u1", category="uniqueness", passed=False, score=0.0),
        ]
        weights = {"completeness": 0.5, "uniqueness": 0.5}
        score = QualityEngine._compute_score(results, weights)
        assert score == pytest.approx(0.5)

    def test_empty_results(self):
        from src.agents.transformation.quality_engine import QualityEngine

        score = QualityEngine._compute_score([], {})
        assert score == 0.0


# ---------------------------------------------------------------------------
# Transformation agent event processing
# ---------------------------------------------------------------------------

class TestTransformationAgent:
    def test_ignores_non_ingestion_events(
        self, test_config, mock_event_bus, mock_state_manager,
        mock_lineage_tracker, mock_schema_registry, mock_gemini_client,
    ):
        from src.agents.transformation.agent import TransformationAgent

        agent = TransformationAgent(
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


# ---------------------------------------------------------------------------
# BiglakeManager table naming
# ---------------------------------------------------------------------------

class TestBigLakeManager:
    def test_format_mapping(self):
        from src.agents.transformation.biglake_manager import BigLakeManager
        from google.cloud.bigquery import SourceFormat

        assert BigLakeManager._format_to_source("parquet") == SourceFormat.PARQUET
        assert BigLakeManager._format_to_source("csv") == SourceFormat.CSV
        assert BigLakeManager._format_to_source("json") == SourceFormat.NEWLINE_DELIMITED_JSON
        assert BigLakeManager._format_to_source("avro") == SourceFormat.AVRO
