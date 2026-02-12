"""Unit tests for the Ingestion Agent's core logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.ingestion.schema_detector import SchemaDetector, _parse_gcs, _wider
from src.common.models import (
    ColumnDef,
    EventType,
    EvolutionType,
    Layer,
    PipelineEvent,
    SchemaEvolution,
    SchemaVersion,
)
from src.common.schema_registry import SchemaRegistry


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

class TestSchemaDetector:
    def test_infer_csv_schema(self):
        csv_bytes = b"id,name,amount\n1,Alice,100.5\n2,Bob,200\n"

        detector = SchemaDetector.__new__(SchemaDetector)
        detector._sample_rows = 1000
        columns = detector._infer_csv(csv_bytes)

        assert len(columns) == 3
        names = {c.name for c in columns}
        assert names == {"id", "name", "amount"}

    def test_infer_json_schema_array(self):
        json_bytes = b'[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'

        detector = SchemaDetector.__new__(SchemaDetector)
        detector._sample_rows = 1000
        columns = detector._infer_json(json_bytes)

        assert len(columns) == 2
        names = {c.name for c in columns}
        assert names == {"id", "name"}

    def test_infer_json_schema_lines(self):
        json_bytes = b'{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}\n'

        detector = SchemaDetector.__new__(SchemaDetector)
        detector._sample_rows = 1000
        columns = detector._infer_json(json_bytes)

        assert len(columns) == 2

    def test_flatten_nested_json(self):
        nested = {"user": {"name": "Alice", "age": 30}, "score": 95}
        flat = SchemaDetector._flatten(nested)

        assert flat == {"user.name": "Alice", "user.age": 30, "score": 95}

    def test_columns_from_dicts_type_promotion(self):
        rows = [
            {"val": 1},
            {"val": 1.5},
        ]
        columns = SchemaDetector._columns_from_dicts(rows)
        col_map = {c.name: c for c in columns}
        # int should be promoted to float.
        assert col_map["val"].data_type == "FLOAT64"

    def test_columns_with_nulls(self):
        rows = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": None},
        ]
        columns = SchemaDetector._columns_from_dicts(rows)
        col_map = {c.name: c for c in columns}
        assert col_map["name"].nullable is True
        assert col_map["id"].nullable is False


# ---------------------------------------------------------------------------
# Schema evolution detection
# ---------------------------------------------------------------------------

class TestSchemaEvolution:
    def test_identical_schemas(self):
        old = SchemaVersion(
            source_id="s1",
            columns=[ColumnDef(name="id", data_type="INT64")],
        )
        new = SchemaVersion(
            source_id="s1",
            columns=[ColumnDef(name="id", data_type="INT64")],
        )
        evo = SchemaRegistry.detect_evolution(old, new)
        assert evo.evolution_type == EvolutionType.IDENTICAL

    def test_additive_change(self):
        old = SchemaVersion(
            source_id="s1",
            columns=[ColumnDef(name="id", data_type="INT64")],
        )
        new = SchemaVersion(
            source_id="s1",
            columns=[
                ColumnDef(name="id", data_type="INT64"),
                ColumnDef(name="name", data_type="STRING"),
            ],
        )
        evo = SchemaRegistry.detect_evolution(old, new)
        assert evo.evolution_type == EvolutionType.ADDITIVE
        assert len(evo.added_columns) == 1
        assert evo.added_columns[0].name == "name"

    def test_breaking_column_removal(self):
        old = SchemaVersion(
            source_id="s1",
            columns=[
                ColumnDef(name="id", data_type="INT64"),
                ColumnDef(name="name", data_type="STRING"),
            ],
        )
        new = SchemaVersion(
            source_id="s1",
            columns=[ColumnDef(name="id", data_type="INT64")],
        )
        evo = SchemaRegistry.detect_evolution(old, new)
        assert evo.evolution_type == EvolutionType.BREAKING
        assert len(evo.removed_columns) == 1

    def test_compatible_type_widening(self):
        old = SchemaVersion(
            source_id="s1",
            columns=[ColumnDef(name="val", data_type="INT64")],
        )
        new = SchemaVersion(
            source_id="s1",
            columns=[ColumnDef(name="val", data_type="FLOAT64")],
        )
        evo = SchemaRegistry.detect_evolution(old, new)
        assert evo.evolution_type == EvolutionType.COMPATIBLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_parse_gcs(self):
        bucket, blob = _parse_gcs("gs://my-bucket/path/to/file.csv")
        assert bucket == "my-bucket"
        assert blob == "path/to/file.csv"

    def test_wider_types(self):
        assert _wider("INT64", "FLOAT64") == "FLOAT64"
        assert _wider("FLOAT64", "INT64") == "FLOAT64"
        assert _wider("STRING", "INT64") == "STRING"


# ---------------------------------------------------------------------------
# Ingestion agent event processing
# ---------------------------------------------------------------------------

class TestIngestionAgent:
    def test_ignores_non_file_arrived_events(
        self, test_config, mock_event_bus, mock_state_manager,
        mock_lineage_tracker, mock_schema_registry, mock_gemini_client,
    ):
        from src.agents.ingestion.agent import IngestionAgent

        agent = IngestionAgent(
            config=test_config,
            event_bus=mock_event_bus,
            state_manager=mock_state_manager,
            lineage_tracker=mock_lineage_tracker,
            schema_registry=mock_schema_registry,
            gemini_client=mock_gemini_client,
        )

        event = PipelineEvent(
            event_type=EventType.INGESTION_COMPLETE,
            source_id="test_source",
            layer=Layer.BRONZE,
        )
        # Should not raise and should not publish anything.
        agent.process_event(event)
        mock_event_bus.publish.assert_not_called()

    def test_schema_from_payload(self):
        from src.agents.ingestion.agent import IngestionAgent

        schema = IngestionAgent._schema_from_payload(
            "test",
            {"stream_data": {"id": 1, "name": "Alice", "amount": 99.9}},
        )
        assert len(schema.columns) == 3
        assert schema.fingerprint  # fingerprint should be computed


# ---------------------------------------------------------------------------
# Schema fingerprinting
# ---------------------------------------------------------------------------

class TestSchemaFingerprint:
    def test_fingerprint_deterministic(self, sample_schema):
        fp1 = sample_schema.compute_fingerprint()
        fp2 = sample_schema.compute_fingerprint()
        assert fp1 == fp2

    def test_different_schemas_different_fingerprints(self):
        s1 = SchemaVersion(
            source_id="a",
            columns=[ColumnDef(name="id", data_type="INT64")],
        )
        s2 = SchemaVersion(
            source_id="a",
            columns=[ColumnDef(name="id", data_type="STRING")],
        )
        s1.compute_fingerprint()
        s2.compute_fingerprint()
        assert s1.fingerprint != s2.fingerprint
