"""Shared Pydantic data models for the agent pipeline."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Layer(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class EventType(str, Enum):
    FILE_ARRIVED = "FILE_ARRIVED"
    INGESTION_COMPLETE = "INGESTION_COMPLETE"
    TRANSFORMATION_COMPLETE = "TRANSFORMATION_COMPLETE"
    AGGREGATION_COMPLETE = "AGGREGATION_COMPLETE"
    QUALITY_FAILURE = "QUALITY_FAILURE"
    SCHEMA_EVOLUTION = "SCHEMA_EVOLUTION"
    ERROR = "ERROR"


class SourceType(str, Enum):
    BATCH = "batch"
    STREAM = "stream"
    CDC = "cdc"


class SchemaStrategy(str, Enum):
    AUTO = "auto"
    PAUSE = "pause"
    ALERT = "alert"


class EvolutionType(str, Enum):
    ADDITIVE = "ADDITIVE"           # new columns
    COMPATIBLE = "COMPATIBLE"       # type widening
    BREAKING = "BREAKING"           # col removal / type narrowing
    IDENTICAL = "IDENTICAL"


class CDCOperation(str, Enum):
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class PipelineEvent(BaseModel):
    event_type: EventType
    source_id: str
    layer: Layer
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ColumnDef(BaseModel):
    name: str
    data_type: str
    nullable: bool = True
    description: str = ""


class SchemaVersion(BaseModel):
    source_id: str
    version: int = 1
    columns: list[ColumnDef] = Field(default_factory=list)
    fingerprint: str = ""
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    def compute_fingerprint(self) -> str:
        raw = "|".join(
            f"{c.name}:{c.data_type}:{c.nullable}" for c in sorted(self.columns, key=lambda c: c.name)
        )
        self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.fingerprint


class SchemaEvolution(BaseModel):
    evolution_type: EvolutionType
    added_columns: list[ColumnDef] = Field(default_factory=list)
    removed_columns: list[ColumnDef] = Field(default_factory=list)
    type_changes: list[dict[str, Any]] = Field(default_factory=list)
    nullability_changes: list[dict[str, Any]] = Field(default_factory=list)


class QualityCheckResult(BaseModel):
    name: str
    category: str
    passed: bool
    score: float = 1.0
    details: str = ""


class QualityReport(BaseModel):
    source_id: str
    checks_passed: int = 0
    checks_failed: int = 0
    score: float = 0.0
    details: list[QualityCheckResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LineageRecord(BaseModel):
    source: str
    target: str
    process: str
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    agent_id: str
    status: str = "idle"
    current_task: str | None = None
    last_checkpoint: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AccessPattern(BaseModel):
    column_name: str
    filter_frequency: int = 0
    group_by_frequency: int = 0
    join_frequency: int = 0
    cardinality: int | None = None
    cost_impact_bytes: int = 0
    score: float = 0.0


class IngestedRecord(BaseModel):
    source_id: str
    gcs_path: str
    format: str
    row_count: int = 0
    schema_version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)
