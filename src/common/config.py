"""YAML + environment variable configuration loader using Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-models (plain Pydantic — not settings)
# ---------------------------------------------------------------------------

class GCPConfig(BaseModel):
    project_id: str
    region: str = "us-central1"
    zone: str = "us-central1-a"


class StorageConfig(BaseModel):
    bronze_bucket: str
    silver_bucket: str
    dlq_bucket: str
    temp_bucket: str


class PubSubConfig(BaseModel):
    ingestion_topic: str
    transformation_topic: str
    aggregation_topic: str
    ingestion_subscription: str
    transformation_subscription: str
    aggregation_subscription: str


class BigQueryConfig(BaseModel):
    bronze_dataset: str = "bronze"
    silver_dataset: str = "silver"
    gold_dataset: str = "gold"
    location: str = "US"


class FirestoreConfig(BaseModel):
    state_collection: str = "agent-states"
    schema_collection: str = "schema-versions"
    checkpoint_ttl_days: int = 30


class IngestionAgentConfig(BaseModel):
    schema_sample_rows: int = 1000
    schema_evolution_strategy: str = "auto"
    max_file_size_mb: int = 500
    supported_formats: list[str] = Field(default_factory=lambda: ["csv", "json", "parquet", "avro"])


class TransformationAgentConfig(BaseModel):
    quality_threshold: float = 0.85
    max_dataflow_workers: int = 10
    code_gen_strategy: str = "hybrid"
    biglake_connection: str = ""


class AggregationAgentConfig(BaseModel):
    query_history_days: int = 30
    max_clustering_columns: int = 4
    mv_staleness_hours: int = 24
    mv_cost_benefit_threshold: float = 0.3


class AgentsConfig(BaseModel):
    ingestion: IngestionAgentConfig = Field(default_factory=IngestionAgentConfig)
    transformation: TransformationAgentConfig = Field(default_factory=TransformationAgentConfig)
    aggregation: AggregationAgentConfig = Field(default_factory=AggregationAgentConfig)


class LLMConfig(BaseModel):
    model: str = "gemini-2.0-flash"
    fallback_model: str = "gemini-1.5-pro"
    max_output_tokens: int = 8192
    temperature: float = 0.1
    retry_attempts: int = 3
    retry_backoff_seconds: int = 2


class HealthCheckConfig(BaseModel):
    port: int = 8080
    path: str = "/health"


class SourceQuality(BaseModel):
    min_score: float = 0.85
    critical_columns: list[str] = Field(default_factory=list)
    max_null_ratio: float = 0.10


class SourceSchemaHints(BaseModel):
    primary_key: list[str] = Field(default_factory=list)
    timestamp_column: str = ""
    partition_column: str = ""
    nested_fields: bool = False
    cdc_type_column: str = ""


class SourceDefinition(BaseModel):
    id: str
    name: str
    type: str  # batch / stream / cdc
    format: str
    location: str
    schedule: str = ""
    schema_hints: SourceSchemaHints = Field(default_factory=SourceSchemaHints)
    quality: SourceQuality = Field(default_factory=SourceQuality)


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


class PipelineConfig(BaseSettings):
    """Root configuration — merges YAML file with env-var overrides."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_nested_delimiter="__",
    )

    gcp: GCPConfig
    storage: StorageConfig
    pubsub: PubSubConfig
    bigquery: BigQueryConfig = Field(default_factory=BigQueryConfig)
    firestore: FirestoreConfig = Field(default_factory=FirestoreConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)

    # Populated separately
    sources: list[SourceDefinition] = Field(default_factory=list)

    @field_validator("gcp")
    @classmethod
    def _validate_project_id(cls, v: GCPConfig) -> GCPConfig:
        if not v.project_id:
            raise ValueError("gcp.project_id must be set")
        return v


def load_config(
    config_dir: str | Path | None = None,
) -> PipelineConfig:
    """Load pipeline_config.yaml + sources.yaml from *config_dir*."""
    if config_dir is None:
        config_dir = Path(os.environ.get("CONFIG_DIR", "config"))
    config_dir = Path(config_dir)

    pipeline_data = _load_yaml(config_dir / "pipeline_config.yaml")
    sources_data = _load_yaml(config_dir / "sources.yaml")

    pipeline_data["sources"] = sources_data.get("sources", [])
    return PipelineConfig(**pipeline_data)
