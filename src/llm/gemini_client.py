"""Vertex AI Gemini integration for autonomous agent decisions."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from vertexai.generative_models import GenerationConfig, GenerativeModel

from src.common.models import SchemaEvolution

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SCHEMA_EVOLUTION_PROMPT = """\
You are a data engineering expert. Analyze the following schema change and
recommend an action.

**Schema Diff:**
{schema_diff}

Respond in JSON with keys:
- "recommendation": one of "auto_evolve", "pause_and_review", "reject"
- "reason": one-sentence explanation
- "migration_steps": list of steps if migration is needed (empty list if not)
"""

_TRANSFORM_CODE_PROMPT = """\
You are an expert data engineer. Generate {language} code to transform data
from the source schema to the target specification.

**Source Schema:**
{source_schema}

**Target Specification:**
{target_spec}

**Sample Data (first 5 rows):**
{data_sample}

**Requirements:**
- Handle nulls gracefully
- Cast types appropriately
- Deduplicate on primary key using ROW_NUMBER
- Standardize date formats to ISO 8601
- Standardize currency to 2 decimal places

Respond in JSON with keys:
- "code": the complete {language} code as a string
- "explanation": one-paragraph explanation
"""

_PARTITION_PROMPT = """\
You are a BigQuery performance expert. Recommend partitioning and clustering
for the following table.

**Table Schema:**
{table_schema}

**Query Access Patterns (last 30 days):**
{query_patterns}

Respond in JSON with keys:
- "partition_column": column name (or null)
- "partition_type": "DAY", "MONTH", "YEAR", or null
- "clustering_columns": list of up to 4 column names (or empty list)
- "reason": one-paragraph explanation
"""

_QUALITY_RULES_PROMPT = """\
You are a data quality expert. Suggest quality rules for the following schema.

**Schema:**
{schema}

**Sample Data (first 5 rows):**
{sample_data}

Respond in JSON with keys:
- "rules": list of objects with "name", "check_type", "column", "threshold", "description"
"""


class GeminiClient:
    """Wrapper around Vertex AI Gemini for agent decision-making."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        fallback_model: str = "gemini-1.5-pro",
        max_output_tokens: int = 8192,
        temperature: float = 0.1,
        retry_attempts: int = 3,
        retry_backoff: int = 2,
    ) -> None:
        self._model_name = model_name
        self._fallback_model = fallback_model
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._retry_attempts = retry_attempts
        self._retry_backoff = retry_backoff

        self._generation_config = GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            response_mime_type="application/json",
        )
        self._model = GenerativeModel(model_name)
        self._fallback = GenerativeModel(fallback_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_schema(self, schema_diff: SchemaEvolution) -> dict[str, Any]:
        """Return a recommendation for handling a schema evolution."""
        prompt = _SCHEMA_EVOLUTION_PROMPT.format(
            schema_diff=schema_diff.model_dump_json(indent=2),
        )
        return self._call(prompt)

    def generate_transform_code(
        self,
        source_schema: dict[str, Any],
        target_spec: dict[str, Any],
        data_sample: list[dict[str, Any]],
        *,
        language: str = "SQL",
    ) -> dict[str, Any]:
        """Generate SQL or Beam Python transform code."""
        prompt = _TRANSFORM_CODE_PROMPT.format(
            language=language,
            source_schema=json.dumps(source_schema, indent=2),
            target_spec=json.dumps(target_spec, indent=2),
            data_sample=json.dumps(data_sample[:5], indent=2, default=str),
        )
        return self._call(prompt)

    def recommend_partitioning(
        self,
        table_schema: dict[str, Any],
        query_patterns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return partition/cluster strategy recommendation."""
        prompt = _PARTITION_PROMPT.format(
            table_schema=json.dumps(table_schema, indent=2),
            query_patterns=json.dumps(query_patterns, indent=2),
        )
        return self._call(prompt)

    def assess_quality_rules(
        self,
        schema: dict[str, Any],
        sample_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Suggest quality rules for a dataset."""
        prompt = _QUALITY_RULES_PROMPT.format(
            schema=json.dumps(schema, indent=2),
            sample_data=json.dumps(sample_data[:5], indent=2, default=str),
        )
        return self._call(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> dict[str, Any]:
        """Send prompt to Gemini with retry + fallback logic."""
        last_error: Exception | None = None

        for model in (self._model, self._fallback):
            for attempt in range(1, self._retry_attempts + 1):
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config=self._generation_config,
                    )
                    text = response.text.strip()
                    return json.loads(text)
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Gemini call failed (model=%s, attempt=%d/%d): %s",
                        model._model_name,
                        attempt,
                        self._retry_attempts,
                        exc,
                    )
                    if attempt < self._retry_attempts:
                        time.sleep(self._retry_backoff * attempt)

        raise RuntimeError(f"All Gemini attempts exhausted: {last_error}") from last_error
