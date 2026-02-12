"""Autonomous code generation for data transformations (SQL + Beam)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from src.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "templates"


class CodeGenerator:
    """Generate transformation code via Jinja2 templates or Gemini LLM."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        strategy: str = "hybrid",
    ) -> None:
        self._gemini = gemini_client
        self._strategy = strategy  # "template" | "gemini" | "hybrid"
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sql(
        self,
        source_table: str,
        target_table: str,
        schema: dict[str, Any],
        *,
        primary_key: list[str] | None = None,
        timestamp_column: str = "",
        quality_rules: list[dict] | None = None,
        data_sample: list[dict] | None = None,
    ) -> str:
        """Return a BigQuery SQL MERGE statement for cleanse + dedup.

        Uses the template for known patterns; falls back to Gemini for
        novel/complex sources when strategy is ``hybrid`` or ``gemini``.
        """
        columns = schema.get("columns", [])
        col_names = [c["name"] if isinstance(c, dict) else c.name for c in columns]

        if self._strategy != "gemini" and col_names:
            # Template path.
            try:
                return self._render_sql_template(
                    source_table, target_table, col_names, primary_key or [], timestamp_column,
                )
            except Exception:
                if self._strategy == "template":
                    raise
                logger.info("Template failed — falling back to Gemini")

        # Gemini path.
        target_spec = {
            "target_table": target_table,
            "primary_key": primary_key or [],
            "timestamp_column": timestamp_column,
        }
        result = self._gemini.generate_transform_code(
            source_schema=schema,
            target_spec=target_spec,
            data_sample=data_sample or [],
            language="SQL",
        )
        sql = result.get("code", "")
        self._validate_sql(sql)
        return sql

    def generate_beam_pipeline(
        self,
        source_path: str,
        target_table: str,
        schema: dict[str, Any],
        *,
        fmt: str = "csv",
        streaming: bool = False,
        data_sample: list[dict] | None = None,
    ) -> str:
        """Return Apache Beam Python pipeline code.

        Uses the batch or streaming template; falls back to Gemini.
        """
        columns = schema.get("columns", [])
        col_names = [c["name"] if isinstance(c, dict) else c.name for c in columns]

        if self._strategy != "gemini" and col_names:
            try:
                tmpl_name = "beam_streaming.py.j2" if streaming else "beam_batch.py.j2"
                return self._render_beam_template(
                    tmpl_name, source_path, target_table, col_names, fmt,
                )
            except Exception:
                if self._strategy == "template":
                    raise
                logger.info("Template failed — falling back to Gemini for Beam code")

        result = self._gemini.generate_transform_code(
            source_schema=schema,
            target_spec={"target_table": target_table, "format": fmt, "streaming": streaming},
            data_sample=data_sample or [],
            language="Python",
        )
        code = result.get("code", "")
        self._validate_python(code)
        return code

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_sql_template(
        self,
        source_table: str,
        target_table: str,
        columns: list[str],
        primary_key: list[str],
        timestamp_column: str,
    ) -> str:
        tmpl = self._jinja.get_template("sql_transform.sql.j2")
        return tmpl.render(
            source_table=source_table,
            target_table=target_table,
            columns=columns,
            primary_key=primary_key,
            timestamp_column=timestamp_column,
        )

    def _render_beam_template(
        self,
        template_name: str,
        source_path: str,
        target_table: str,
        columns: list[str],
        fmt: str,
    ) -> str:
        tmpl = self._jinja.get_template(template_name)
        return tmpl.render(
            source_path=source_path,
            target_table=target_table,
            columns=columns,
            format=fmt,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_sql(sql: str) -> None:
        """Basic SQL validation — ensures the string is non-empty and
        contains at least one keyword."""
        if not sql or not sql.strip():
            raise ValueError("Generated SQL is empty")
        lower = sql.lower()
        if not any(kw in lower for kw in ("select", "merge", "insert", "create")):
            raise ValueError("Generated SQL lacks recognizable statements")

    @staticmethod
    def _validate_python(code: str) -> None:
        """Parse Python code to verify it is syntactically valid."""
        if not code or not code.strip():
            raise ValueError("Generated Python code is empty")
        ast.parse(code)
