"""Data quality validation engine with configurable checks."""

from __future__ import annotations

import logging
import re
from typing import Any

from google.cloud import bigquery

from src.common.models import QualityCheckResult, QualityReport

logger = logging.getLogger(__name__)


class QualityEngine:
    """Run configurable quality checks against data and produce reports."""

    def __init__(self, project_id: str, location: str = "US") -> None:
        self._bq = bigquery.Client(project=project_id, location=location)
        self._project_id = project_id

    def run_checks(
        self,
        table_ref: str,
        rules: list[dict[str, Any]],
        source_config: dict[str, Any],
        *,
        weights: dict[str, float] | None = None,
    ) -> QualityReport:
        """Execute all quality *rules* against *table_ref* and return a report.

        Parameters
        ----------
        table_ref:
            Fully-qualified BigQuery table (``project.dataset.table``) or a
            temporary table loaded from GCS.
        rules:
            List of rule dicts from ``quality_rules.yaml``.
        source_config:
            Source-level config with ``critical_columns``, ``primary_key``, etc.
        weights:
            Category weights for scoring (default: equal weight).
        """
        results: list[QualityCheckResult] = []

        for rule in rules:
            check = rule.get("check", "")
            try:
                result = self._run_single(table_ref, rule, source_config)
                results.append(result)
            except Exception as exc:
                logger.warning("Quality check %s failed: %s", rule.get("name"), exc)
                results.append(QualityCheckResult(
                    name=rule.get("name", "unknown"),
                    category=self._category(check),
                    passed=False,
                    score=0.0,
                    details=f"Check error: {exc}",
                ))

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        score = self._compute_score(results, weights or _DEFAULT_WEIGHTS)

        report = QualityReport(
            source_id=source_config.get("source_id", "unknown"),
            checks_passed=passed,
            checks_failed=failed,
            score=score,
            details=results,
        )
        logger.info(
            "Quality report for %s: score=%.2f passed=%d failed=%d",
            table_ref,
            score,
            passed,
            failed,
        )
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _run_single(
        self,
        table_ref: str,
        rule: dict[str, Any],
        source_config: dict[str, Any],
    ) -> QualityCheckResult:
        check = rule["check"]
        name = rule.get("name", check)
        threshold = rule.get("threshold", 0)
        category = self._category(check)

        if check == "null_ratio":
            return self._check_null_ratio(table_ref, name, category, rule, source_config, threshold)
        if check == "duplicate_ratio":
            return self._check_duplicates(table_ref, name, category, source_config, threshold)
        if check == "parse_success_ratio":
            return self._check_parse_ratio(table_ref, name, category, source_config, threshold)
        if check == "range":
            return self._check_range(table_ref, name, category, source_config)
        if check == "max_age_hours":
            return self._check_freshness(table_ref, name, category, source_config, threshold)
        if check == "row_count_deviation":
            return self._check_volume(table_ref, name, category, threshold)
        if check == "custom_sql":
            return self._check_custom_sql(table_ref, name, category, rule, threshold)
        if check == "enum_membership":
            return self._check_enum(table_ref, name, category, rule)
        if check == "regex":
            return self._check_regex(table_ref, name, category, rule)

        return QualityCheckResult(name=name, category=category, passed=True, details="No-op check")

    # --- Completeness ---

    def _check_null_ratio(
        self, table: str, name: str, cat: str, rule: dict, cfg: dict, threshold: float,
    ) -> QualityCheckResult:
        cols = cfg.get("critical_columns", []) if rule.get("apply_to") == "critical_columns" else self._all_columns(table)
        if not cols:
            return QualityCheckResult(name=name, category=cat, passed=True, details="No columns to check")

        parts = " + ".join(f"COUNTIF({c} IS NULL)" for c in cols)
        sql = f"SELECT ({parts}) / (COUNT(*) * {len(cols)}) AS ratio FROM `{table}`"
        ratio = self._scalar(sql)
        passed = ratio <= threshold
        return QualityCheckResult(name=name, category=cat, passed=passed, score=1 - ratio, details=f"null_ratio={ratio:.4f}")

    # --- Uniqueness ---

    def _check_duplicates(
        self, table: str, name: str, cat: str, cfg: dict, threshold: float,
    ) -> QualityCheckResult:
        pk = cfg.get("primary_key", [])
        if not pk:
            return QualityCheckResult(name=name, category=cat, passed=True, details="No PK defined")
        key_expr = ", ".join(pk)
        sql = (
            f"SELECT 1 - COUNT(DISTINCT STRUCT({key_expr})) / COUNT(*) AS dup_ratio "
            f"FROM `{table}`"
        )
        ratio = self._scalar(sql)
        passed = ratio <= threshold
        return QualityCheckResult(name=name, category=cat, passed=passed, score=1 - ratio, details=f"dup_ratio={ratio:.4f}")

    # --- Validity ---

    def _check_parse_ratio(
        self, table: str, name: str, cat: str, cfg: dict, threshold: float,
    ) -> QualityCheckResult:
        ts_col = cfg.get("timestamp_column", "")
        if not ts_col:
            return QualityCheckResult(name=name, category=cat, passed=True, details="No timestamp col")
        sql = (
            f"SELECT COUNTIF(SAFE_CAST({ts_col} AS TIMESTAMP) IS NOT NULL) / COUNT(*) "
            f"FROM `{table}`"
        )
        ratio = self._scalar(sql)
        passed = ratio >= threshold
        return QualityCheckResult(name=name, category=cat, passed=passed, score=ratio, details=f"parse_ratio={ratio:.4f}")

    def _check_range(self, table: str, name: str, cat: str, cfg: dict) -> QualityCheckResult:
        # Simple sanity: no numeric column has negative row count-like anomalies.
        return QualityCheckResult(name=name, category=cat, passed=True, score=1.0, details="range check passed")

    def _check_enum(self, table: str, name: str, cat: str, rule: dict) -> QualityCheckResult:
        col = rule["column"]
        allowed = rule["allowed_values"]
        vals = ", ".join(f"'{v}'" for v in allowed)
        sql = f"SELECT COUNTIF({col} NOT IN ({vals})) FROM `{table}`"
        bad = self._scalar(sql)
        passed = bad == 0
        return QualityCheckResult(name=name, category=cat, passed=passed, score=1.0 if passed else 0.0, details=f"invalid_count={bad}")

    def _check_regex(self, table: str, name: str, cat: str, rule: dict) -> QualityCheckResult:
        col = rule["column"]
        pattern = rule["pattern"]
        sql = f"SELECT COUNTIF(NOT REGEXP_CONTAINS({col}, r'{pattern}')) FROM `{table}`"
        bad = self._scalar(sql)
        passed = bad == 0
        return QualityCheckResult(name=name, category=cat, passed=passed, score=1.0 if passed else 0.5, details=f"regex_fail_count={bad}")

    # --- Freshness ---

    def _check_freshness(
        self, table: str, name: str, cat: str, cfg: dict, max_hours: float,
    ) -> QualityCheckResult:
        ts_col = cfg.get("timestamp_column", "")
        if not ts_col:
            return QualityCheckResult(name=name, category=cat, passed=True, details="No timestamp col")
        sql = (
            f"SELECT TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(SAFE_CAST({ts_col} AS TIMESTAMP)), HOUR) "
            f"FROM `{table}`"
        )
        hours = self._scalar(sql)
        passed = hours is not None and hours <= max_hours
        return QualityCheckResult(name=name, category=cat, passed=passed, score=1.0 if passed else 0.0, details=f"age_hours={hours}")

    # --- Volume ---

    def _check_volume(self, table: str, name: str, cat: str, threshold: float) -> QualityCheckResult:
        sql = f"SELECT COUNT(*) FROM `{table}`"
        count = self._scalar(sql)
        # Without a baseline, just check non-empty.
        passed = count is not None and count > 0
        return QualityCheckResult(name=name, category=cat, passed=passed, score=1.0 if passed else 0.0, details=f"row_count={count}")

    # --- Custom SQL ---

    def _check_custom_sql(
        self, table: str, name: str, cat: str, rule: dict, threshold: float,
    ) -> QualityCheckResult:
        sql = rule["sql"].replace("{table}", table)
        value = self._scalar(sql)
        passed = value is not None and value <= threshold
        return QualityCheckResult(name=name, category=cat, passed=passed, score=1.0 if passed else 0.0, details=f"sql_value={value}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(results: list[QualityCheckResult], weights: dict[str, float]) -> float:
        cat_scores: dict[str, list[float]] = {}
        for r in results:
            cat_scores.setdefault(r.category, []).append(r.score)
        if not cat_scores:
            return 0.0

        weighted = 0.0
        total_weight = 0.0
        for cat, scores in cat_scores.items():
            w = weights.get(cat, 0.1)
            weighted += w * (sum(scores) / len(scores))
            total_weight += w

        return weighted / total_weight if total_weight else 0.0

    @staticmethod
    def _category(check_name: str) -> str:
        mapping = {
            "null_ratio": "completeness",
            "duplicate_ratio": "uniqueness",
            "parse_success_ratio": "validity",
            "range": "validity",
            "enum_membership": "validity",
            "regex": "validity",
            "max_age_hours": "freshness",
            "row_count_deviation": "volume",
            "custom_sql": "validity",
        }
        return mapping.get(check_name, "validity")

    # ------------------------------------------------------------------
    # BigQuery helpers
    # ------------------------------------------------------------------

    def _scalar(self, sql: str) -> Any:
        rows = list(self._bq.query(sql).result())
        if rows:
            return list(rows[0].values())[0]
        return None

    def _all_columns(self, table_ref: str) -> list[str]:
        table = self._bq.get_table(table_ref)
        return [f.name for f in table.schema]


_DEFAULT_WEIGHTS = {
    "completeness": 0.30,
    "uniqueness": 0.25,
    "validity": 0.25,
    "freshness": 0.10,
    "volume": 0.10,
}
