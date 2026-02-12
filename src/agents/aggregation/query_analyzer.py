"""INFORMATION_SCHEMA access pattern analysis for optimization decisions."""

from __future__ import annotations

import logging
import re
from typing import Any

from google.cloud import bigquery

from src.common.models import AccessPattern

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyze historical BigQuery query patterns to guide optimization."""

    def __init__(self, project_id: str, location: str = "US") -> None:
        self._bq = bigquery.Client(project=project_id, location=location)
        self._project_id = project_id
        self._location = location

    def analyze(
        self,
        dataset: str,
        table_name: str,
        *,
        days: int = 30,
    ) -> list[AccessPattern]:
        """Query INFORMATION_SCHEMA.JOBS for access patterns on *table_name*.

        Returns a ranked list of ``AccessPattern`` objects.
        """
        patterns: dict[str, AccessPattern] = {}

        # Pull recent completed query jobs referencing this table.
        jobs = self._get_recent_jobs(dataset, table_name, days)

        for job in jobs:
            query_text = job.get("query", "")
            bytes_scanned = job.get("total_bytes_processed", 0)
            self._extract_patterns(query_text, patterns, bytes_scanned)

        # Score and rank.
        result = list(patterns.values())
        for p in result:
            p.score = (
                p.filter_frequency * 3.0
                + p.group_by_frequency * 2.0
                + p.join_frequency * 1.5
            )
        result.sort(key=lambda p: p.score, reverse=True)

        logger.info(
            "Analyzed %d jobs for %s.%s — top columns: %s",
            len(jobs),
            dataset,
            table_name,
            [p.column_name for p in result[:5]],
        )
        return result

    # ------------------------------------------------------------------
    # Query INFORMATION_SCHEMA
    # ------------------------------------------------------------------

    def _get_recent_jobs(
        self, dataset: str, table_name: str, days: int,
    ) -> list[dict[str, Any]]:
        sql = f"""\
        SELECT
            query,
            total_bytes_processed,
            creation_time
        FROM `{self._project_id}.region-{self._location}`.INFORMATION_SCHEMA.JOBS
        WHERE
            creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
            AND state = 'DONE'
            AND error_result IS NULL
            AND CONTAINS_SUBSTR(query, '{table_name}')
        ORDER BY creation_time DESC
        LIMIT 500
        """
        rows = list(self._bq.query(sql).result())
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Pattern extraction from SQL text
    # ------------------------------------------------------------------

    def _extract_patterns(
        self,
        query: str,
        patterns: dict[str, AccessPattern],
        bytes_scanned: int,
    ) -> None:
        upper = query.upper()

        # WHERE clause columns.
        for col in self._extract_where_columns(query):
            p = patterns.setdefault(col, AccessPattern(column_name=col))
            p.filter_frequency += 1
            p.cost_impact_bytes += bytes_scanned

        # GROUP BY columns.
        for col in self._extract_group_by_columns(query):
            p = patterns.setdefault(col, AccessPattern(column_name=col))
            p.group_by_frequency += 1

        # JOIN columns.
        for col in self._extract_join_columns(query):
            p = patterns.setdefault(col, AccessPattern(column_name=col))
            p.join_frequency += 1

    @staticmethod
    def _extract_where_columns(query: str) -> list[str]:
        """Extract column names referenced in WHERE predicates."""
        pattern = re.compile(
            r"WHERE\s+(.+?)(?:GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING|$)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(query)
        if not match:
            return []
        where_clause = match.group(1)
        # Find identifiers before comparison operators.
        cols = re.findall(r"(\b\w+)\s*(?:=|!=|<|>|<=|>=|LIKE|IN|IS|BETWEEN)", where_clause, re.IGNORECASE)
        return [c.lower() for c in cols if not _is_keyword(c)]

    @staticmethod
    def _extract_group_by_columns(query: str) -> list[str]:
        pattern = re.compile(
            r"GROUP\s+BY\s+(.+?)(?:HAVING|ORDER\s+BY|LIMIT|$)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(query)
        if not match:
            return []
        cols = re.findall(r"\b(\w+)\b", match.group(1))
        return [c.lower() for c in cols if not _is_keyword(c) and not c.isdigit()]

    @staticmethod
    def _extract_join_columns(query: str) -> list[str]:
        pattern = re.compile(r"(?:ON|USING)\s*\(?\s*(.+?)\s*\)?(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$)", re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(query)
        cols: list[str] = []
        for m in matches:
            found = re.findall(r"\b(\w+)\b", m)
            cols.extend(c.lower() for c in found if not _is_keyword(c))
        return cols

    # ------------------------------------------------------------------
    # Cardinality enrichment
    # ------------------------------------------------------------------

    def enrich_cardinality(
        self,
        table_ref: str,
        patterns: list[AccessPattern],
    ) -> list[AccessPattern]:
        """Query APPROX_COUNT_DISTINCT for each pattern column."""
        for p in patterns:
            try:
                sql = f"SELECT APPROX_COUNT_DISTINCT(`{p.column_name}`) FROM `{table_ref}`"
                rows = list(self._bq.query(sql).result())
                if rows:
                    p.cardinality = list(rows[0].values())[0]
            except Exception:
                logger.debug("Could not get cardinality for %s", p.column_name)
        return patterns


_KEYWORDS = frozenset({
    "select", "from", "where", "and", "or", "not", "in", "is", "null",
    "like", "between", "as", "on", "join", "left", "right", "inner",
    "outer", "full", "cross", "group", "by", "order", "having", "limit",
    "offset", "union", "all", "distinct", "case", "when", "then", "else",
    "end", "true", "false", "asc", "desc", "exists", "any", "some",
    "count", "sum", "avg", "min", "max", "cast", "coalesce", "nullif",
    "current_timestamp", "timestamp_sub", "interval", "day", "hour",
})


def _is_keyword(word: str) -> bool:
    return word.lower() in _KEYWORDS
