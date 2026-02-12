"""Partition, clustering, and materialized view optimization engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from google.cloud import bigquery

from src.common.models import AccessPattern
from src.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


@dataclass
class PartitionRecommendation:
    column: str | None
    partition_type: str | None  # DAY, MONTH, YEAR
    reason: str = ""


@dataclass
class ClusterRecommendation:
    columns: list[str]
    reason: str = ""


@dataclass
class MVRecommendation:
    create: bool
    query: str = ""
    name: str = ""
    reason: str = ""
    estimated_savings: float = 0.0


class OptimizationEngine:
    """Recommend partitioning, clustering, and materialized views."""

    def __init__(
        self,
        project_id: str,
        location: str = "US",
        gemini_client: GeminiClient | None = None,
        max_clustering_columns: int = 4,
        mv_staleness_hours: int = 24,
        mv_cost_threshold: float = 0.3,
    ) -> None:
        self._bq = bigquery.Client(project=project_id, location=location)
        self._project_id = project_id
        self._gemini = gemini_client
        self._max_cluster = max_clustering_columns
        self._mv_staleness = mv_staleness_hours
        self._mv_threshold = mv_cost_threshold

    # ==================================================================
    # Partition Advisor
    # ==================================================================

    def recommend_partition(
        self,
        table_schema: list[dict[str, Any]],
        patterns: list[AccessPattern],
    ) -> PartitionRecommendation:
        """Evaluate candidates and return a partition recommendation."""
        # Prefer date/timestamp columns that appear frequently in filters.
        date_types = {"DATE", "TIMESTAMP", "DATETIME"}
        candidates: list[tuple[str, float]] = []

        schema_map = {col["name"]: col.get("type", "STRING") for col in table_schema}

        for p in patterns:
            col_type = schema_map.get(p.column_name, "STRING")
            if col_type in date_types and p.filter_frequency > 0:
                candidates.append((p.column_name, p.filter_frequency * 5 + p.cost_impact_bytes / 1e9))

        if not candidates:
            # Fall back: any date/timestamp column.
            for col in table_schema:
                if col.get("type", "STRING") in date_types:
                    candidates.append((col["name"], 1.0))

        if not candidates:
            return PartitionRecommendation(
                column=None,
                partition_type=None,
                reason="No suitable partition column found",
            )

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_col = candidates[0][0]

        # Choose granularity based on expected data volume.
        partition_type = "DAY"  # default for most cases

        return PartitionRecommendation(
            column=best_col,
            partition_type=partition_type,
            reason=f"Column '{best_col}' has highest filter frequency among date/timestamp columns",
        )

    # ==================================================================
    # Cluster Advisor
    # ==================================================================

    def recommend_clustering(
        self,
        patterns: list[AccessPattern],
    ) -> ClusterRecommendation:
        """Rank columns for clustering (up to 4)."""
        # Score: filter frequency (high), medium cardinality preferred.
        scored: list[tuple[str, float]] = []

        for p in patterns:
            cardinality_factor = 1.0
            if p.cardinality is not None:
                # Prefer medium cardinality (100 – 100,000).
                if 100 <= p.cardinality <= 100_000:
                    cardinality_factor = 2.0
                elif p.cardinality > 100_000:
                    cardinality_factor = 1.0
                else:
                    cardinality_factor = 0.5

            score = (
                p.filter_frequency * 3.0
                + p.group_by_frequency * 1.5
                + p.join_frequency * 1.0
            ) * cardinality_factor
            if score > 0:
                scored.append((p.column_name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        columns = [c for c, _ in scored[: self._max_cluster]]

        return ClusterRecommendation(
            columns=columns,
            reason=f"Top {len(columns)} columns by filter/group-by frequency × cardinality",
        )

    # ==================================================================
    # Materialized View Manager
    # ==================================================================

    def recommend_materialized_views(
        self,
        dataset: str,
        table_name: str,
        patterns: list[AccessPattern],
    ) -> list[MVRecommendation]:
        """Identify frequent aggregation patterns worth materializing."""
        recommendations: list[MVRecommendation] = []

        # Group patterns by group_by columns — frequent groupings are MV candidates.
        groupby_cols = [p for p in patterns if p.group_by_frequency > 2]
        if not groupby_cols:
            return recommendations

        # Build a candidate MV for the top group-by combination.
        top_cols = [p.column_name for p in sorted(groupby_cols, key=lambda p: p.group_by_frequency, reverse=True)[:3]]
        if not top_cols:
            return recommendations

        group_expr = ", ".join(top_cols)
        mv_name = f"mv_{table_name}_{'_'.join(top_cols[:2])}"
        mv_query = (
            f"SELECT {group_expr}, COUNT(*) AS cnt, "
            f"SUM(1) AS total "  # placeholder — real metrics depend on schema
            f"FROM `{self._project_id}.{dataset}.{table_name}` "
            f"GROUP BY {group_expr}"
        )

        # Cost-benefit: rough heuristic based on frequency.
        total_freq = sum(p.group_by_frequency for p in groupby_cols)
        estimated_savings = min(total_freq * 0.05, 1.0)  # cap at 100% savings

        if estimated_savings >= self._mv_threshold:
            recommendations.append(MVRecommendation(
                create=True,
                query=mv_query,
                name=mv_name,
                reason=f"Aggregation on ({group_expr}) appears {total_freq} times",
                estimated_savings=estimated_savings,
            ))

        return recommendations

    def create_materialized_view(self, dataset: str, mv: MVRecommendation) -> None:
        """Execute DDL to create a materialized view."""
        fqn = f"`{self._project_id}.{dataset}.{mv.name}`"
        ddl = (
            f"CREATE MATERIALIZED VIEW IF NOT EXISTS {fqn}\n"
            f"OPTIONS (enable_refresh = true, "
            f"refresh_interval_minutes = {self._mv_staleness * 60})\n"
            f"AS\n{mv.query}"
        )
        self._bq.query(ddl).result()
        logger.info("Created materialized view %s", fqn)

    def drop_stale_views(self, dataset: str, usage_threshold: int = 1) -> int:
        """Drop MVs that haven't been queried recently.  Returns count dropped."""
        # List MVs in dataset.
        tables = self._bq.list_tables(f"{self._project_id}.{dataset}")
        dropped = 0
        for t in tables:
            if t.table_type == "MATERIALIZED_VIEW":
                # Check if any recent query references this MV.
                sql = (
                    f"SELECT COUNT(*) AS cnt "
                    f"FROM `{self._project_id}.region-{self._bq.location}`.INFORMATION_SCHEMA.JOBS "
                    f"WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {self._mv_staleness} HOUR) "
                    f"AND CONTAINS_SUBSTR(query, '{t.table_id}')"
                )
                rows = list(self._bq.query(sql).result())
                cnt = list(rows[0].values())[0] if rows else 0
                if cnt < usage_threshold:
                    fqn = f"`{self._project_id}.{dataset}.{t.table_id}`"
                    self._bq.query(f"DROP MATERIALIZED VIEW IF EXISTS {fqn}").result()
                    dropped += 1
                    logger.info("Dropped stale MV %s (usage=%d)", fqn, cnt)
        return dropped

    # ==================================================================
    # Gemini-assisted optimization
    # ==================================================================

    def consult_gemini(
        self,
        table_schema: list[dict[str, Any]],
        patterns: list[AccessPattern],
    ) -> dict[str, Any]:
        """Ask Gemini for complex optimization decisions."""
        if self._gemini is None:
            return {}
        return self._gemini.recommend_partitioning(
            table_schema=table_schema,
            query_patterns=[p.model_dump() for p in patterns],
        )
