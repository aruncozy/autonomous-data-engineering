"""Data Catalog Lineage API integration for tracking data flow."""

from __future__ import annotations

import logging
from datetime import datetime

from google.cloud import datacatalog_lineage_v1 as lineage_v1

from src.common.models import LineageRecord

logger = logging.getLogger(__name__)


class LineageTracker:
    """Record and query data lineage via Data Catalog Lineage API."""

    def __init__(self, project_id: str, location: str = "us-central1") -> None:
        self._project_id = project_id
        self._location = location
        self._client = lineage_v1.LineageClient()
        self._parent = f"projects/{project_id}/locations/{location}"

    # ------------------------------------------------------------------
    # Record lineage
    # ------------------------------------------------------------------

    def record(
        self,
        source_ref: str,
        target_ref: str,
        process_name: str,
        run_id: str,
    ) -> LineageRecord:
        """Create a lineage entry linking *source_ref* → *target_ref*.

        Parameters
        ----------
        source_ref:
            Fully-qualified source (e.g. ``gs://bucket/path`` or
            ``bigquery:project.dataset.table``).
        target_ref:
            Fully-qualified target.
        process_name:
            Human-readable process name (e.g. ``ingestion-agent``).
        run_id:
            Unique run / correlation ID.
        """
        # 1. Create (or reuse) a Process.
        process = self._ensure_process(process_name)

        # 2. Create a Run under the process.
        run = lineage_v1.Run(
            display_name=run_id,
            state=lineage_v1.Run.State.STARTED,
            start_time=datetime.utcnow(),
        )
        run = self._client.create_run(parent=process.name, run=run)

        # 3. Create a LineageEvent linking source → target.
        event = lineage_v1.LineageEvent(
            start_time=datetime.utcnow(),
            links=[
                lineage_v1.EventLink(
                    source=lineage_v1.EntityReference(fully_qualified_name=source_ref),
                    target=lineage_v1.EntityReference(fully_qualified_name=target_ref),
                ),
            ],
        )
        self._client.create_lineage_event(parent=run.name, lineage_event=event)

        # 4. Mark run as completed.
        run.state = lineage_v1.Run.State.COMPLETED
        run.end_time = datetime.utcnow()
        self._client.update_run(run=run)

        record = LineageRecord(
            source=source_ref,
            target=target_ref,
            process=process_name,
            run_id=run_id,
        )
        logger.info(
            "Lineage recorded: %s → %s (process=%s, run=%s)",
            source_ref,
            target_ref,
            process_name,
            run_id,
        )
        return record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_process(self, process_name: str) -> lineage_v1.Process:
        """Get an existing process or create a new one."""
        # List processes to find an existing one by display name.
        for proc in self._client.list_processes(parent=self._parent):
            if proc.display_name == process_name:
                return proc

        process = lineage_v1.Process(display_name=process_name)
        return self._client.create_process(parent=self._parent, process=process)
