"""Firestore-backed agent state persistence and checkpoint management."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from google.cloud import firestore

from src.common.models import AgentState

logger = logging.getLogger(__name__)


class StateManager:
    """Persist agent state in Firestore for crash-recovery."""

    def __init__(self, project_id: str, collection: str = "agent-states") -> None:
        self._db = firestore.Client(project=project_id, database="agent-db")
        self._collection = collection

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def save_checkpoint(self, agent_id: str, state: AgentState) -> None:
        """Persist the current *state* for *agent_id*."""
        state.last_checkpoint = datetime.utcnow()
        doc_ref = self._db.collection(self._collection).document(agent_id)
        doc_ref.set(state.model_dump(mode="json"))
        logger.info("Checkpoint saved for %s", agent_id)

    def load_checkpoint(self, agent_id: str) -> AgentState | None:
        """Load the last saved state for *agent_id*, or ``None``."""
        doc = self._db.collection(self._collection).document(agent_id).get()
        if doc.exists:
            logger.info("Checkpoint loaded for %s", agent_id)
            return AgentState.model_validate(doc.to_dict())
        logger.info("No checkpoint found for %s", agent_id)
        return None

    def update_status(self, agent_id: str, status: str, task: str | None = None) -> None:
        """Update status and optionally the current task."""
        doc_ref = self._db.collection(self._collection).document(agent_id)
        update: dict = {
            "status": status,
            "last_checkpoint": datetime.utcnow().isoformat(),
        }
        if task is not None:
            update["current_task"] = task
        doc_ref.update(update)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_checkpoints(self, ttl_days: int = 30) -> int:
        """Delete checkpoints older than *ttl_days*.  Returns count deleted."""
        cutoff = datetime.utcnow() - timedelta(days=ttl_days)
        query = (
            self._db.collection(self._collection)
            .where("last_checkpoint", "<", cutoff.isoformat())
        )
        deleted = 0
        for doc in query.stream():
            doc.reference.delete()
            deleted += 1
        if deleted:
            logger.info("Cleaned up %d stale checkpoints (>%d days)", deleted, ttl_days)
        return deleted
