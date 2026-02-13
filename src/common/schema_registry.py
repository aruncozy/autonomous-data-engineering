"""Firestore-backed schema versioning and evolution detection."""

from __future__ import annotations

import logging
from datetime import datetime

from google.cloud import firestore

from src.common.models import ColumnDef, EvolutionType, SchemaEvolution, SchemaVersion

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """Store, version, and compare schemas in Firestore."""

    def __init__(self, project_id: str, collection: str = "schema-versions") -> None:
        self._db = firestore.Client(project=project_id, database="agent-db")
        self._collection = collection

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, source_id: str, schema: SchemaVersion) -> SchemaEvolution | None:
        """Register a new schema version.

        Returns a ``SchemaEvolution`` describing the diff against the
        previous version, or ``None`` if this is the first registration.
        """
        schema.compute_fingerprint()
        current = self.get_current(source_id)

        if current is not None and current.fingerprint == schema.fingerprint:
            logger.debug("Schema unchanged for %s (fp=%s)", source_id, schema.fingerprint)
            return SchemaEvolution(evolution_type=EvolutionType.IDENTICAL)

        if current is not None:
            schema.version = current.version + 1
            evolution = self.detect_evolution(current, schema)
        else:
            schema.version = 1
            evolution = None

        schema.detected_at = datetime.utcnow()

        doc_ref = (
            self._db.collection(self._collection)
            .document(source_id)
            .collection("versions")
            .document(str(schema.version))
        )
        doc_ref.set(schema.model_dump(mode="json"))

        # Also update a "latest" pointer for fast lookups.
        self._db.collection(self._collection).document(source_id).set(
            schema.model_dump(mode="json"),
        )

        logger.info(
            "Registered schema v%d for %s (fp=%s)",
            schema.version,
            source_id,
            schema.fingerprint,
        )
        return evolution

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_current(self, source_id: str) -> SchemaVersion | None:
        """Return the latest schema for *source_id*, or ``None``."""
        doc = self._db.collection(self._collection).document(source_id).get()
        if doc.exists:
            return SchemaVersion.model_validate(doc.to_dict())
        return None

    # ------------------------------------------------------------------
    # Evolution detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_evolution(old: SchemaVersion, new: SchemaVersion) -> SchemaEvolution:
        """Classify changes between *old* and *new* schemas."""
        old_cols = {c.name: c for c in old.columns}
        new_cols = {c.name: c for c in new.columns}

        added = [c for name, c in new_cols.items() if name not in old_cols]
        removed = [c for name, c in old_cols.items() if name not in new_cols]

        type_changes: list[dict] = []
        nullability_changes: list[dict] = []
        for name in old_cols.keys() & new_cols.keys():
            oc, nc = old_cols[name], new_cols[name]
            if oc.data_type != nc.data_type:
                type_changes.append({
                    "column": name,
                    "old_type": oc.data_type,
                    "new_type": nc.data_type,
                })
            if oc.nullable != nc.nullable:
                nullability_changes.append({
                    "column": name,
                    "old_nullable": oc.nullable,
                    "new_nullable": nc.nullable,
                })

        # Classify overall evolution type.
        if removed or any(_is_narrowing(tc) for tc in type_changes):
            etype = EvolutionType.BREAKING
        elif type_changes:
            etype = EvolutionType.COMPATIBLE
        elif added:
            etype = EvolutionType.ADDITIVE
        else:
            etype = EvolutionType.IDENTICAL

        return SchemaEvolution(
            evolution_type=etype,
            added_columns=added,
            removed_columns=removed,
            type_changes=type_changes,
            nullability_changes=nullability_changes,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WIDENING_PAIRS: set[tuple[str, str]] = {
    ("INT64", "FLOAT64"),
    ("INT64", "NUMERIC"),
    ("FLOAT64", "NUMERIC"),
    ("STRING", "STRING"),
}


def _is_narrowing(tc: dict) -> bool:
    return (tc["old_type"], tc["new_type"]) not in _WIDENING_PAIRS
