"""Entrypoint — agent runner for Data Engineering Agents."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import yaml

from src.common.config import load_config
from src.common.event_bus import EventBus
from src.common.lineage_tracker import LineageTracker
from src.common.logging_config import setup_logging
from src.common.schema_registry import SchemaRegistry
from src.common.state_manager import StateManager
from src.llm.gemini_client import GeminiClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Engineering Agent Runner")
    parser.add_argument(
        "--agent",
        choices=["ingestion", "transformation", "aggregation"],
        required=True,
        help="Which agent to run",
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Path to config directory (default: config/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Trace event flow without executing GCP operations",
    )
    args = parser.parse_args()

    # Load configuration.
    config = load_config(args.config_dir)
    setup_logging(agent_id=args.agent)
    logger = logging.getLogger(__name__)

    if args.dry_run:
        logger.info("DRY-RUN mode — no GCP operations will be executed")

    logger.info(
        "Starting %s agent (project=%s, region=%s)",
        args.agent,
        config.gcp.project_id,
        config.gcp.region,
    )

    # Initialize shared services.
    event_bus = EventBus(config.gcp.project_id)
    state_manager = StateManager(
        config.gcp.project_id,
        config.firestore.state_collection,
    )
    lineage_tracker = LineageTracker(
        config.gcp.project_id,
        config.gcp.region,
    )
    schema_registry = SchemaRegistry(
        config.gcp.project_id,
        config.firestore.schema_collection,
    )
    gemini_client = GeminiClient(
        model_name=config.llm.model,
        fallback_model=config.llm.fallback_model,
        max_output_tokens=config.llm.max_output_tokens,
        temperature=config.llm.temperature,
        retry_attempts=config.llm.retry_attempts,
        retry_backoff=config.llm.retry_backoff_seconds,
    )

    # Load quality rules for the transformation agent.
    quality_rules: dict = {}
    quality_rules_path = Path(args.config_dir) / "quality_rules.yaml"
    if quality_rules_path.exists():
        with open(quality_rules_path) as f:
            quality_rules = yaml.safe_load(f) or {}

    # Instantiate the selected agent.
    if args.agent == "ingestion":
        from src.agents.ingestion.agent import IngestionAgent

        agent = IngestionAgent(
            config=config,
            event_bus=event_bus,
            state_manager=state_manager,
            lineage_tracker=lineage_tracker,
            schema_registry=schema_registry,
            gemini_client=gemini_client,
        )
    elif args.agent == "transformation":
        from src.agents.transformation.agent import TransformationAgent

        agent = TransformationAgent(
            config=config,
            event_bus=event_bus,
            state_manager=state_manager,
            lineage_tracker=lineage_tracker,
            schema_registry=schema_registry,
            gemini_client=gemini_client,
            quality_rules=quality_rules,
        )
    elif args.agent == "aggregation":
        from src.agents.aggregation.agent import AggregationAgent

        agent = AggregationAgent(
            config=config,
            event_bus=event_bus,
            state_manager=state_manager,
            lineage_tracker=lineage_tracker,
            schema_registry=schema_registry,
            gemini_client=gemini_client,
        )
    else:
        logger.error("Unknown agent: %s", args.agent)
        sys.exit(1)

    # Start the agent (blocks until shutdown signal).
    try:
        agent.start()
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
        agent.stop()
    except Exception:
        logger.exception("Agent crashed")
        agent.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
