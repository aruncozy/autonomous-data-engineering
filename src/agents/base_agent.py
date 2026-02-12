"""Abstract base class for all pipeline agents."""

from __future__ import annotations

import logging
import signal
import threading
from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from src.common.config import PipelineConfig
from src.common.event_bus import EventBus
from src.common.lineage_tracker import LineageTracker
from src.common.models import AgentState, EventType, Layer, PipelineEvent
from src.common.state_manager import StateManager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Skeleton shared by Ingestion, Transformation, and Aggregation agents."""

    agent_name: str = "base"
    layer: Layer = Layer.BRONZE

    def __init__(
        self,
        config: PipelineConfig,
        event_bus: EventBus,
        state_manager: StateManager,
        lineage_tracker: LineageTracker,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.lineage_tracker = lineage_tracker

        self._running = False
        self._subscription_future = None
        self._health_server: HTTPServer | None = None
        self._state = AgentState(agent_id=self.agent_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe to events, restore state, enter run loop."""
        logger.info("Starting %s agent …", self.agent_name)

        # Restore last checkpoint.
        saved = self.state_manager.load_checkpoint(self.agent_name)
        if saved is not None:
            self._state = saved
            logger.info("Restored checkpoint for %s (status=%s)", self.agent_name, saved.status)

        self._state.status = "running"
        self.state_manager.save_checkpoint(self.agent_name, self._state)

        # Register signal handlers.
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Start health-check server in background.
        self._start_health_server()

        # Begin event subscription (blocks).
        self._running = True
        self._subscription_future = self.event_bus.subscribe(
            self._subscription_name(),
            self._dispatch_event,
        )

        try:
            self._subscription_future.result()
        except Exception:
            if self._running:
                raise

    def stop(self) -> None:
        """Graceful shutdown with checkpoint."""
        logger.info("Stopping %s agent …", self.agent_name)
        self._running = False

        if self._subscription_future is not None:
            self._subscription_future.cancel()

        self._state.status = "stopped"
        self.state_manager.save_checkpoint(self.agent_name, self._state)

        if self._health_server:
            self._health_server.shutdown()

        self.event_bus.close()
        logger.info("%s agent stopped.", self.agent_name)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _dispatch_event(self, event: PipelineEvent) -> None:
        """Route an event to the concrete handler with error wrapping."""
        try:
            self._state.status = "processing"
            self._state.current_task = f"{event.event_type.value}:{event.source_id}"
            self.state_manager.save_checkpoint(self.agent_name, self._state)

            self.process_event(event)

            self._state.status = "running"
            self._state.current_task = None
            self.state_manager.save_checkpoint(self.agent_name, self._state)
        except Exception as exc:
            self.handle_error(exc, event)

    @abstractmethod
    def process_event(self, event: PipelineEvent) -> None:
        """Handle a single pipeline event — implemented by each agent."""

    @abstractmethod
    def _subscription_name(self) -> str:
        """Return the Pub/Sub subscription this agent listens on."""

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def handle_error(self, error: Exception, event: PipelineEvent) -> None:
        """Route failures to DLQ and emit an ERROR event."""
        logger.exception(
            "Error in %s processing %s (source=%s, corr=%s): %s",
            self.agent_name,
            event.event_type.value,
            event.source_id,
            event.correlation_id,
            error,
        )

        self._state.status = "error"
        self._state.metadata["last_error"] = str(error)
        self.state_manager.save_checkpoint(self.agent_name, self._state)

        error_event = PipelineEvent(
            event_type=EventType.ERROR,
            source_id=event.source_id,
            layer=self.layer,
            correlation_id=event.correlation_id,
            payload={
                "original_event": event.event_type.value,
                "error": str(error),
                "agent": self.agent_name,
            },
        )
        try:
            self.event_bus.publish(
                self.config.pubsub.ingestion_topic,
                error_event,
            )
        except Exception:
            logger.exception("Failed to publish error event")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _start_health_server(self) -> None:
        agent_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == agent_ref.config.health_check.path:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass  # suppress default stderr logging

        port = self.config.health_check.port
        self._health_server = HTTPServer(("0.0.0.0", port), _Handler)
        t = threading.Thread(target=self._health_server.serve_forever, daemon=True)
        t.start()
        logger.info("Health check listening on :%d%s", port, self.config.health_check.path)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info("Received signal %d — shutting down", signum)
        self.stop()
