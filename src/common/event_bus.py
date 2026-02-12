"""Pub/Sub-backed event bus for inter-agent communication."""

from __future__ import annotations

import json
import logging
from typing import Callable

from google.cloud import pubsub_v1

from src.common.models import PipelineEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Publish / subscribe to pipeline events via Google Cloud Pub/Sub."""

    def __init__(self, project_id: str) -> None:
        self._project_id = project_id
        self._publisher = pubsub_v1.PublisherClient()
        self._subscriber = pubsub_v1.SubscriberClient()

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, topic: str, event: PipelineEvent) -> str:
        """Serialize *event* and publish to the given Pub/Sub *topic*.

        Returns the published message ID.
        """
        topic_path = self._publisher.topic_path(self._project_id, topic)

        data = event.model_dump_json().encode("utf-8")

        attributes = {
            "event_type": event.event_type.value,
            "source_id": event.source_id,
            "layer": event.layer.value,
            "correlation_id": event.correlation_id,
        }

        future = self._publisher.publish(topic_path, data=data, **attributes)
        message_id = future.result()
        logger.info(
            "Published %s to %s (msg=%s, corr=%s)",
            event.event_type.value,
            topic,
            message_id,
            event.correlation_id,
        )
        return message_id

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        subscription: str,
        callback: Callable[[PipelineEvent], None],
        *,
        max_messages: int = 10,
        ack_deadline: int = 60,
    ) -> pubsub_v1.subscriber.futures.StreamingPullFuture:
        """Start a streaming-pull subscription.

        *callback* receives a deserialized ``PipelineEvent``.  Ack is sent
        automatically on success; nack on exception.
        """
        subscription_path = self._subscriber.subscription_path(
            self._project_id, subscription,
        )

        def _wrapper(message: pubsub_v1.subscriber.message.Message) -> None:
            try:
                event = PipelineEvent.model_validate_json(message.data)
                logger.debug(
                    "Received %s from %s (corr=%s)",
                    event.event_type.value,
                    subscription,
                    event.correlation_id,
                )
                callback(event)
                message.ack()
            except Exception:
                logger.exception("Error processing message %s", message.message_id)
                message.nack()

        flow_control = pubsub_v1.types.FlowControl(max_messages=max_messages)
        future = self._subscriber.subscribe(
            subscription_path,
            callback=_wrapper,
            flow_control=flow_control,
        )
        logger.info("Subscribed to %s", subscription)
        return future

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._publisher.transport.close()
        self._subscriber.close()
