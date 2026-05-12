"""Synthetic clickstream event generator.

Simulates realistic user behavior including concept drift:
- Phase 1 (t=0..N/2): stable CTR pattern
- Phase 2 (t=N/2..N): shifted user preferences (drift event)

This is what real production systems face: user behavior changes over time
and batch-trained models silently degrade.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Generator
from kafka import KafkaProducer


@dataclass
class ClickEvent:
    event_id: str
    user_id: str
    item_id: str
    session_id: str
    event_type: str        # impression | click | add_to_cart | purchase
    category: str
    price: float
    timestamp: str
    user_segment: str      # young_adult | parent | professional
    device: str            # mobile | desktop
    phase: int             # 1 = stable, 2 = drift (for evaluation only)


CATEGORIES = ["eletrônicos", "moda", "casa", "esportes", "beleza"]
SEGMENTS = ["young_adult", "parent", "professional"]
DEVICES = ["mobile", "desktop"]

# Phase 1: young_adults click eletrônicos most
# Phase 2: after drift, young_adults shift to moda (concept drift)
PHASE_CTR = {
    1: {"young_adult": {"eletrônicos": 0.15, "moda": 0.05, "casa": 0.03},
        "parent":      {"eletrônicos": 0.05, "moda": 0.08, "casa": 0.12},
        "professional":{"eletrônicos": 0.10, "moda": 0.06, "esportes": 0.09}},
    2: {"young_adult": {"eletrônicos": 0.05, "moda": 0.18, "casa": 0.03},  # drift!
        "parent":      {"eletrônicos": 0.05, "moda": 0.08, "casa": 0.12},
        "professional":{"eletrônicos": 0.10, "moda": 0.06, "esportes": 0.09}},
}


def _get_ctr(phase: int, segment: str, category: str) -> float:
    return PHASE_CTR.get(phase, PHASE_CTR[1]).get(segment, {}).get(category, 0.04)


def generate_events(
    n_events: int = 100_000,
    drift_at: float = 0.5,
) -> Generator[ClickEvent, None, None]:
    for i in range(n_events):
        phase = 1 if i < n_events * drift_at else 2
        segment = random.choice(SEGMENTS)
        category = random.choice(CATEGORIES)
        ctr = _get_ctr(phase, segment, category)
        event_type = "click" if random.random() < ctr else "impression"

        yield ClickEvent(
            event_id=f"ev_{i:08d}",
            user_id=f"u_{random.randint(0, 9999):05d}",
            item_id=f"item_{random.randint(0, 4999):05d}",
            session_id=f"sess_{random.randint(0, 49999):06d}",
            event_type=event_type,
            category=category,
            price=round(random.uniform(10, 2000), 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_segment=segment,
            device=random.choice(DEVICES),
            phase=phase,
        )


class EventProducer:
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "clickstream"):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
        )
        self.topic = topic

    def stream(self, events_per_second: float = 100.0, **kwargs) -> None:
        delay = 1.0 / events_per_second
        for event in generate_events(**kwargs):
            self.producer.send(
                self.topic,
                key=event.user_id,
                value=asdict(event),
            )
            time.sleep(delay)

    def flush(self) -> None:
        self.producer.flush()
