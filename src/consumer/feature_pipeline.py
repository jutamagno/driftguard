"""Streaming feature computation pipeline with online model updates.

Consumes Kafka clickstream events and:
1. Computes real-time features (sliding window aggregations)
2. Updates an online CTR prediction model (River)
3. Detects feature drift via ADWIN
4. Writes features to Redis for serving

This pattern is what separates an MLE who only knows batch training from one
who understands production ML systems.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import redis
from kafka import KafkaConsumer

from driftguard.drift.detector import FeatureDriftMonitor, DriftAlert


try:
    import river.linear_model as lm
    import river.preprocessing as pp
    import river.metrics as metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


@dataclass
class ComputedFeatures:
    user_id: str
    item_id: str
    category: str
    user_ctr_1h: float     # user click-through rate in last hour
    item_ctr_1h: float     # item CTR in last hour
    category_ctr_1h: float
    user_session_length: int
    label: int             # 1 = click, 0 = impression


class SlidingWindowAggregator:
    """Computes rolling aggregations over time-windowed events."""

    def __init__(self, window_seconds: int = 3600):
        self.window = window_seconds
        # {key: deque of (timestamp, value)}
        self._clicks: dict[str, deque] = defaultdict(deque)
        self._impressions: dict[str, deque] = defaultdict(deque)

    def add_event(self, key: str, is_click: bool, timestamp: float) -> None:
        bucket = self._clicks if is_click else self._impressions
        bucket[key].append(timestamp)
        self._evict(key, timestamp)

    def _evict(self, key: str, now: float) -> None:
        cutoff = now - self.window
        for bucket in (self._clicks, self._impressions):
            dq = bucket[key]
            while dq and dq[0] < cutoff:
                dq.popleft()

    def ctr(self, key: str) -> float:
        clicks = len(self._clicks[key])
        impressions = len(self._impressions[key]) + clicks
        return clicks / impressions if impressions > 0 else 0.0


class OnlineCTRModel:
    """Logistic regression trained incrementally with River.

    Updates on every event — no batch retraining required.
    """

    def __init__(self):
        if not RIVER_AVAILABLE:
            raise ImportError("pip install river")
        self.model = pp.StandardScaler() | lm.LogisticRegression()
        self.metric = metrics.ROCAUC()
        self.n_updates = 0

    def learn(self, features: dict[str, float], label: int) -> float:
        x = features
        y_pred = self.model.predict_proba_one(x)
        self.model.learn_one(x, label)
        self.metric.update(label, y_pred.get(1, 0.0))
        self.n_updates += 1
        return y_pred.get(1, 0.0)

    @property
    def roc_auc(self) -> float:
        return self.metric.get()


class FeaturePipeline:
    """Main consumer loop: Kafka → features → online model → Redis."""

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        topic: str = "clickstream",
    ):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            group_id="feature-pipeline",
        )
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.aggregator = SlidingWindowAggregator(window_seconds=3600)
        self.drift_monitor = FeatureDriftMonitor(
            features=["user_ctr_1h", "item_ctr_1h", "category_ctr_1h"]
        )
        self.online_model = OnlineCTRModel() if RIVER_AVAILABLE else None
        self.session_lengths: dict[str, int] = defaultdict(int)

    def run(self, max_messages: int | None = None) -> None:
        processed = 0
        for message in self.consumer:
            event = message.value
            self._process(event)
            processed += 1
            if max_messages and processed >= max_messages:
                break

    def _process(self, event: dict) -> None:
        now = time.time()
        user_id = event["user_id"]
        item_id = event["item_id"]
        category = event["category"]
        is_click = event["event_type"] == "click"

        self.aggregator.add_event(f"user:{user_id}", is_click, now)
        self.aggregator.add_event(f"item:{item_id}", is_click, now)
        self.aggregator.add_event(f"cat:{category}", is_click, now)
        self.session_lengths[event["session_id"]] += 1

        features = ComputedFeatures(
            user_id=user_id,
            item_id=item_id,
            category=category,
            user_ctr_1h=self.aggregator.ctr(f"user:{user_id}"),
            item_ctr_1h=self.aggregator.ctr(f"item:{item_id}"),
            category_ctr_1h=self.aggregator.ctr(f"cat:{category}"),
            user_session_length=self.session_lengths[event["session_id"]],
            label=int(is_click),
        )

        self._write_to_redis(features)

        alerts = self.drift_monitor.update({
            "user_ctr_1h": features.user_ctr_1h,
            "item_ctr_1h": features.item_ctr_1h,
            "category_ctr_1h": features.category_ctr_1h,
        })
        for alert in alerts:
            self._handle_drift(alert)

        if self.online_model:
            feat_dict = {
                "user_ctr_1h": features.user_ctr_1h,
                "item_ctr_1h": features.item_ctr_1h,
                "category_ctr_1h": features.category_ctr_1h,
                "session_length": float(features.user_session_length),
            }
            self.online_model.learn(feat_dict, features.label)

    def _write_to_redis(self, features: ComputedFeatures) -> None:
        key = f"features:user:{features.user_id}"
        self.redis.hset(key, mapping={
            "user_ctr_1h": features.user_ctr_1h,
            "item_ctr_1h": features.item_ctr_1h,
            "category_ctr_1h": features.category_ctr_1h,
            "session_length": features.user_session_length,
        })
        self.redis.expire(key, 7200)

    def _handle_drift(self, alert: DriftAlert) -> None:
        print(
            f"[DRIFT] {alert.feature_name}: "
            f"{alert.old_mean:.4f} → {alert.new_mean:.4f} "
            f"(magnitude={alert.magnitude:.4f}, n={alert.n_samples_seen})"
        )
        self.redis.publish("drift:alerts", json.dumps({
            "feature": alert.feature_name,
            "old_mean": alert.old_mean,
            "new_mean": alert.new_mean,
            "magnitude": alert.magnitude,
        }))
