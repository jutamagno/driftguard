# driftguard

[![CI](https://github.com/jutamagno/driftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/jutamagno/driftguard/actions/workflows/ci.yml)

Streaming feature platform with online learning and concept drift detection.

**The problem:** Batch-trained models silently degrade as user behavior shifts. Most ML pipelines have no mechanism to detect this. `driftguard` continuously monitors feature distributions and model performance in real time, alerting before silent degradation becomes a business problem.

---

## How it differs from a standard feature store

| Standard Feast setup | driftguard |
|---|---|
| Batch materialization | Real-time streaming features |
| Static model | Online learning (updates per event) |
| No drift detection | ADWIN + Page-Hinkley monitors |
| No alerting | Redis pub/sub drift alerts |

---

## Architecture

```
Kafka topic: "clickstream"
    │
    ▼
[SlidingWindowAggregator]   computes user/item/category CTR over a 1h window
    │
    ├──→ [Redis]              online feature serving (<5ms latency)
    │      key: "features:user:{user_id}"   TTL: 7200s
    │
    ├──→ [OnlineCTRModel]     River LogisticRegression — learns per event, no batch retraining
    │
    └──→ [FeatureDriftMonitor]
              ├── ADWIN        abrupt drift (Bifet & Gavalda, 2007)
              └── PageHinkley  gradual drift (persistent trend changes)
                       │
                       └──→ Redis pub/sub: "drift:alerts"
```

---

## The drift simulation

The event generator intentionally injects a concept drift at the midpoint of the stream:

| Phase | Segment | Category | CTR |
|---|---|---|---|
| 1 (stable) | `young_adult` | `eletrônicos` | 0.15 |
| 1 (stable) | `young_adult` | `moda` | 0.05 |
| 2 (drift)  | `young_adult` | `eletrônicos` | 0.05 |
| 2 (drift)  | `young_adult` | `moda` | 0.18 |

A model trained on Phase 1 data will silently degrade in Phase 2. ADWIN detects the shift within ~300 events of the inflection point.

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for Kafka and Redis)
- pip / uv

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/jutamagno/driftguard
cd driftguard
pip install -e ".[dev]"

# 2. Start infrastructure (Kafka + Redis)
cd docker && docker-compose up -d && cd ..

# 3. In one terminal — produce events (drift injected at 50%)
python -m driftguard.producer.event_generator --n-events 50000 --eps 200

# 4. In another terminal — consume and detect drift
python -m driftguard.consumer.feature_pipeline
```

Drift alerts appear in stdout and are published to Redis `drift:alerts`:

```
[DRIFT] category_ctr_1h: 0.0821 → 0.1523 (magnitude=0.0702, n=25311)
```

---

## Local development

### Project structure

```
driftguard/
├── src/
│   ├── consumer/
│   │   └── feature_pipeline.py   SlidingWindowAggregator, OnlineCTRModel, FeaturePipeline
│   ├── drift/
│   │   └── detector.py           ADWIN, PageHinkley, FeatureDriftMonitor, DriftAlert
│   └── producer/
│       └── event_generator.py    ClickEvent, EventProducer, PHASE_CTR
├── docker/
│   └── docker-compose.yml        Zookeeper, Kafka, Redis, Kafka UI
├── notebooks/
│   └── 01_drift_simulation.ipynb end-to-end analysis with plots
├── tests/
│   ├── test_drift_detector.py
│   ├── test_event_generator.py
│   └── test_sliding_window.py
└── pyproject.toml
```

### Infrastructure ports

| Service | Port | Notes |
|---|---|---|
| Kafka | `9092` | Plaintext, auto-creates topics |
| Zookeeper | `2181` | Required by Kafka |
| Redis | `6379` | maxmemory 512mb, allkeys-lru |
| Kafka UI | `8080` | Browse topics and messages |

### Producer flags

```
python -m driftguard.producer.event_generator [OPTIONS]

  --n-events INT    Total events to generate  [default: 100000]
  --eps FLOAT       Events per second          [default: 100.0]
  --drift-at FLOAT  Fraction of stream where drift is injected  [default: 0.5]
  --dry-run         Print events to stdout instead of sending to Kafka
```

### Consumer flags

```
python -m driftguard.consumer.feature_pipeline [OPTIONS]

  --kafka-servers STR   Bootstrap servers  [default: localhost:9092]
  --redis-host STR      Redis host          [default: localhost]
  --redis-port INT      Redis port          [default: 6379]
  --topic STR           Kafka topic         [default: clickstream]
```

### Querying features from Redis

```bash
# Get all features for a specific user
redis-cli hgetall "features:user:u_00042"
# → user_ctr_1h   0.0921
# → item_ctr_1h   0.0714
# → category_ctr_1h 0.1204
# → session_length  7

# Subscribe to drift alerts
redis-cli subscribe drift:alerts
```

### Subscribing to drift alerts in Python

```python
import redis, json

r = redis.Redis(decode_responses=True)
pubsub = r.pubsub()
pubsub.subscribe("drift:alerts")

for message in pubsub.listen():
    if message["type"] == "message":
        alert = json.loads(message["data"])
        print(f"[DRIFT] {alert['feature']}: {alert['old_mean']:.4f} → {alert['new_mean']:.4f}")
```

---

## Features computed

| Feature | Description | Key format |
|---|---|---|
| `user_ctr_1h` | User click-through rate in last 60 min | `user:{user_id}` |
| `item_ctr_1h` | Item CTR in last 60 min | `item:{item_id}` |
| `category_ctr_1h` | Category CTR in last 60 min | `cat:{category}` |
| `user_session_length` | Number of events in current session | `sess:{session_id}` |

All features are evicted from Redis after 2 hours (TTL 7200s).

---

## Running tests

```bash
# All tests (no Docker required — mocks Kafka/Redis)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests cover ADWIN detection logic, Page-Hinkley threshold behavior, sliding window eviction, and event generation phase transitions.

---

## Design decisions

**Why ADWIN over a fixed-window mean test?**  
ADWIN maintains a variable-size window and detects abrupt shifts without requiring a predefined window size or significance threshold. Fixed-window tests need you to know in advance how fast drift will occur.

**Why Page-Hinkley in addition to ADWIN?**  
ADWIN detects abrupt changes; Page-Hinkley detects persistent gradual trends. Production drift is often both — a sudden event (new campaign) followed by a gradual stabilization.

**Why River for online learning?**  
River's incremental API (`learn_one`, `predict_proba_one`) fits the streaming architecture natively. No in-memory dataset accumulation; the model's memory footprint is constant regardless of stream length.

**Why Redis pub/sub for drift alerts?**  
The alert consumer (a downstream retraining trigger, an SLO watchdog, or an on-call pager) should be decoupled from the feature pipeline. Redis pub/sub lets multiple consumers subscribe independently without coordinating.

---

## Roadmap

- [ ] **FastAPI metrics endpoint** — expose current feature means, drift status, and online model AUC at `GET /metrics`
- [ ] **Persistent drift log** — write all drift alerts to a time-series store (TimescaleDB / InfluxDB) for post-hoc analysis
- [ ] **Multi-feature correlation drift** — current detection is per-feature; joint distribution shifts (e.g., `user_ctr_1h` ↑ while `item_ctr_1h` ↓) are not caught
- [ ] **Configurable detector thresholds** — ADWIN `delta` and Page-Hinkley `lambda_` are hardcoded; expose them via environment variables or a config file
- [ ] **Graceful Kafka reconnect** — the consumer loop raises on disconnect; add exponential backoff and reconnect logic
- [ ] **Kubernetes manifests** — Helm chart for the producer/consumer pair with autoscaling based on Kafka consumer lag

## Relation to other projects

`driftguard` is the **feature-service** inside [`personaflow`](../personaflow), and is cited in [`recsys-adtech`](../recsys-adtech) as the foundation for concept drift handling in fatigue modeling.
