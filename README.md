# driftguard

Streaming feature platform with online learning and concept drift detection.

**The problem**: Batch-trained models silently degrade as user behavior shifts.
Most ML pipelines have no mechanism to detect this. `driftguard` continuously
monitors feature distributions and model performance in real time, alerting
before silent degradation becomes a business problem.

## What makes this different from a standard feature store

| Standard Feast setup | driftguard |
|---|---|
| Batch materialization | Real-time streaming features |
| Static model | Online learning (updates per event) |
| No drift detection | ADWIN + Page-Hinkley monitors |
| No alerting | Redis pub/sub drift alerts |

## Architecture

```
Kafka (clickstream)
    │
    ▼
[SlidingWindowAggregator]  ← computes user/item/category CTR (1h window)
    │
    ├──→ [Redis]            ← online feature serving (<5ms)
    │
    ├──→ [OnlineCTRModel]   ← River LogisticRegression, learns per event
    │
    └──→ [FeatureDriftMonitor]
              ├── ADWIN (abrupt drift)
              └── PageHinkley (gradual drift)
                       │
                       └──→ Redis pub/sub: "drift:alerts"
```

## The drift simulation

The event generator intentionally introduces a concept drift at t=N/2:
- **Phase 1**: `young_adult` users prefer `eletrônicos` (CTR=0.15)
- **Phase 2**: same users shift to `moda` (CTR=0.18) — `eletrônicos` drops to 0.05

A batch model trained on Phase 1 data will silently degrade in Phase 2.
`driftguard` detects this within ~300 events of the shift.

## Quickstart

```bash
# Start infrastructure
cd docker && docker-compose up -d

# Install
pip install -e .

# Run producer (generates drift at 50% of events)
python -m driftguard.producer.event_generator --n-events 50000 --eps 200

# Run consumer (watch for drift alerts in stdout)
python -m driftguard.consumer.feature_pipeline

# Query features from Redis
redis-cli hgetall "features:user:u_00042"
```

## Monitoring drift alerts

```python
import redis
r = redis.Redis()
pubsub = r.pubsub()
pubsub.subscribe("drift:alerts")
for message in pubsub.listen():
    print(message)
```
