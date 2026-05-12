"""Tests for sliding window aggregator."""

import pytest
import time
from src.consumer.feature_pipeline import SlidingWindowAggregator


class TestSlidingWindowAggregator:
    def test_ctr_no_events(self):
        agg = SlidingWindowAggregator(window_seconds=3600)
        assert agg.ctr("user:x") == 0.0

    def test_all_clicks(self):
        agg = SlidingWindowAggregator(window_seconds=3600)
        now = time.time()
        for _ in range(5):
            agg.add_event("user:x", is_click=True, timestamp=now)
        assert agg.ctr("user:x") == pytest.approx(1.0)

    def test_all_impressions(self):
        agg = SlidingWindowAggregator(window_seconds=3600)
        now = time.time()
        for _ in range(5):
            agg.add_event("user:x", is_click=False, timestamp=now)
        assert agg.ctr("user:x") == pytest.approx(0.0)

    def test_ctr_ratio(self):
        agg = SlidingWindowAggregator(window_seconds=3600)
        now = time.time()
        agg.add_event("u", is_click=True, timestamp=now)   # 1 click
        agg.add_event("u", is_click=False, timestamp=now)  # 1 impression
        # ctr = 1 click / (1 click + 1 impression) = 0.5
        assert agg.ctr("u") == pytest.approx(0.5)

    def test_evicts_old_events(self):
        agg = SlidingWindowAggregator(window_seconds=60)
        old_time = time.time() - 120  # 2 minutes ago — outside 60s window
        now = time.time()

        agg.add_event("u", is_click=True, timestamp=old_time)
        # Trigger eviction by adding a new event
        agg.add_event("u", is_click=False, timestamp=now)
        # Old click should be evicted, only the new impression remains
        assert agg.ctr("u") == pytest.approx(0.0)

    def test_independent_keys(self):
        agg = SlidingWindowAggregator()
        now = time.time()
        agg.add_event("user:A", is_click=True, timestamp=now)
        agg.add_event("user:B", is_click=False, timestamp=now)
        assert agg.ctr("user:A") == pytest.approx(1.0)
        assert agg.ctr("user:B") == pytest.approx(0.0)
