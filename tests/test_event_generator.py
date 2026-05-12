"""Tests for the synthetic event generator and drift simulation."""

import pytest
from src.producer.event_generator import generate_events, ClickEvent, PHASE_CTR


class TestGenerateEvents:
    def test_yields_correct_count(self):
        events = list(generate_events(n_events=100))
        assert len(events) == 100

    def test_all_events_are_click_events(self):
        for event in generate_events(n_events=20):
            assert isinstance(event, ClickEvent)

    def test_event_types_valid(self):
        valid_types = {"impression", "click"}
        for event in generate_events(n_events=50):
            assert event.event_type in valid_types

    def test_phase_assignment(self):
        events = list(generate_events(n_events=100, drift_at=0.5))
        phase1 = [e for e in events[:50]]
        phase2 = [e for e in events[50:]]
        assert all(e.phase == 1 for e in phase1)
        assert all(e.phase == 2 for e in phase2)

    def test_drift_changes_ctr_distribution(self):
        """After drift, young_adult CTR for eletrônicos should drop."""
        events = list(generate_events(n_events=10_000, drift_at=0.5))

        def ctr(phase, segment, category):
            relevant = [
                e for e in events
                if e.phase == phase and e.user_segment == segment and e.category == category
            ]
            if not relevant:
                return 0.0
            return sum(1 for e in relevant if e.event_type == "click") / len(relevant)

        # Phase 1: young_adult eletrônicos CTR should be higher than phase 2
        ctr_p1 = ctr(1, "young_adult", "eletrônicos")
        ctr_p2 = ctr(2, "young_adult", "eletrônicos")
        assert ctr_p1 > ctr_p2, (
            f"Expected CTR to drop after drift: p1={ctr_p1:.3f}, p2={ctr_p2:.3f}"
        )

    def test_segments_and_devices_covered(self):
        events = list(generate_events(n_events=500))
        segments = {e.user_segment for e in events}
        devices = {e.device for e in events}
        assert "young_adult" in segments
        assert "mobile" in devices and "desktop" in devices

    def test_unique_event_ids(self):
        events = list(generate_events(n_events=100))
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))
