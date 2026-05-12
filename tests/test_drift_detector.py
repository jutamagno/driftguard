"""Tests for ADWIN and Page-Hinkley drift detectors."""

import pytest
from src.drift.detector import ADWIN, PageHinkley, FeatureDriftMonitor, DriftStatus


class TestADWIN:
    def test_no_drift_on_stable_stream(self):
        detector = ADWIN(delta=0.002)
        detections = 0
        for _ in range(500):
            if detector.update(0.5):
                detections += 1
        # Stable stream should have near-zero false positive rate
        assert detections == 0

    def test_detects_abrupt_shift(self):
        """Stream shifts from mean=0.1 to mean=0.9 — ADWIN must detect it."""
        detector = ADWIN(delta=0.002)

        for _ in range(300):
            detector.update(0.1)

        detections = 0
        for _ in range(300):
            if detector.update(0.9):
                detections += 1

        assert detections > 0, "ADWIN failed to detect abrupt shift from 0.1 to 0.9"

    def test_mean_tracks_current_distribution(self):
        detector = ADWIN(delta=0.002)
        for _ in range(500):
            detector.update(0.2)
        # After drift
        for _ in range(500):
            detector.update(0.8)
        # Mean should be closer to 0.8 than 0.2
        assert detector.mean > 0.5

    def test_detection_count_increments(self):
        detector = ADWIN(delta=0.002)
        for _ in range(200):
            detector.update(0.0)
        for _ in range(200):
            detector.update(1.0)
        assert detector.n_detections >= 1


class TestPageHinkley:
    def test_no_drift_stable(self):
        ph = PageHinkley(delta=0.005, lambda_=50.0)
        detections = [ph.update(0.5) for _ in range(500)]
        assert not any(detections)

    def test_detects_upward_shift(self):
        ph = PageHinkley(delta=0.005, lambda_=10.0)
        for _ in range(100):
            ph.update(0.0)
        detected = [ph.update(1.0) for _ in range(200)]
        assert any(detected), "Page-Hinkley failed to detect upward mean shift"


class TestFeatureDriftMonitor:
    def test_no_alerts_on_stable(self):
        monitor = FeatureDriftMonitor(["ctr", "cvr"], delta=0.002)
        for _ in range(500):
            alerts = monitor.update({"ctr": 0.05, "cvr": 0.02})
            assert alerts == []

    def test_alerts_on_drift(self):
        monitor = FeatureDriftMonitor(["ctr"], delta=0.002)
        for _ in range(300):
            monitor.update({"ctr": 0.05})

        alerts_after_drift = []
        for _ in range(300):
            alerts = monitor.update({"ctr": 0.95})
            alerts_after_drift.extend(alerts)

        assert len(alerts_after_drift) > 0
        assert all(a.feature_name == "ctr" for a in alerts_after_drift)
        assert all(a.status == DriftStatus.DRIFT for a in alerts_after_drift)

    def test_ignores_unknown_features(self):
        monitor = FeatureDriftMonitor(["ctr"], delta=0.002)
        alerts = monitor.update({"ctr": 0.05, "unknown_feature": 99.0})
        assert alerts == []

    def test_summary_contains_all_features(self):
        monitor = FeatureDriftMonitor(["ctr", "cvr"])
        monitor.update({"ctr": 0.1, "cvr": 0.05})
        summary = monitor.summary()
        assert "ctr" in summary
        assert "cvr" in summary
        assert "current_mean" in summary["ctr"]

    def test_n_samples_increments(self):
        monitor = FeatureDriftMonitor(["ctr"])
        for i in range(5):
            monitor.update({"ctr": 0.1})
        assert monitor.n_samples == 5
