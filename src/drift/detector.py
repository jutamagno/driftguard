"""Concept drift detection using ADWIN and Page-Hinkley test.

ADWIN (ADaptive WINdowing) detects distribution shifts in streaming data
by maintaining a variable-size sliding window and testing for statistical
differences between sub-windows.

This is what separates a toy ML pipeline from a production one: knowing
*when* your features or model have drifted, before it impacts revenue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class DriftStatus(str, Enum):
    STABLE = "stable"
    WARNING = "warning"
    DRIFT = "drift"


@dataclass
class DriftAlert:
    feature_name: str
    status: DriftStatus
    old_mean: float
    new_mean: float
    magnitude: float
    n_samples_seen: int


class ADWIN:
    """Adaptive Windowing drift detector.

    Maintains a window W. When a statistically significant difference is
    detected between two sub-windows of W, drift is declared and the
    older sub-window is dropped.

    Reference: Bifet & Gavalda (2007).
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: deque[float] = deque()
        self._n = 0
        self._sum = 0.0
        self._variance = 0.0
        self.n_detections = 0

    def update(self, value: float) -> bool:
        """Add new value. Returns True if drift detected."""
        self._window.append(value)
        self._n += 1
        old_mean = self._sum / max(self._n - 1, 1)
        self._sum += value
        delta_value = value - old_mean
        self._variance += delta_value * (value - self._sum / self._n)

        return self._check_drift()

    def _check_drift(self) -> bool:
        window = list(self._window)
        n = len(window)
        if n < 2:
            return False

        total_sum = sum(window)
        total_mean = total_sum / n

        n0_sum = 0.0
        for i in range(n - 1):
            n0 = i + 1
            n1 = n - n0
            n0_sum += window[i]
            mean0 = n0_sum / n0
            mean1 = (total_sum - n0_sum) / n1

            epsilon_cut = self._epsilon_cut(n0, n1, self.delta)
            if abs(mean0 - mean1) >= epsilon_cut:
                # drop old part of window
                for _ in range(n0):
                    if self._window:
                        removed = self._window.popleft()
                        self._sum -= removed
                        self._n -= 1
                self.n_detections += 1
                return True

        return False

    @staticmethod
    def _epsilon_cut(n0: int, n1: int, delta: float) -> float:
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        return math.sqrt((1.0 / (2 * m)) * math.log(4 * (n0 + n1) / delta))

    @property
    def mean(self) -> float:
        if self._n == 0:
            return 0.0
        return self._sum / self._n


class PageHinkley:
    """Page-Hinkley test for detecting persistent changes in mean.

    Better than ADWIN for detecting gradual drift (slow trends).
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 0.9999):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._sum = 0.0
        self._n = 0
        self._mean = 0.0
        self._ph = 0.0
        self._ph_min = float("inf")

    def update(self, value: float) -> bool:
        self._n += 1
        if self._n == 1:
            self._mean = value  # warm-start to avoid false positives at startup
        else:
            self._mean = self._mean * self.alpha + value * (1 - self.alpha)
        self._sum += value - self._mean - self.delta
        self._ph_min = min(self._ph_min, self._sum)
        self._ph = self._sum - self._ph_min
        return self._ph > self.lambda_


class FeatureDriftMonitor:
    """Monitors multiple features for drift simultaneously."""

    def __init__(self, features: list[str], delta: float = 0.002):
        self.features = features
        self.detectors: dict[str, ADWIN] = {f: ADWIN(delta) for f in features}
        self.baselines: dict[str, list[float]] = {f: [] for f in features}
        self.n_samples = 0

    def update(self, feature_values: dict[str, float]) -> list[DriftAlert]:
        self.n_samples += 1
        alerts = []

        for name, value in feature_values.items():
            if name not in self.detectors:
                continue

            detector = self.detectors[name]
            old_mean = detector.mean
            self.baselines[name].append(value)

            drifted = detector.update(value)
            if drifted:
                alerts.append(DriftAlert(
                    feature_name=name,
                    status=DriftStatus.DRIFT,
                    old_mean=old_mean,
                    new_mean=detector.mean,
                    magnitude=abs(detector.mean - old_mean),
                    n_samples_seen=self.n_samples,
                ))

        return alerts

    def summary(self) -> dict[str, dict]:
        return {
            name: {
                "current_mean": detector.mean,
                "n_detections": detector.n_detections,
            }
            for name, detector in self.detectors.items()
        }
