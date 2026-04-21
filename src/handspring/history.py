"""Ring buffer of per-hand HandFeatures samples with timestamps.

Provides enough state for motion detectors (wave, pinch, drag, clap)
to analyze the last ~1 second of hand motion without holding state
inside the classifiers themselves.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from handspring.types import HandFeatures


@dataclass(frozen=True)
class HandSample:
    """One historical hand sample."""

    features: HandFeatures
    timestamp: float  # seconds (monotonic clock); caller-provided


class HandHistory:
    """Fixed-capacity ring buffer of HandSamples, oldest first on iteration."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._buf: deque[HandSample] = deque(maxlen=capacity)

    def push(self, features: HandFeatures, timestamp: float) -> None:
        self._buf.append(HandSample(features=features, timestamp=timestamp))

    def samples(self) -> list[HandSample]:
        return list(self._buf)

    def latest(self) -> HandSample | None:
        return self._buf[-1] if self._buf else None

    def clear(self) -> None:
        self._buf.clear()

    def samples_since(self, since_timestamp: float) -> list[HandSample]:
        return [s for s in self._buf if s.timestamp >= since_timestamp]
