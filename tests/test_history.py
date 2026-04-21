"""History ring buffer tests."""

from __future__ import annotations

from handspring.history import HandHistory
from handspring.types import HandFeatures


def _hf(x: float = 0.5, y: float = 0.5, pinch: float = 0.0) -> HandFeatures:
    return HandFeatures(x=x, y=y, z=0.0, openness=1.0, pinch=pinch, index_x=x, index_y=y)


def test_empty_history_no_samples():
    h = HandHistory(capacity=10)
    assert h.samples() == []


def test_push_stores_sample():
    h = HandHistory(capacity=10)
    h.push(_hf(x=0.1), timestamp=0.0)
    samples = h.samples()
    assert len(samples) == 1
    assert samples[0].features.x == 0.1
    assert samples[0].timestamp == 0.0


def test_ring_buffer_discards_oldest():
    h = HandHistory(capacity=3)
    for i in range(5):
        h.push(_hf(x=float(i)), timestamp=float(i))
    samples = h.samples()
    assert len(samples) == 3
    assert [s.features.x for s in samples] == [2.0, 3.0, 4.0]


def test_samples_in_chronological_order():
    h = HandHistory(capacity=5)
    for i in range(4):
        h.push(_hf(x=float(i)), timestamp=float(i))
    xs = [s.features.x for s in h.samples()]
    assert xs == [0.0, 1.0, 2.0, 3.0]


def test_clear_empties_buffer():
    h = HandHistory(capacity=5)
    h.push(_hf(), 0.0)
    h.push(_hf(), 1.0)
    h.clear()
    assert h.samples() == []


def test_latest_returns_most_recent():
    h = HandHistory(capacity=3)
    h.push(_hf(x=0.1), 0.0)
    h.push(_hf(x=0.2), 1.0)
    assert h.latest() is not None
    assert h.latest().features.x == 0.2  # type: ignore[union-attr]


def test_latest_none_when_empty():
    h = HandHistory(capacity=3)
    assert h.latest() is None


def test_samples_since_time():
    h = HandHistory(capacity=10)
    for i in range(10):
        h.push(_hf(), timestamp=float(i) * 0.1)
    # since t=0.5 → samples at 0.5, 0.6, 0.7, 0.8, 0.9 = 5 samples
    recent = h.samples_since(0.45)
    assert len(recent) == 5
