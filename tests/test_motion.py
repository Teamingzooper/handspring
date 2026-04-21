"""Motion detector tests."""

from __future__ import annotations

import math

from handspring.history import HandHistory
from handspring.motion import MotionDetector, bi_hand_clap_detector
from handspring.types import HandFeatures


def _hf(x: float = 0.5, y: float = 0.3, pinch: float = 0.0) -> HandFeatures:
    return HandFeatures(x=x, y=y, z=0.0, openness=1.0, pinch=pinch)


def _fill(history: HandHistory, samples: list[tuple[float, HandFeatures]]) -> None:
    for t, f in samples:
        history.push(f, t)


def test_pinch_event_on_rising_edge():
    h = HandHistory(capacity=30)
    d = MotionDetector()
    # Below threshold — no event.
    _fill(h, [(0.0, _hf(pinch=0.0)), (0.033, _hf(pinch=0.3))])
    update = d.update(h, now=0.033)
    assert update.event is None
    # Rise past 0.85 — event fires once.
    h.push(_hf(pinch=0.9), 0.066)
    update = d.update(h, now=0.066)
    assert update.event == "pinch"
    assert update.pinching is True
    # Holding pinch — no repeat.
    h.push(_hf(pinch=0.95), 0.099)
    update = d.update(h, now=0.099)
    assert update.event is None
    assert update.pinching is True


def test_expand_event_on_falling_edge():
    h = HandHistory(capacity=30)
    d = MotionDetector()
    _fill(h, [(0.0, _hf(pinch=0.9))])
    d.update(h, now=0.0)  # enters pinched state; cooldown set to 0.3
    # Release after cooldown expires (> 0.3 s) so the falling-edge check passes.
    h.push(_hf(pinch=0.3), 0.4)
    update = d.update(h, now=0.4)
    assert update.event == "expand"
    assert update.pinching is False


def test_drag_start_after_sustained_motion_while_pinching():
    h = HandHistory(capacity=30)
    d = MotionDetector()
    # Pinch on, then move horizontally for > 0.2 s
    x = 0.2
    t = 0.0
    while t <= 0.35:
        x += 0.02  # 0.02/frame × 30 fps ≈ 0.6 units/sec
        h.push(_hf(x=x, pinch=0.95), t)
        update = d.update(h, now=t)
        t += 0.033
    assert update.dragging is True
    # Event "drag_start" should have fired exactly once during that sequence.
    # Record events by re-running a fresh detector:
    h2 = HandHistory(capacity=30)
    d2 = MotionDetector()
    events = []
    x = 0.2
    t = 0.0
    while t <= 0.35:
        x += 0.02
        h2.push(_hf(x=x, pinch=0.95), t)
        u = d2.update(h2, now=t)
        if u.event is not None:
            events.append(u.event)
        t += 0.033
    assert "drag_start" in events
    assert events.count("drag_start") == 1


def test_drag_end_on_release():
    h = HandHistory(capacity=30)
    d = MotionDetector()
    # Drag for a bit
    x = 0.2
    for i in range(12):
        x += 0.03
        h.push(_hf(x=x, pinch=0.95), i * 0.033)
        d.update(h, now=i * 0.033)
    # Release pinch
    h.push(_hf(x=x, pinch=0.2), 0.4)
    update = d.update(h, now=0.4)
    assert update.event == "drag_end"
    assert update.dragging is False


def test_wave_detected_by_oscillation():
    h = HandHistory(capacity=60)
    d = MotionDetector()
    # Simulate 2 Hz oscillation with amplitude 0.1, for 1 second
    events = []
    for i in range(40):  # ~1.3 s at 30 fps
        t = i * 0.033
        x = 0.5 + 0.1 * math.sin(2 * math.pi * 2.0 * t)
        h.push(_hf(x=x, y=0.3), t)  # y < 0.5
        u = d.update(h, now=t)
        if u.event is not None:
            events.append(u.event)
    assert "wave" in events


def test_wave_not_detected_if_hand_low():
    h = HandHistory(capacity=60)
    d = MotionDetector()
    # Same oscillation but y = 0.8 (below mid-frame).
    events = []
    for i in range(40):
        t = i * 0.033
        x = 0.5 + 0.1 * math.sin(2 * math.pi * 2.0 * t)
        h.push(_hf(x=x, y=0.8), t)
        u = d.update(h, now=t)
        if u.event is not None:
            events.append(u.event)
    assert "wave" not in events


def test_drag_dx_dy_relative_to_start():
    h = HandHistory(capacity=60)
    d = MotionDetector()
    # Start drag at x=0.1, y=0.3. Use 0.05/frame so velocity is clearly above threshold
    # and enough distance accumulates after drag_start fires (~0.2 s arming period).
    x, y = 0.1, 0.3
    for i in range(20):
        x += 0.05  # 0.05/frame × ~30fps ≈ 1.5 units/sec, well above _DRAG_VELOCITY_ON
        h.push(_hf(x=x, y=y, pinch=0.95), i * 0.033)
        d.update(h, now=i * 0.033)
    # After 20 frames, hand is at x=0.1 + 20*0.05 = 1.1.
    # drag_start fires once arming (~0.2 s) passes; the remaining frames supply > 0.3 of dx.
    last = d.update(h, now=20 * 0.033)
    assert last.dragging is True
    assert last.drag_dx > 0.2  # moved meaningfully right after drag_start


def test_clap_detector_fires_once_on_impact():
    # Track distance between hands as they clap
    events = []
    det = bi_hand_clap_detector()
    distances = [0.4, 0.35, 0.3, 0.2, 0.1, 0.05, 0.12, 0.3, 0.4]  # clap, then release
    for i, d in enumerate(distances):
        if det.update(d, now=i * 0.033):
            events.append(i)
    # One impact event expected.
    assert len(events) == 1
