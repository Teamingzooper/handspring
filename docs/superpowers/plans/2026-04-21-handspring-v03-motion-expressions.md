# handspring v0.3.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. TDD. Checkbox progress.

**Goal:** Add temporal motion gestures (wave, pinch, expand, drag, clap), facial expressions, and three more hand shapes (`ok`, `rock`, `three`).

**Architecture:** New `history.py` (per-hand ring buffer of HandFeatures + timestamps). New `motion.py` (pure-function motion detectors reading history). New `expressions.py`. `gestures.py` extended. `tracker.py` owns the history buffers and drives motion/expression classification each frame. Types grow: `Expression`, `MotionEvent`, `MotionState`; `HandState.motion`, `FaceState.expression`, `FaceState.eye_{left,right}_open`, `FrameResult.clap_event`. OSC emitter adds pinching/dragging/drag_dx/drag_dy/event/clap/eye_open/expression addresses. Plan follows TDD throughout.

**Tech:** no new dependencies. `refine_landmarks=True` on FaceMesh (already in MediaPipe).

**Spec:** [`2026-04-21-handspring-v03-motion-expressions.md`](../specs/2026-04-21-handspring-v03-motion-expressions.md)

---

## Task 1: Types + Extension of HandState / FaceState / FrameResult

**Files:**
- Modify: `src/handspring/types.py`
- Modify: `tests/test_types.py`

### Step 1: Extend `tests/test_types.py`

Append:

```python
from handspring.types import Expression, MotionEvent, MotionState


def test_motion_state_default():
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    assert m.pinching is False
    assert m.event is None


def test_motion_state_with_event():
    m = MotionState(pinching=True, dragging=False, drag_dx=0.0, drag_dy=0.0, event="pinch")
    assert m.event == "pinch"


def test_hand_state_has_motion():
    from handspring.types import HandFeatures, HandState
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0)
    hs = HandState(present=True, features=hf, gesture="open", motion=m)
    assert hs.motion.pinching is False


def test_face_state_has_expression_and_eye_open():
    from handspring.types import FaceState
    fs = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    assert fs.expression == "neutral"
    assert fs.eye_left_open == 0.0


def test_frame_result_has_clap_event():
    from handspring.types import FaceState, FrameResult, HandState, PoseState
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    hs = HandState(present=False, features=None, gesture="none", motion=m)
    fs = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    ps = PoseState(present=False, joints=None)
    fr = FrameResult(left=hs, right=hs, face=fs, pose=ps, fps=30.0, clap_event=False)
    assert fr.clap_event is False
```

Also update every existing `HandState(...)`, `FaceState(...)`, `FrameResult(...)` constructor in this file to include the new fields. Specifically:
- All `HandState(...)` additions need `motion=<MotionState with all-False defaults>`.
- All `FaceState(...)` additions need `expression="neutral"`, `eye_left_open=0.0`, `eye_right_open=0.0`.
- All `FrameResult(...)` additions need `clap_event=False`.

### Step 2: Replace `src/handspring/types.py`

```python
"""Core dataclasses for frame-level tracking results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
Gesture = Literal[
    "fist", "open", "point", "peace", "thumbs_up", "ok", "rock", "three", "none"
]
Expression = Literal[
    "smile", "frown", "surprise", "wink_left", "wink_right", "neutral"
]
MotionEvent = Literal["wave", "pinch", "expand", "drag_start", "drag_end"]
Joint = Literal[
    "shoulder_left",
    "shoulder_right",
    "elbow_left",
    "elbow_right",
    "wrist_left",
    "wrist_right",
    "hip_left",
    "hip_right",
]


@dataclass(frozen=True)
class HandFeatures:
    x: float
    y: float
    z: float
    openness: float
    pinch: float


@dataclass(frozen=True)
class MotionState:
    """Per-hand motion state for a single frame.

    `event` is a one-shot: non-None only on the frame a motion event is
    detected. `pinching` and `dragging` are continuous state flags.
    `drag_dx` / `drag_dy` are measured from the frame where `drag_start`
    fired; they are 0.0 whenever `dragging` is False.
    """

    pinching: bool
    dragging: bool
    drag_dx: float
    drag_dy: float
    event: MotionEvent | None


@dataclass(frozen=True)
class HandState:
    present: bool
    features: HandFeatures | None
    gesture: Gesture
    motion: MotionState


@dataclass(frozen=True)
class FaceFeatures:
    yaw: float
    pitch: float
    mouth_open: float


@dataclass(frozen=True)
class FaceState:
    present: bool
    features: FaceFeatures | None
    expression: Expression
    eye_left_open: float
    eye_right_open: float


@dataclass(frozen=True)
class PoseLandmark:
    x: float
    y: float
    z: float
    visible: bool


@dataclass(frozen=True)
class PoseState:
    present: bool
    joints: dict[Joint, PoseLandmark] | None


@dataclass(frozen=True)
class FrameResult:
    left: HandState
    right: HandState
    face: FaceState
    pose: PoseState
    fps: float
    clap_event: bool
```

### Step 3: Run + verify

```bash
pytest tests/test_types.py -v
```

All types tests pass (existing + 5 new). Other test files will break because of the new required fields — that's expected, later tasks fix them.

```bash
ruff check src/handspring/types.py tests/test_types.py
ruff format --check src/handspring/types.py tests/test_types.py
mypy src/handspring/types.py
```

### Step 4: Commit

```bash
git add src/handspring/types.py tests/test_types.py
git commit -m "feat(types): add Expression, MotionEvent, MotionState; grow HandState/FaceState/FrameResult"
```

---

## Task 2: History ring buffer

**Files:**
- Create: `src/handspring/history.py`
- Create: `tests/test_history.py`

### Step 1: Failing tests

```python
"""History ring buffer tests."""
from __future__ import annotations

from handspring.history import HandHistory, HandSample
from handspring.types import HandFeatures


def _hf(x: float = 0.5, y: float = 0.5, pinch: float = 0.0) -> HandFeatures:
    return HandFeatures(x=x, y=y, z=0.0, openness=1.0, pinch=pinch)


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
```

### Step 2: Run; expect ImportError.

### Step 3: Implement `src/handspring/history.py`

```python
"""Ring buffer of per-hand HandFeatures samples with timestamps.

Provides enough state for motion detectors (wave, pinch, drag, clap)
to analyze the last ~1 second of hand motion without holding state
inside the classifiers themselves.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

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
        self._buf: Deque[HandSample] = deque(maxlen=capacity)

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
```

### Step 4: Run + lint + commit

```bash
pytest tests/test_history.py -v
ruff check .
ruff format --check .
mypy src/
```

```bash
git add src/handspring/history.py tests/test_history.py
git commit -m "feat(history): per-hand ring buffer of features+timestamps"
```

---

## Task 3: Motion detectors

**Files:**
- Create: `src/handspring/motion.py`
- Create: `tests/test_motion.py`

### Step 1: Tests

```python
"""Motion detector tests."""
from __future__ import annotations

import math

from handspring.history import HandHistory
from handspring.motion import MotionDetector, MotionUpdate, bi_hand_clap_detector
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
    d.update(h, now=0.0)  # enters pinched state
    h.push(_hf(pinch=0.3), 0.1)
    update = d.update(h, now=0.1)
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
    h = HandHistory(capacity=30)
    d = MotionDetector()
    # Start drag at x=0.3, y=0.3
    x, y = 0.3, 0.3
    for i in range(12):
        x += 0.03
        h.push(_hf(x=x, y=y, pinch=0.95), i * 0.033)
        d.update(h, now=i * 0.033)
    # At end, hand is at x=0.3 + 12*0.03 = 0.66
    last = d.update(h, now=12 * 0.033)
    # drag_dx should be end - start. start was first sample at x=0.33 (after first +0.03).
    # Some tolerance.
    assert last.dragging is True
    assert last.drag_dx > 0.2  # moved meaningfully right


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
```

### Step 2: Implement `src/handspring/motion.py`

```python
"""Pure-function motion detectors reading from HandHistory buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from handspring.types import MotionEvent

if TYPE_CHECKING:
    from handspring.history import HandHistory


# ---- Tuning thresholds (documented in spec) ----

_PINCH_ON = 0.85
_PINCH_OFF = 0.4
_PINCH_COOLDOWN = 0.3      # seconds
_DRAG_VELOCITY_ON = 0.1    # units/second (normalized frame units)
_DRAG_VELOCITY_OFF = 0.03
_DRAG_ARM_DURATION = 0.2   # seconds of sustained motion while pinching to enter drag
_DRAG_IDLE_DURATION = 0.3  # seconds of low velocity → exit drag
_WAVE_AMPLITUDE = 0.05
_WAVE_MIN_FREQ = 1.5
_WAVE_MAX_FREQ = 4.0
_WAVE_MIN_DURATION = 0.8
_WAVE_Y_MAX = 0.5          # wave only if hand.y < this (above mid-frame)
_WAVE_COOLDOWN = 1.0
_CLAP_NEAR = 0.08
_CLAP_FAR = 0.25
_CLAP_WINDOW = 0.3
_CLAP_COOLDOWN = 0.4


@dataclass(frozen=True)
class MotionUpdate:
    pinching: bool
    dragging: bool
    drag_dx: float
    drag_dy: float
    event: MotionEvent | None


class MotionDetector:
    """Stateful per-hand motion detector.

    Reads features from the caller's HandHistory and emits one-shot
    MotionEvents plus continuous pinching/dragging state. State is
    minimal (just recent transitions and cooldowns); the HandHistory
    carries the bulk of the time-series.
    """

    def __init__(self) -> None:
        self._was_pinching = False
        self._pinch_cooldown_until = -1.0
        self._wave_cooldown_until = -1.0

        self._dragging = False
        self._drag_start_xy: tuple[float, float] | None = None
        self._arming_drag_since: float | None = None
        self._idle_since: float | None = None

    def update(self, history: HandHistory, now: float) -> MotionUpdate:
        latest = history.latest()
        if latest is None:
            self._was_pinching = False
            self._reset_drag()
            return MotionUpdate(False, False, 0.0, 0.0, None)

        pinching = latest.features.pinch >= _PINCH_ON or (
            self._was_pinching and latest.features.pinch > _PINCH_OFF
        )
        event: MotionEvent | None = None

        # --- Pinch rising edge ---
        if pinching and not self._was_pinching and now >= self._pinch_cooldown_until:
            event = "pinch"
            self._pinch_cooldown_until = now + _PINCH_COOLDOWN

        # --- Expand falling edge ---
        if not pinching and self._was_pinching and now >= self._pinch_cooldown_until:
            event = "expand"
            self._pinch_cooldown_until = now + _PINCH_COOLDOWN
            # Ending pinch always ends drag too.
            if self._dragging:
                event = "drag_end"
                self._reset_drag()
            else:
                self._reset_drag()

        self._was_pinching = pinching

        # --- Drag logic (only while pinching) ---
        drag_dx = 0.0
        drag_dy = 0.0
        if pinching:
            velocity = _recent_velocity(history, window=0.1, now=now)
            if not self._dragging:
                if velocity >= _DRAG_VELOCITY_ON:
                    if self._arming_drag_since is None:
                        self._arming_drag_since = now
                    elif now - self._arming_drag_since >= _DRAG_ARM_DURATION:
                        # Commit to drag.
                        self._dragging = True
                        self._drag_start_xy = (latest.features.x, latest.features.y)
                        self._arming_drag_since = None
                        self._idle_since = None
                        if event is None:
                            event = "drag_start"
                else:
                    self._arming_drag_since = None
            else:
                # Currently dragging — update delta + watch for idle.
                if self._drag_start_xy is not None:
                    drag_dx = latest.features.x - self._drag_start_xy[0]
                    drag_dy = latest.features.y - self._drag_start_xy[1]
                if velocity < _DRAG_VELOCITY_OFF:
                    if self._idle_since is None:
                        self._idle_since = now
                    elif now - self._idle_since >= _DRAG_IDLE_DURATION:
                        if event is None:
                            event = "drag_end"
                        self._reset_drag()
                else:
                    self._idle_since = None
        else:
            self._arming_drag_since = None

        # --- Wave detection (only if no event claimed yet) ---
        if event is None and now >= self._wave_cooldown_until:
            if _detect_wave(history, now=now):
                event = "wave"
                self._wave_cooldown_until = now + _WAVE_COOLDOWN

        return MotionUpdate(
            pinching=pinching,
            dragging=self._dragging,
            drag_dx=drag_dx,
            drag_dy=drag_dy,
            event=event,
        )

    def _reset_drag(self) -> None:
        self._dragging = False
        self._drag_start_xy = None
        self._arming_drag_since = None
        self._idle_since = None


def _recent_velocity(history: "HandHistory", window: float, now: float) -> float:
    """Average speed (units/sec) over samples in the last `window` seconds."""
    recent = history.samples_since(now - window)
    if len(recent) < 2:
        return 0.0
    dx = recent[-1].features.x - recent[0].features.x
    dy = recent[-1].features.y - recent[0].features.y
    dt = recent[-1].timestamp - recent[0].timestamp
    if dt <= 0:
        return 0.0
    return math.hypot(dx, dy) / dt


def _detect_wave(history: "HandHistory", now: float) -> bool:
    """Detect a periodic horizontal oscillation in the last ~1s window."""
    window_start = now - 1.0
    samples = history.samples_since(window_start)
    if len(samples) < 10:
        return False
    # Require hand above mid-frame (avoid resting-hand-shake false positives).
    if any(s.features.y > _WAVE_Y_MAX for s in samples):
        return False
    xs = [s.features.x for s in samples]
    min_x = min(xs)
    max_x = max(xs)
    amplitude = (max_x - min_x) / 2.0
    if amplitude < _WAVE_AMPLITUDE:
        return False
    # Count zero-crossings around the mean to estimate frequency.
    mean = sum(xs) / len(xs)
    crossings = 0
    prev_above = xs[0] > mean
    for x in xs[1:]:
        above = x > mean
        if above != prev_above:
            crossings += 1
            prev_above = above
    # Two crossings per cycle.
    duration = samples[-1].timestamp - samples[0].timestamp
    if duration < _WAVE_MIN_DURATION:
        return False
    freq = crossings / 2.0 / duration if duration > 0 else 0.0
    return _WAVE_MIN_FREQ <= freq <= _WAVE_MAX_FREQ


class _ClapDetector:
    """Detects hand-clap impacts from a running stream of hand-to-hand distances."""

    def __init__(self) -> None:
        self._history: list[tuple[float, float]] = []  # (timestamp, distance)
        self._cooldown_until = -1.0

    def update(self, distance: float, now: float) -> bool:
        # Keep only last `_CLAP_WINDOW + 0.1` seconds.
        cutoff = now - (_CLAP_WINDOW + 0.1)
        self._history = [(t, d) for (t, d) in self._history if t >= cutoff]
        self._history.append((now, distance))

        if now < self._cooldown_until:
            return False
        if distance >= _CLAP_NEAR:
            return False
        # distance < near; check if recently we were >= far.
        recent = [d for (t, d) in self._history if t >= now - _CLAP_WINDOW]
        if any(d >= _CLAP_FAR for d in recent):
            self._cooldown_until = now + _CLAP_COOLDOWN
            return True
        return False


def bi_hand_clap_detector() -> _ClapDetector:
    """Factory — keeps the class internal but returns a reusable instance."""
    return _ClapDetector()
```

### Step 3: Run + lint + commit

```bash
pytest tests/test_motion.py -v
ruff check src/handspring/motion.py tests/test_motion.py
ruff format --check .
mypy src/
```

Fixture tweaks may be needed if thresholds narrowly miss. Adjust fixture values, not production thresholds, unless the threshold is wrong.

```bash
git add src/handspring/motion.py tests/test_motion.py
git commit -m "feat(motion): per-hand motion detector + clap detector"
```

---

## Task 4: Extended hand-shape classifier

**Files:**
- Modify: `src/handspring/gestures.py`
- Modify: `src/handspring/types.py` (already done in Task 1 — `Gesture` includes new values)
- Modify: `tests/test_gestures.py`
- Modify: `tests/fixtures.py`

### Step 1: Add fixtures

Append to `tests/fixtures.py`:

```python
def hand_ok() -> NDArray[np.float32]:
    """Thumb tip meets index tip; middle/ring/pinky extended."""
    lm = _hand_skeleton((True, True, True, True, True))
    # Move thumb tip and index tip to a shared point between them.
    meeting_point = (lm[4] + lm[8]) * 0.5
    lm[4] = meeting_point
    lm[8] = meeting_point
    return lm


def hand_rock() -> NDArray[np.float32]:
    """Index + pinky extended; middle + ring curled."""
    return _hand_skeleton((False, True, False, False, True))


def hand_three() -> NDArray[np.float32]:
    """Index + middle + ring extended; pinky curled; thumb ignored."""
    return _hand_skeleton((False, True, True, True, False))
```

### Step 2: Add tests

Append to `tests/test_gestures.py`:

```python
from tests.fixtures import hand_ok, hand_rock, hand_three


def test_ok_classifies_ok():
    assert classify_hand(hand_ok()) == "ok"


def test_rock_classifies_rock():
    assert classify_hand(hand_rock()) == "rock"


def test_three_classifies_three():
    assert classify_hand(hand_three()) == "three"
```

### Step 3: Update `src/handspring/gestures.py`

Add new classifier branches. Crucial: priority order matters because `ok` and `open` both have index/middle/ring/pinky extended — `ok` must be checked before `open`.

Extend `classify_hand` near the end:

```python
def classify_hand(landmarks: NDArray[np.floating[Any]]) -> Gesture:
    if not np.all(np.isfinite(landmarks)):
        raise ValueError("landmarks contain NaN or Inf")
    if landmarks.shape != (21, 3):
        raise ValueError(f"expected (21, 3), got {landmarks.shape}")

    thumb = _thumb_extended(landmarks)
    index = _finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
    middle = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring = _finger_extended(landmarks, RING_TIP, RING_PIP)
    pinky = _finger_extended(landmarks, PINKY_TIP, PINKY_PIP)

    extended_count = sum([thumb, index, middle, ring, pinky])

    # thumbs_up: thumb extended upward, four other fingers curled.
    if thumb and not index and not middle and not ring and not pinky and _thumb_up(landmarks):
        return "thumbs_up"

    # ok: thumb + index tips touching; middle/ring/pinky extended.
    if _thumb_index_touching(landmarks) and middle and ring and pinky:
        return "ok"

    # open: all five extended.
    if extended_count == 5:
        return "open"

    # fist: all four non-thumb curled.
    if not index and not middle and not ring and not pinky:
        return "fist"

    # peace: index + middle, nothing else (ring + pinky curled).
    if index and middle and not ring and not pinky:
        return "peace"

    # rock: index + pinky extended, middle + ring curled.
    if index and not middle and not ring and pinky:
        return "rock"

    # three: index + middle + ring extended; pinky curled.
    if index and middle and ring and not pinky:
        return "three"

    # point: index only, thumb state ignored.
    if index and not middle and not ring and not pinky:
        return "point"

    return "none"
```

Add a new helper:

```python
def _thumb_index_touching(landmarks: NDArray[np.floating[Any]]) -> bool:
    """Distance between thumb tip and index tip < 0.2 × palm width."""
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    palm_width = float(np.linalg.norm(pinky_mcp - index_mcp))
    if palm_width < 1e-6:
        return False
    thumb_index_dist = float(np.linalg.norm(thumb_tip - index_tip))
    return thumb_index_dist < palm_width * 0.2
```

### Step 4: Run + commit

```bash
pytest tests/test_gestures.py -v
ruff check src/handspring/gestures.py tests/test_gestures.py tests/fixtures.py
ruff format --check .
mypy src/
```

```bash
git add src/handspring/gestures.py tests/test_gestures.py tests/fixtures.py
git commit -m "feat(gestures): add ok, rock, three to hand vocabulary"
```

---

## Task 5: Facial expression classifier

**Files:**
- Create: `src/handspring/expressions.py`
- Create: `tests/test_expressions.py`
- Modify: `tests/fixtures.py` (add expression fixtures)

### Step 1: Add fixtures

Append to `tests/fixtures.py`. Also need more landmark indices — add them:

```python
# Additional face mesh indices used by the expression classifier.
FACE_UPPER_LIP_LEFT = 14  # using existing for simplicity — will refine in test data
FACE_LEFT_EYE_UPPER = 159
FACE_LEFT_EYE_LOWER = 145
FACE_LEFT_EYE_LEFT_CORNER = 33
FACE_LEFT_EYE_RIGHT_CORNER = 133
FACE_RIGHT_EYE_UPPER = 386
FACE_RIGHT_EYE_LOWER = 374
FACE_RIGHT_EYE_LEFT_CORNER = 362
FACE_RIGHT_EYE_RIGHT_CORNER = 263


def _face_neutral_refined() -> NDArray[np.float32]:
    """Face fixture populated with eye landmarks as well as mouth."""
    lm = _face_base()
    # Eye landmarks — assume eyes are open (lid gap ~= 0.025 relative to eye width 0.08)
    # Left eye (user's left): left corner 33, right corner 133, upper 159, lower 145
    lm[FACE_LEFT_EYE_LEFT_CORNER] = (0.33, 0.45, 0.0)
    lm[FACE_LEFT_EYE_RIGHT_CORNER] = (0.41, 0.45, 0.0)
    lm[FACE_LEFT_EYE_UPPER] = (0.37, 0.438, 0.0)
    lm[FACE_LEFT_EYE_LOWER] = (0.37, 0.462, 0.0)
    lm[FACE_RIGHT_EYE_LEFT_CORNER] = (0.59, 0.45, 0.0)
    lm[FACE_RIGHT_EYE_RIGHT_CORNER] = (0.67, 0.45, 0.0)
    lm[FACE_RIGHT_EYE_UPPER] = (0.63, 0.438, 0.0)
    lm[FACE_RIGHT_EYE_LOWER] = (0.63, 0.462, 0.0)
    return lm


def face_neutral() -> NDArray[np.float32]:
    return _face_neutral_refined()


def face_smile() -> NDArray[np.float32]:
    lm = _face_neutral_refined()
    # Mouth corners up (y smaller = higher in image space).
    lm[FACE_LEFT_MOUTH] = (0.4, 0.68, 0.0)   # was 0.70
    lm[FACE_RIGHT_MOUTH] = (0.6, 0.68, 0.0)
    return lm


def face_frown() -> NDArray[np.float32]:
    lm = _face_neutral_refined()
    lm[FACE_LEFT_MOUTH] = (0.4, 0.72, 0.0)   # corners down
    lm[FACE_RIGHT_MOUTH] = (0.6, 0.72, 0.0)
    return lm


def face_surprise() -> NDArray[np.float32]:
    lm = _face_neutral_refined()
    # Wide open mouth
    lm[FACE_UPPER_LIP] = (0.5, 0.67, 0.0)
    lm[FACE_LOWER_LIP] = (0.5, 0.75, 0.0)
    # Wider eyes — bigger lid gap
    lm[FACE_LEFT_EYE_UPPER] = (0.37, 0.43, 0.0)
    lm[FACE_LEFT_EYE_LOWER] = (0.37, 0.47, 0.0)
    lm[FACE_RIGHT_EYE_UPPER] = (0.63, 0.43, 0.0)
    lm[FACE_RIGHT_EYE_LOWER] = (0.63, 0.47, 0.0)
    return lm


def face_wink_left() -> NDArray[np.float32]:
    lm = _face_neutral_refined()
    # Left eye closed (upper and lower lid very close)
    lm[FACE_LEFT_EYE_UPPER] = (0.37, 0.449, 0.0)
    lm[FACE_LEFT_EYE_LOWER] = (0.37, 0.451, 0.0)
    return lm


def face_wink_right() -> NDArray[np.float32]:
    lm = _face_neutral_refined()
    lm[FACE_RIGHT_EYE_UPPER] = (0.63, 0.449, 0.0)
    lm[FACE_RIGHT_EYE_LOWER] = (0.63, 0.451, 0.0)
    return lm
```

### Step 2: Expression tests

Create `tests/test_expressions.py`:

```python
"""Expression classifier tests."""
from __future__ import annotations

from handspring.expressions import classify_expression, eye_open_values
from tests.fixtures import (
    face_frown,
    face_neutral,
    face_smile,
    face_surprise,
    face_wink_left,
    face_wink_right,
)


def test_neutral_classifies_neutral():
    assert classify_expression(face_neutral()) == "neutral"


def test_smile_classifies_smile():
    assert classify_expression(face_smile()) == "smile"


def test_frown_classifies_frown():
    assert classify_expression(face_frown()) == "frown"


def test_surprise_classifies_surprise():
    assert classify_expression(face_surprise()) == "surprise"


def test_wink_left_classifies_wink_left():
    assert classify_expression(face_wink_left()) == "wink_left"


def test_wink_right_classifies_wink_right():
    assert classify_expression(face_wink_right()) == "wink_right"


def test_eye_open_values_closed_low():
    left, right = eye_open_values(face_wink_left())
    assert left < 0.2
    assert right > 0.5


def test_eye_open_values_open_high():
    left, right = eye_open_values(face_neutral())
    assert left > 0.3
    assert right > 0.3
```

### Step 3: Implement `src/handspring/expressions.py`

```python
"""Classify facial expressions from MediaPipe FaceMesh landmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from handspring.features import (
    F_LEFT_EYE,
    F_LEFT_MOUTH,
    F_LOWER_LIP,
    F_NOSE_TIP,
    F_RIGHT_EYE,
    F_RIGHT_MOUTH,
    F_UPPER_LIP,
)
from handspring.types import Expression

# Additional FaceMesh indices used for eye-open computation.
# With refine_landmarks=True these indices give upper/lower lids + eye corners.
_LEFT_EYE_UPPER = 159
_LEFT_EYE_LOWER = 145
_LEFT_EYE_LEFT_CORNER = 33
_LEFT_EYE_RIGHT_CORNER = 133
_RIGHT_EYE_UPPER = 386
_RIGHT_EYE_LOWER = 374
_RIGHT_EYE_LEFT_CORNER = 362
_RIGHT_EYE_RIGHT_CORNER = 263

_EYE_OPEN_MIN_RAW = 0.02   # eye width ratio at closed
_EYE_OPEN_MAX_RAW = 0.32   # eye width ratio at wide open


def eye_open_values(landmarks: NDArray[np.floating[Any]]) -> tuple[float, float]:
    """Return (left_eye_open, right_eye_open), each in [0, 1].

    Computed as lid-gap / eye-width, clamped to the expected dynamic range
    and normalized so 0 ≈ fully closed, 1 ≈ fully open.
    """
    left = _eye_open_one(
        landmarks,
        upper=_LEFT_EYE_UPPER,
        lower=_LEFT_EYE_LOWER,
        left_corner=_LEFT_EYE_LEFT_CORNER,
        right_corner=_LEFT_EYE_RIGHT_CORNER,
    )
    right = _eye_open_one(
        landmarks,
        upper=_RIGHT_EYE_UPPER,
        lower=_RIGHT_EYE_LOWER,
        left_corner=_RIGHT_EYE_LEFT_CORNER,
        right_corner=_RIGHT_EYE_RIGHT_CORNER,
    )
    return left, right


def classify_expression(landmarks: NDArray[np.floating[Any]]) -> Expression:
    """Classify a face into one of six expressions."""
    if not np.all(np.isfinite(landmarks)):
        raise ValueError("landmarks contain NaN or Inf")

    left_eye_open, right_eye_open = eye_open_values(landmarks)
    mouth_open = _mouth_open(landmarks)
    eye_distance = _eye_distance(landmarks)
    mouth_corner_delta = _mouth_corner_delta(landmarks, eye_distance)

    # surprise dominates: wide mouth + wide eyes.
    if mouth_open > 0.55 and left_eye_open > 0.6 and right_eye_open > 0.6:
        return "surprise"

    # wink: asymmetric eye closure. Prefer wink over smile/frown.
    if left_eye_open < 0.2 and right_eye_open > 0.6:
        return "wink_left"
    if right_eye_open < 0.2 and left_eye_open > 0.6:
        return "wink_right"

    # smile / frown: corners above / below the mid-lip baseline.
    if mouth_corner_delta > 0.02:
        return "smile"
    if mouth_corner_delta < -0.015:
        return "frown"

    return "neutral"


# ---- helpers ----


def _eye_open_one(
    landmarks: NDArray[np.floating[Any]],
    *,
    upper: int,
    lower: int,
    left_corner: int,
    right_corner: int,
) -> float:
    up = landmarks[upper]
    lo = landmarks[lower]
    l_corner = landmarks[left_corner]
    r_corner = landmarks[right_corner]
    vertical = float(abs(lo[1] - up[1]))
    width = float(abs(r_corner[0] - l_corner[0]))
    if width < 1e-6:
        return 0.0
    raw = vertical / width
    norm = (raw - _EYE_OPEN_MIN_RAW) / (_EYE_OPEN_MAX_RAW - _EYE_OPEN_MIN_RAW)
    return float(np.clip(norm, 0.0, 1.0))


def _mouth_open(landmarks: NDArray[np.floating[Any]]) -> float:
    """Same formula as features.face_features; duplicated here to avoid
    cross-module mutual dependency."""
    upper = landmarks[F_UPPER_LIP]
    lower = landmarks[F_LOWER_LIP]
    left_mouth = landmarks[F_LEFT_MOUTH]
    right_mouth = landmarks[F_RIGHT_MOUTH]
    vertical = float(abs(lower[1] - upper[1]))
    mouth_width = float(abs(right_mouth[0] - left_mouth[0]))
    if mouth_width < 1e-6:
        mouth_width = 1e-6
    return float(np.clip((vertical / mouth_width) * 2.2, 0.0, 1.0))


def _eye_distance(landmarks: NDArray[np.floating[Any]]) -> float:
    left_eye = landmarks[F_LEFT_EYE]
    right_eye = landmarks[F_RIGHT_EYE]
    dist = float(abs(right_eye[0] - left_eye[0]))
    return max(dist, 1e-6)


def _mouth_corner_delta(
    landmarks: NDArray[np.floating[Any]], eye_distance: float
) -> float:
    """Average vertical offset of mouth corners above the lip midline,
    normalized by eye distance. Positive = corners up (smile), negative
    = corners down (frown)."""
    left_corner = landmarks[F_LEFT_MOUTH]
    right_corner = landmarks[F_RIGHT_MOUTH]
    upper_lip = landmarks[F_UPPER_LIP]
    lower_lip = landmarks[F_LOWER_LIP]
    mid_lip_y = (upper_lip[1] + lower_lip[1]) / 2.0
    # Below midline = larger y = corners low (frown).
    avg_corner_y = (left_corner[1] + right_corner[1]) / 2.0
    # Corners up = avg_corner_y SMALLER than mid_lip_y = negative delta.
    # Invert so "up" is positive.
    return float((mid_lip_y - avg_corner_y) / eye_distance)
```

### Step 4: Run + commit

```bash
pytest tests/test_expressions.py -v
ruff check src/handspring/expressions.py tests/test_expressions.py tests/fixtures.py
ruff format --check .
mypy src/
```

If fixtures don't trigger expected classifications, adjust fixture values (not thresholds).

```bash
git add src/handspring/expressions.py tests/test_expressions.py tests/fixtures.py
git commit -m "feat(expressions): face classifier (smile/frown/surprise/wink) + eye-open"
```

---

## Task 6: Tracker — refine landmarks, per-hand history, motion, expressions

**Files:**
- Modify: `src/handspring/tracker.py`

Changes:
1. Enable `refine_landmarks=True` on FaceMesh.
2. Create per-hand `HandHistory(capacity=30)` and `MotionDetector` instances.
3. Create bi-hand `ClapDetector`.
4. After deriving per-hand features, push to history and call `motion_detector.update(history, now)` to produce `MotionState`.
5. Assemble `HandState` with `motion=...`.
6. Call `classify_expression` + `eye_open_values` on face landmarks, populate `FaceState`.
7. Compute `clap_event = clap_detector.update(distance_between_hands, now)`.
8. Build `FrameResult` including `clap_event`.

Full rewrite of `src/handspring/tracker.py`:

```python
"""MediaPipe wrapper: accepts BGR frames, returns FrameResult."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from handspring.expressions import classify_expression, eye_open_values
from handspring.features import face_features, hand_features
from handspring.gestures import classify_hand
from handspring.history import HandHistory
from handspring.motion import MotionDetector, bi_hand_clap_detector
from handspring.types import (
    FaceState,
    FrameResult,
    HandState,
    Joint,
    MotionState,
    PoseLandmark,
    PoseState,
    Side,
)

_POSE_JOINTS: dict[Joint, int] = {
    "shoulder_left": 12,
    "shoulder_right": 11,
    "elbow_left": 14,
    "elbow_right": 13,
    "wrist_left": 16,
    "wrist_right": 15,
    "hip_left": 24,
    "hip_right": 23,
}

_VISIBILITY_THRESHOLD = 0.5
_HISTORY_CAPACITY = 30


@dataclass
class TrackerConfig:
    max_hands: int = 2
    track_face: bool = True
    track_pose: bool = True
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


class Tracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self._config.max_hands,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        if self._config.track_face:
            self._face_mesh: Any = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # NEW: needed for accurate eye/lip landmarks
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
        else:
            self._face_mesh = None

        if self._config.track_pose:
            self._pose: Any = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
        else:
            self._pose = None

        self._left_history = HandHistory(capacity=_HISTORY_CAPACITY)
        self._right_history = HandHistory(capacity=_HISTORY_CAPACITY)
        self._left_motion = MotionDetector()
        self._right_motion = MotionDetector()
        self._clap_detector = bi_hand_clap_detector()

        self._start_time = time.perf_counter()
        self._last_frame_time: float | None = None
        self._fps_ema: float = 0.0

    def process(self, bgr_frame: NDArray[np.uint8]) -> FrameResult:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_results = self._hands.process(rgb)
        face_result: Any = self._face_mesh.process(rgb) if self._face_mesh is not None else None
        pose_result: Any = self._pose.process(rgb) if self._pose is not None else None

        now = time.perf_counter() - self._start_time

        left_state, right_state = self._hand_states(hand_results, now)
        face_state = self._face_state(face_result)
        pose_state = self._pose_state(pose_result)

        clap_event = False
        if left_state.present and right_state.present and left_state.features and right_state.features:
            dx = left_state.features.x - right_state.features.x
            dy = left_state.features.y - right_state.features.y
            distance = math.hypot(dx, dy)
            clap_event = self._clap_detector.update(distance, now)

        # FPS EMA
        fps = 0.0
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                instant = 1.0 / dt
                self._fps_ema = 0.9 * self._fps_ema + 0.1 * instant if self._fps_ema else instant
                fps = self._fps_ema
        self._last_frame_time = now

        return FrameResult(
            left=left_state,
            right=right_state,
            face=face_state,
            pose=pose_state,
            fps=fps,
            clap_event=clap_event,
        )

    def close(self) -> None:
        self._hands.close()
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._pose is not None:
            self._pose.close()

    def _hand_states(self, hand_results: Any, now: float) -> tuple[HandState, HandState]:
        absent_motion = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
        absent = HandState(present=False, features=None, gesture="none", motion=absent_motion)

        left = absent
        right = absent
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for landmarks, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
                strict=False,
            ):
                label: str = handedness.classification[0].label
                side: Side = "right" if label == "Left" else "left"
                arr = _landmark_list_to_array(landmarks)
                feats = hand_features(arr)
                gesture = classify_hand(arr)

                history = self._left_history if side == "left" else self._right_history
                detector = self._left_motion if side == "left" else self._right_motion
                history.push(feats, now)
                m = detector.update(history, now)
                motion = MotionState(
                    pinching=m.pinching,
                    dragging=m.dragging,
                    drag_dx=m.drag_dx,
                    drag_dy=m.drag_dy,
                    event=m.event,
                )
                state = HandState(present=True, features=feats, gesture=gesture, motion=motion)
                if side == "left":
                    left = state
                else:
                    right = state

        # Tick motion detectors even when a hand is absent so state doesn't stale.
        if not left.present:
            self._left_motion.update(self._left_history, now)
        if not right.present:
            self._right_motion.update(self._right_history, now)

        return left, right

    def _face_state(self, face_result: Any) -> FaceState:
        if face_result is None or not face_result.multi_face_landmarks:
            return FaceState(
                present=False,
                features=None,
                expression="neutral",
                eye_left_open=0.0,
                eye_right_open=0.0,
            )
        lm = face_result.multi_face_landmarks[0]
        arr = _landmark_list_to_array(lm)
        feats = face_features(arr)
        expression = classify_expression(arr)
        left_open, right_open = eye_open_values(arr)
        return FaceState(
            present=True,
            features=feats,
            expression=expression,
            eye_left_open=left_open,
            eye_right_open=right_open,
        )

    def _pose_state(self, pose_result: Any) -> PoseState:
        if pose_result is None or pose_result.pose_landmarks is None:
            return PoseState(present=False, joints=None)
        landmarks = pose_result.pose_landmarks.landmark
        joints: dict[Joint, PoseLandmark] = {}
        for joint_name, mp_idx in _POSE_JOINTS.items():
            lm = landmarks[mp_idx]
            joints[joint_name] = PoseLandmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visible=float(lm.visibility) >= _VISIBILITY_THRESHOLD,
            )
        return PoseState(present=True, joints=joints)


def _landmark_list_to_array(landmark_list: Any) -> NDArray[np.float32]:
    count = len(landmark_list.landmark)
    arr = np.zeros((count, 3), dtype=np.float32)
    for i, lm in enumerate(landmark_list.landmark):
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
    return arr
```

### Verify + commit

```bash
ruff check src/handspring/tracker.py
ruff format --check src/handspring/tracker.py
mypy src/
python -c "from handspring.tracker import Tracker, TrackerConfig; t = Tracker(); t.close(); print('ok')"
pytest tests/test_types.py tests/test_features.py tests/test_gestures.py tests/test_expressions.py tests/test_motion.py tests/test_history.py -v
```

```bash
git add src/handspring/tracker.py
git commit -m "feat(tracker): refine_landmarks, per-hand history, motion + expressions"
```

---

## Task 7: OSC emitter — motion + expressions

**Files:**
- Modify: `src/handspring/osc_out.py`
- Modify: `tests/test_osc_out.py`

### Step 1: Extend `tests/test_osc_out.py`

First fix all existing `FrameResult(...)`, `HandState(...)`, `FaceState(...)` constructors to include new fields (`motion=MotionState(False, False, 0.0, 0.0, None)`, `expression="neutral"`, `eye_left_open=0.0`, `eye_right_open=0.0`, `clap_event=False`). Add imports:

```python
from handspring.types import MotionState
```

Then append new tests:

```python
def test_motion_continuous_state_emitted():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0)
    m = MotionState(pinching=True, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    left = HandState(present=True, features=hf, gesture="none", motion=m)
    right = HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    face = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    pose = PoseState(present=False, joints=None)
    emitter.emit(
        FrameResult(
            left=left, right=right, face=face, pose=pose, fps=30.0, clap_event=False
        )
    )
    assert ("/hand/left/pinching", 1) in fake.sent
    assert ("/hand/left/dragging", 0) in fake.sent


def test_motion_event_fires():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0)
    m = MotionState(pinching=True, dragging=False, drag_dx=0.0, drag_dy=0.0, event="pinch")
    left = HandState(present=True, features=hf, gesture="none", motion=m)
    right = HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    face = FaceState(
        present=False, features=None, expression="neutral",
        eye_left_open=0.0, eye_right_open=0.0,
    )
    pose = PoseState(present=False, joints=None)
    emitter.emit(FrameResult(left, right, face, pose, 30.0, False))
    assert ("/hand/left/event", "pinch") in fake.sent


def test_drag_dxdy_only_when_dragging():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0)
    m = MotionState(pinching=True, dragging=True, drag_dx=0.25, drag_dy=-0.1, event=None)
    left = HandState(present=True, features=hf, gesture="none", motion=m)
    right = HandState(
        present=False, features=None, gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    face = FaceState(
        present=False, features=None, expression="neutral",
        eye_left_open=0.0, eye_right_open=0.0,
    )
    emitter.emit(FrameResult(left, right, face, PoseState(False, None), 30.0, False))
    assert ("/hand/left/drag_dx", 0.25) in fake.sent
    assert ("/hand/left/drag_dy", -0.1) in fake.sent


def test_clap_event_emits():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    # Build a frame with clap_event=True
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0)
    m = MotionState(False, False, 0.0, 0.0, None)
    left = HandState(present=True, features=hf, gesture="none", motion=m)
    right = HandState(present=True, features=hf, gesture="none", motion=m)
    face = FaceState(
        present=False, features=None, expression="neutral",
        eye_left_open=0.0, eye_right_open=0.0,
    )
    emitter.emit(FrameResult(left, right, face, PoseState(False, None), 30.0, True))
    assert ("/motion/clap", 1) in fake.sent


def test_face_expression_event_on_change():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)

    def frame(expr):
        return FrameResult(
            left=HandState(
                present=False, features=None, gesture="none",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
            right=HandState(
                present=False, features=None, gesture="none",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
            face=FaceState(
                present=True,
                features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=0.0),
                expression=expr,
                eye_left_open=0.8,
                eye_right_open=0.8,
            ),
            pose=PoseState(False, None),
            fps=30.0,
            clap_event=False,
        )

    emitter.emit(frame("smile"))
    emitter.emit(frame("smile"))
    emitter.emit(frame("frown"))
    exprs = [v for a, v in fake.sent if a == "/face/expression"]
    assert exprs == ["smile", "frown"]


def test_face_eye_open_continuous():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    face = FaceState(
        present=True,
        features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=0.0),
        expression="neutral",
        eye_left_open=0.75,
        eye_right_open=0.92,
    )
    fr = FrameResult(
        left=HandState(False, None, "none", MotionState(False, False, 0.0, 0.0, None)),
        right=HandState(False, None, "none", MotionState(False, False, 0.0, 0.0, None)),
        face=face,
        pose=PoseState(False, None),
        fps=30.0,
        clap_event=False,
    )
    emitter.emit(fr)
    assert ("/face/eye_left_open", 0.75) in fake.sent
    assert ("/face/eye_right_open", 0.92) in fake.sent
```

### Step 2: Implement additions in `src/handspring/osc_out.py`

Add `_emit_motion` to `_emit_hand`, add expression state tracking, add `/motion/clap`, add eye_open emission.

Key changes inside `_emit_hand`:

```python
# After emitting features, before gesture:
m = state.motion
self._client.send_message(f"/hand/{side}/pinching", 1 if m.pinching else 0)
self._client.send_message(f"/hand/{side}/dragging", 1 if m.dragging else 0)
if m.dragging:
    self._client.send_message(f"/hand/{side}/drag_dx", float(m.drag_dx))
    self._client.send_message(f"/hand/{side}/drag_dy", float(m.drag_dy))
if m.event is not None:
    self._client.send_message(f"/hand/{side}/event", m.event)
```

Inside `_emit_face`, add expression + eye_open emission:

```python
# Eye openness (continuous)
self._client.send_message("/face/eye_left_open", float(state.eye_left_open))
self._client.send_message("/face/eye_right_open", float(state.eye_right_open))
# Expression (state-change)
if state.expression != self._last_expression:
    self._client.send_message("/face/expression", state.expression)
    self._last_expression = state.expression
```

Track `self._last_expression` as a field initialized to `"neutral"` (matches absent-face default).

Add clap emission to the top-level `emit()`:

```python
if frame.clap_event:
    self._client.send_message("/motion/clap", 1)
```

Full relevant shape of the updated `osc_out.py`:

```python
"""Emit per-frame tracking results as OSC messages over UDP."""
from __future__ import annotations

from typing import Any, Protocol

from handspring.types import (
    Expression,
    FaceState,
    FrameResult,
    Gesture,
    HandState,
    PoseState,
    Side,
)


class _SendsOsc(Protocol):
    def send_message(self, address: str, value: Any) -> None: ...


def _make_client(host: str, port: int) -> _SendsOsc:
    from pythonosc.udp_client import SimpleUDPClient
    return SimpleUDPClient(host, port)


class OscEmitter:
    def __init__(
        self,
        *,
        client: _SendsOsc | None = None,
        host: str = "127.0.0.1",
        port: int = 9000,
    ) -> None:
        self._client: _SendsOsc = client if client is not None else _make_client(host, port)
        self._last_gesture: dict[Side, Gesture] = {"left": "none", "right": "none"}
        self._last_expression: Expression = "neutral"

    def emit(self, frame: FrameResult) -> None:
        self._emit_hand("left", frame.left)
        self._emit_hand("right", frame.right)
        self._emit_face(frame.face)
        self._emit_pose(frame.pose)
        if frame.clap_event:
            self._client.send_message("/motion/clap", 1)

    def _emit_hand(self, side: Side, state: HandState) -> None:
        self._client.send_message(f"/hand/{side}/present", 1 if state.present else 0)
        if state.present and state.features is not None:
            f = state.features
            self._client.send_message(f"/hand/{side}/x", float(f.x))
            self._client.send_message(f"/hand/{side}/y", float(f.y))
            self._client.send_message(f"/hand/{side}/z", float(f.z))
            self._client.send_message(f"/hand/{side}/openness", float(f.openness))
            self._client.send_message(f"/hand/{side}/pinch", float(f.pinch))

        # Motion continuous state (always emitted).
        m = state.motion
        self._client.send_message(f"/hand/{side}/pinching", 1 if m.pinching else 0)
        self._client.send_message(f"/hand/{side}/dragging", 1 if m.dragging else 0)
        if m.dragging:
            self._client.send_message(f"/hand/{side}/drag_dx", float(m.drag_dx))
            self._client.send_message(f"/hand/{side}/drag_dy", float(m.drag_dy))
        if m.event is not None:
            self._client.send_message(f"/hand/{side}/event", m.event)

        # Static gesture (state-change only).
        current: Gesture = state.gesture
        if current != self._last_gesture[side]:
            self._client.send_message(f"/hand/{side}/gesture", current)
            self._last_gesture[side] = current

    def _emit_face(self, state: FaceState) -> None:
        self._client.send_message("/face/present", 1 if state.present else 0)
        if state.present and state.features is not None:
            f = state.features
            self._client.send_message("/face/yaw", float(f.yaw))
            self._client.send_message("/face/pitch", float(f.pitch))
            self._client.send_message("/face/mouth_open", float(f.mouth_open))

        # Eye openness (continuous; emits 0.0 when face absent, which matches default).
        self._client.send_message("/face/eye_left_open", float(state.eye_left_open))
        self._client.send_message("/face/eye_right_open", float(state.eye_right_open))

        # Expression (state-change).
        if state.expression != self._last_expression:
            self._client.send_message("/face/expression", state.expression)
            self._last_expression = state.expression

    def _emit_pose(self, state: PoseState) -> None:
        self._client.send_message("/pose/present", 1 if state.present else 0)
        if not state.present or state.joints is None:
            return
        for joint_name, lm in state.joints.items():
            self._client.send_message(f"/pose/{joint_name}/visible", 1 if lm.visible else 0)
            if not lm.visible:
                continue
            self._client.send_message(f"/pose/{joint_name}/x", float(lm.x))
            self._client.send_message(f"/pose/{joint_name}/y", float(lm.y))
            self._client.send_message(f"/pose/{joint_name}/z", float(lm.z))
```

### Step 3: Run + commit

```bash
pytest
ruff check . && ruff format --check . && mypy src/
```

All green.

```bash
git add src/handspring/osc_out.py tests/test_osc_out.py
git commit -m "feat(osc_out): emit motion state/events, expressions, eye-open, clap"
```

---

## Task 8: Wire into `__main__.py` + status line

**Files:**
- Modify: `src/handspring/__main__.py`
- Modify: `src/handspring/preview.py` (status line updated)

### Step 1: `_print_status` extended

Replace `_print_status` in `__main__.py`:

```python
def _print_status(result: FrameResult) -> None:
    left = result.left.gesture if result.left.present else "-"
    right = result.right.gesture if result.right.present else "-"
    face = result.face.expression if result.face.present else "-"
    clap = "CLAP" if result.clap_event else "    "
    print(
        f"\rFPS {result.fps:5.1f}  L:{left:<10} R:{right:<10} face:{face:<10} {clap}",
        end="",
        flush=True,
    )
```

### Step 2: preview status overlay adds expression line

Find `_draw_status` in `preview.py` and add:

```python
lines = [
    f"FPS: {frame_result.fps:5.1f}",
    f"OSC -> {osc_target}",
    f"Left:  {frame_result.left.gesture if frame_result.left.present else '-'}",
    f"Right: {frame_result.right.gesture if frame_result.right.present else '-'}",
    f"Pose:  {'on' if frame_result.pose.present else '-'}",
    f"Face:  {frame_result.face.expression if frame_result.face.present else '-'}",
]
```

### Step 3: Verify + commit

```bash
pytest
ruff check . && ruff format --check . && mypy src/
python -m handspring --version
python -m handspring --help
```

```bash
git add src/handspring/__main__.py src/handspring/preview.py
git commit -m "feat(main,preview): status lines show expression + clap"
```

---

## Task 9: README + tag v0.3.0

**Files:**
- Modify: `README.md`
- Modify: `src/handspring/__init__.py` (bump __version__)
- Modify: `pyproject.toml` (bump version)

### Step 1: Bump version

`__init__.py`:
```python
__version__ = "0.3.0"
```

`pyproject.toml`:
```toml
version = "0.3.0"
```

### Step 2: README additions

In the **OSC reference** section, add after the existing hand/face/pose tables:

```markdown

Motion state per hand (continuous per frame):

| Address | Type | Notes |
|---|---|---|
| `/hand/<side>/pinching` | int | 0 or 1 |
| `/hand/<side>/dragging` | int | 0 or 1 |
| `/hand/<side>/drag_dx` | float | x offset from drag-start, only when `dragging=1` |
| `/hand/<side>/drag_dy` | float | y offset |

Motion events (one-shot per frame):

| Address | Type | Values |
|---|---|---|
| `/hand/<side>/event` | string | `wave` \| `pinch` \| `expand` \| `drag_start` \| `drag_end` |
| `/motion/clap` | int | `1` on each clap impact |

Face (continuous + state-change):

| Address | Type | Notes |
|---|---|---|
| `/face/eye_left_open` | float | 0..1 |
| `/face/eye_right_open` | float | 0..1 |
| `/face/expression` | string | `smile` \| `frown` \| `surprise` \| `wink_left` \| `wink_right` \| `neutral` — emitted only on change |
```

In the **hand gesture** table, update the list to include the new three:

```
`/hand/<side>/gesture` | string | `fist` \| `open` \| `point` \| `peace` \| `thumbs_up` \| `ok` \| `rock` \| `three` \| `none`
```

### Step 3: Full checks + commit + tag

```bash
ruff check .
ruff format --check .
mypy src/
pytest
python -m handspring --version   # should print 0.3.0
```

All green. Then:

```bash
git add README.md src/handspring/__init__.py pyproject.toml
git commit -m "docs: document motion events, expressions, eye-open, new shapes"
git tag -a v0.3.0 -m "handspring v0.3.0 — motion gestures + expressions + more shapes"
```

### Step 4: Summary

Report:
- Commits since v0.2.1
- Total test count
- Any fixture tweaks
- Any threshold adjustments

---

## Execution notes

- Sequential only.
- Every task ends with full `pytest`, `ruff check`, `ruff format --check`, `mypy src/` clean.
- If a threshold in `motion.py` or `expressions.py` narrowly misses the test fixture, tweak the **fixture**, not the threshold, unless the threshold was wrong for real-world reasons.
- `refine_landmarks=True` changes some FaceMesh landmark indices. The indices we use (1, 33, 263, 13, 14, 61, 291, plus the new eye ones) are stable across refine on/off per MediaPipe docs. If a test fails with real landmarks, check `refine_landmarks` docs.
