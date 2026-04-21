# handspring v0.5.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. TDD. Checkbox progress.

**Goal:** Build JARVIS mode — a mouth-toggled second mode that hosts transparent-window creation, grab-to-drag, and point-to-tap interactions.

**Architecture:** Two new modules: `app_mode.py` (mode state machine + mouth debouncer), `jarvis.py` (Window, WindowManager, gesture state machines for create/grab/tap, renderer). Add `index_x`/`index_y` to HandFeatures so tap can use the fingertip. Wire into `__main__`; preview gets a mode badge; synth muted when in Jarvis. OSC emits app mode + window events.

**Tech:** No new dependencies. All additions are in pure Python using existing numpy/cv2.

**Spec:** [`2026-04-21-handspring-v05-jarvis.md`](../specs/2026-04-21-handspring-v05-jarvis.md)

---

## Task 1: Add `index_x` / `index_y` to HandFeatures

**Files:**
- Modify: `src/handspring/types.py`
- Modify: `src/handspring/features.py`
- Modify: `tests/test_types.py`, `tests/test_features.py`, `tests/test_osc_out.py`, `tests/test_motion.py`, `tests/test_synth_ui.py` (all call sites that construct `HandFeatures`)

### Step 1: Extend `HandFeatures` in `types.py`

Add two required fields (no defaults — forces explicit updates at every call site, avoids silent bugs):

```python
@dataclass(frozen=True)
class HandFeatures:
    x: float
    y: float
    z: float
    openness: float
    pinch: float
    index_x: float
    index_y: float
```

### Step 2: Extend `hand_features` in `features.py`

At the end of `hand_features`, compute index tip coordinates from landmark `INDEX_TIP` (8):

```python
    index_x = float(np.clip(landmarks[INDEX_TIP][0], 0.0, 1.0))
    index_y = float(np.clip(landmarks[INDEX_TIP][1], 0.0, 1.0))

    return HandFeatures(
        x=x, y=y, z=z, openness=openness, pinch=pinch,
        index_x=index_x, index_y=index_y,
    )
```

### Step 3: Update every test that constructs `HandFeatures`

Add `index_x=<value>`, `index_y=<value>` to every call. Where the test's `x`/`y` represent palm, use those same values for `index_x`/`index_y` by default (the tests don't care about index tip specifically).

Files to update with all-call-site pattern:
- `tests/test_types.py` — at minimum `test_hand_features_frozen`, `test_hand_state_present`
- `tests/test_osc_out.py` — `_frame()` and `_frame_with_pose()` helpers (used by many tests)
- `tests/test_motion.py` — `_hf()` helper
- `tests/test_synth_ui.py` — `_hand()` helper

Simplest pattern for helpers: default `index_x=0.5, index_y=0.5` or mirror `x`/`y`.

### Step 4: Add feature tests

In `tests/test_features.py`:

```python
def test_hand_features_index_tip_extracted():
    # Build a skeleton; move only INDEX_TIP to a known coord.
    from tests.fixtures import hand_open
    lm = hand_open()
    lm[8] = (0.73, 0.21, 0.0)  # INDEX_TIP = landmark 8
    f = hand_features(lm)
    assert abs(f.index_x - 0.73) < 1e-6
    assert abs(f.index_y - 0.21) < 1e-6
```

### Step 5: Verify

```bash
pytest
ruff check .
ruff format --check .
mypy src/
```

All green. If any call site was missed, the test will fail with a clear "missing 1 required keyword argument" error.

### Step 6: Commit

```bash
git add src/handspring/types.py src/handspring/features.py tests/
git commit -m "feat(features): add index_x/index_y to HandFeatures"
```

---

## Task 2: Emit index tip via OSC

**Files:**
- Modify: `src/handspring/osc_out.py`
- Modify: `tests/test_osc_out.py`

### Step 1: Test

Append:

```python
def test_index_tip_emitted():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame())  # right and left both present with hf defaults
    addresses = [addr for addr, _ in fake.sent]
    assert "/hand/left/index_x" in addresses
    assert "/hand/left/index_y" in addresses
    assert "/hand/right/index_x" in addresses
    assert "/hand/right/index_y" in addresses
```

### Step 2: Implementation

In `_emit_hand`, inside the `if state.present and state.features is not None:` block, after the existing feature sends add:

```python
            self._client.send_message(f"/hand/{side}/index_x", float(f.index_x))
            self._client.send_message(f"/hand/{side}/index_y", float(f.index_y))
```

### Step 3: Verify + commit

```bash
pytest
ruff check . && ruff format --check . && mypy src/
git add src/handspring/osc_out.py tests/test_osc_out.py
git commit -m "feat(osc_out): emit /hand/<side>/index_x and /index_y"
```

---

## Task 3: AppMode state machine with mouth debouncer

**Files:**
- Create: `src/handspring/app_mode.py`
- Create: `tests/test_app_mode.py`

### Step 1: Tests

```python
"""App-level mode toggle driven by sustained mouth-open."""
from __future__ import annotations

from handspring.app_mode import AppModeController


def _step(ctrl: AppModeController, mouth: float, n: int) -> None:
    for _ in range(n):
        ctrl.update(mouth_open=mouth, face_present=True, now=0.0 if n == 0 else None)


def test_default_is_synth():
    c = AppModeController()
    assert c.mode() == "synth"


def test_sustained_mouth_open_toggles_to_jarvis():
    c = AppModeController()
    # 14 frames is not enough (need 15).
    for i in range(14):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    assert c.mode() == "synth"
    c.update(mouth_open=0.8, face_present=True, now=14 * 0.033)
    assert c.mode() == "jarvis"


def test_second_toggle_respects_cooldown():
    c = AppModeController()
    # First toggle.
    for i in range(15):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    assert c.mode() == "jarvis"
    # Immediately hold mouth open: should NOT toggle back (cooldown active).
    for i in range(15):
        c.update(mouth_open=0.8, face_present=True, now=(15 + i) * 0.033)
    assert c.mode() == "jarvis"
    # After 1.5 s cooldown: toggle allowed.
    for i in range(15):
        c.update(mouth_open=0.8, face_present=True, now=2.0 + i * 0.033)
    assert c.mode() == "synth"


def test_brief_mouth_open_does_not_toggle():
    c = AppModeController()
    # 10 frames of open, then closed — should not toggle.
    for i in range(10):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    for i in range(20):
        c.update(mouth_open=0.1, face_present=True, now=(10 + i) * 0.033)
    assert c.mode() == "synth"


def test_face_absent_resets_counter():
    c = AppModeController()
    for i in range(10):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    # Face gone — counter resets.
    c.update(mouth_open=0.8, face_present=False, now=10 * 0.033)
    for i in range(10):
        c.update(mouth_open=0.8, face_present=True, now=(11 + i) * 0.033)
    assert c.mode() == "synth"


def test_mode_transition_callback():
    transitions = []
    c = AppModeController(on_change=lambda mode: transitions.append(mode))
    for i in range(15):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    assert transitions == ["jarvis"]
```

### Step 2: Implement `src/handspring/app_mode.py`

```python
"""App-level mode state machine: toggle between synth and jarvis via mouth-open."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

AppMode = Literal["synth", "jarvis"]

_MOUTH_THRESHOLD = 0.7
_DEBOUNCE_FRAMES = 15           # ~500 ms at 30 FPS
_COOLDOWN_SECONDS = 1.5


class AppModeController:
    def __init__(
        self,
        *,
        initial_mode: AppMode = "synth",
        on_change: Callable[[AppMode], None] | None = None,
    ) -> None:
        self._mode: AppMode = initial_mode
        self._on_change = on_change
        self._frames_open = 0
        self._last_toggle_time: float = -_COOLDOWN_SECONDS

    def mode(self) -> AppMode:
        return self._mode

    def update(
        self,
        *,
        mouth_open: float,
        face_present: bool,
        now: float,
    ) -> None:
        """Call once per frame with the current face.mouth_open + face.present."""
        if not face_present:
            self._frames_open = 0
            return

        if mouth_open >= _MOUTH_THRESHOLD:
            self._frames_open += 1
        else:
            self._frames_open = 0

        if self._frames_open >= _DEBOUNCE_FRAMES and now - self._last_toggle_time >= _COOLDOWN_SECONDS:
            self._mode = "jarvis" if self._mode == "synth" else "synth"
            self._frames_open = 0
            self._last_toggle_time = now
            if self._on_change is not None:
                self._on_change(self._mode)
```

### Step 3: Verify + commit

```bash
pytest tests/test_app_mode.py -v
ruff check . && ruff format --check . && mypy src/
git add src/handspring/app_mode.py tests/test_app_mode.py
git commit -m "feat(app_mode): mouth-open-triggered mode toggle with debounce+cooldown"
```

---

## Task 4: Window + WindowManager (data + geometry)

**Files:**
- Create: `src/handspring/jarvis.py` (initial skeleton: Window, WindowManager)
- Create: `tests/test_jarvis.py`

This task only covers the data layer and hit-testing. Gesture-state machines and rendering come in subsequent tasks.

### Step 1: Tests

```python
"""Window data + WindowManager basic operations."""
from __future__ import annotations

from handspring.jarvis import Window, WindowManager


def test_window_contains_palm():
    w = Window(id=1, x=0.2, y=0.3, width=0.4, height=0.3, z=0, color_idx=0)
    assert w.contains(0.4, 0.45)   # inside
    assert not w.contains(0.1, 0.1)  # outside


def test_manager_create_assigns_ids():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.2)
    b = m.create(x=0.5, y=0.5, width=0.3, height=0.2)
    assert a.id != b.id


def test_manager_cap_evicts_oldest():
    m = WindowManager(max_windows=3)
    created = [m.create(x=0.1 * i, y=0.1, width=0.1, height=0.1) for i in range(5)]
    windows = m.windows()
    assert len(windows) == 3
    # The three most-recently-created survive.
    remaining_ids = {w.id for w in windows}
    assert created[0].id not in remaining_ids
    assert created[1].id not in remaining_ids
    assert created[-1].id in remaining_ids


def test_manager_z_order_create_goes_top():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    b = m.create(x=0.2, y=0.2, width=0.3, height=0.3)
    # b is on top.
    assert m.topmost_at(0.25, 0.25) is not None
    assert m.topmost_at(0.25, 0.25).id == b.id


def test_promote_to_top():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    b = m.create(x=0.2, y=0.2, width=0.3, height=0.3)
    m.promote(a.id)
    # Now a should be topmost under the overlap.
    top = m.topmost_at(0.25, 0.25)
    assert top is not None and top.id == a.id


def test_move_window():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    m.move(a.id, dx=0.2, dy=-0.05)
    moved = m.get(a.id)
    assert moved is not None
    assert abs(moved.x - 0.3) < 1e-6
    assert abs(moved.y - 0.05) < 1e-6


def test_move_clamps_to_frame():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    m.move(a.id, dx=10.0, dy=10.0)
    moved = m.get(a.id)
    assert moved is not None
    assert moved.x + moved.width <= 1.0 + 1e-6
    assert moved.y + moved.height <= 1.0 + 1e-6


def test_cycle_color():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    initial = a.color_idx
    m.cycle_color(a.id)
    assert m.get(a.id).color_idx == (initial + 1) % 3  # type: ignore[union-attr]
```

### Step 2: Implement `src/handspring/jarvis.py` (data layer only)

```python
"""JARVIS mode: transparent-window creation, grab/drag, tap."""

from __future__ import annotations

from dataclasses import dataclass, replace


_MAX_WINDOWS_DEFAULT = 8
_NUM_COLORS = 3
# Palette (BGR tuples for OpenCV).
WINDOW_COLORS = [
    (255, 180, 77),   # blue  #4DB4FF
    (100, 230, 120),  # green
    (220, 110, 210),  # purple
]


@dataclass(frozen=True)
class Window:
    """Rectangle in normalized (0..1) coordinates."""

    id: int
    x: float
    y: float
    width: float
    height: float
    z: int            # z-order index (higher = on top)
    color_idx: int    # index into WINDOW_COLORS

    def contains(self, px: float, py: float) -> bool:
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


class WindowManager:
    def __init__(self, *, max_windows: int = _MAX_WINDOWS_DEFAULT) -> None:
        self._max = max_windows
        self._next_id = 1
        self._next_z = 1
        self._windows: list[Window] = []  # creation order

    def create(self, *, x: float, y: float, width: float, height: float) -> Window:
        w = Window(
            id=self._next_id,
            x=x,
            y=y,
            width=width,
            height=height,
            z=self._next_z,
            color_idx=0,
        )
        self._next_id += 1
        self._next_z += 1
        self._windows.append(w)
        # Evict FIFO when exceeding the cap.
        while len(self._windows) > self._max:
            self._windows.pop(0)
        return w

    def windows(self) -> list[Window]:
        """Return windows in z-order (bottom → top)."""
        return sorted(self._windows, key=lambda w: w.z)

    def get(self, window_id: int) -> Window | None:
        for w in self._windows:
            if w.id == window_id:
                return w
        return None

    def topmost_at(self, px: float, py: float) -> Window | None:
        hit = [w for w in self._windows if w.contains(px, py)]
        if not hit:
            return None
        return max(hit, key=lambda w: w.z)

    def promote(self, window_id: int) -> None:
        w = self.get(window_id)
        if w is None:
            return
        new_w = replace(w, z=self._next_z)
        self._next_z += 1
        self._replace(new_w)

    def move(self, window_id: int, *, dx: float, dy: float) -> None:
        w = self.get(window_id)
        if w is None:
            return
        nx = max(0.0, min(1.0 - w.width, w.x + dx))
        ny = max(0.0, min(1.0 - w.height, w.y + dy))
        self._replace(replace(w, x=nx, y=ny))

    def cycle_color(self, window_id: int) -> None:
        w = self.get(window_id)
        if w is None:
            return
        self._replace(replace(w, color_idx=(w.color_idx + 1) % _NUM_COLORS))

    def _replace(self, new_w: Window) -> None:
        for i, w in enumerate(self._windows):
            if w.id == new_w.id:
                self._windows[i] = new_w
                return
```

### Step 3: Verify + commit

```bash
pytest tests/test_jarvis.py -v
ruff check . && ruff format --check . && mypy src/
git add src/handspring/jarvis.py tests/test_jarvis.py
git commit -m "feat(jarvis): Window dataclass + WindowManager (create, z-order, move)"
```

---

## Task 5: Jarvis gesture state machine (create / grab / tap)

**Files:**
- Modify: `src/handspring/jarvis.py` — add `JarvisController`
- Modify: `tests/test_jarvis.py` — gesture tests

This is the state machine layer that reads a FrameResult and drives the WindowManager.

### Step 1: Tests (append to `tests/test_jarvis.py`)

```python
from handspring.jarvis import JarvisController
from handspring.types import (
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
    MotionState,
    PoseState,
)


def _hf(x: float, y: float, pinch: float = 0.0) -> HandFeatures:
    return HandFeatures(
        x=x, y=y, z=0.0, openness=1.0, pinch=pinch, index_x=x, index_y=y
    )


def _hand(gesture: str, x: float, y: float, pinch: float = 0.0) -> HandState:
    return HandState(
        present=True,
        features=_hf(x, y, pinch),
        gesture=gesture,  # type: ignore[arg-type]
        motion=MotionState(pinching=pinch >= 0.85, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None),
    )


def _absent() -> HandState:
    return HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )


def _face() -> FaceState:
    return FaceState(
        present=False, features=None,
        expression="neutral", eye_left_open=0.0, eye_right_open=0.0,
    )


def _frame(left: HandState, right: HandState) -> FrameResult:
    return FrameResult(
        left=left, right=right, face=_face(),
        pose=PoseState(False, None), fps=30.0, clap_event=False,
    )


def test_both_pinch_and_pull_creates_window():
    c = JarvisController()
    # Start: both hands close together, pinching.
    c.update(_frame(_hand("fist", 0.4, 0.5, pinch=0.95), _hand("fist", 0.5, 0.5, pinch=0.95)))
    assert c.manager.windows() == []  # not created yet (still pinching)
    # Pull apart.
    c.update(_frame(_hand("fist", 0.25, 0.35, pinch=0.95), _hand("fist", 0.65, 0.65, pinch=0.95)))
    # Release one hand's pinch → commit.
    c.update(_frame(_hand("fist", 0.25, 0.35, pinch=0.2), _hand("fist", 0.65, 0.65, pinch=0.95)))
    assert len(c.manager.windows()) == 1


def test_tiny_pull_discarded():
    c = JarvisController()
    c.update(_frame(_hand("fist", 0.45, 0.50, pinch=0.95), _hand("fist", 0.47, 0.50, pinch=0.95)))
    # Release at almost-identical positions — diagonal < 0.1.
    c.update(_frame(_hand("fist", 0.45, 0.50, pinch=0.2), _hand("fist", 0.47, 0.50, pinch=0.95)))
    assert c.manager.windows() == []


def test_grab_drag_release():
    c = JarvisController()
    # Seed a window at center.
    w = c.manager.create(x=0.3, y=0.3, width=0.3, height=0.3)
    # Open hand over the window, right side.
    c.update(_frame(_absent(), _hand("open", 0.45, 0.45)))
    # Close to fist over the window — grab.
    c.update(_frame(_absent(), _hand("fist", 0.45, 0.45)))
    # Move hand right.
    c.update(_frame(_absent(), _hand("fist", 0.60, 0.45)))
    # Open hand — release.
    c.update(_frame(_absent(), _hand("open", 0.60, 0.45)))
    moved = c.manager.get(w.id)
    assert moved is not None
    assert moved.x > 0.3  # dragged rightward


def test_point_tap_after_hover():
    c = JarvisController()
    w = c.manager.create(x=0.3, y=0.3, width=0.3, height=0.3)
    initial_color = w.color_idx
    # Point with index tip inside window for 5 frames at 30 FPS.
    for i in range(6):
        c.update(_frame(_absent(), _hand("point", 0.45, 0.45)))
    post = c.manager.get(w.id)
    assert post is not None
    assert post.color_idx != initial_color
    # Tap event reported for this frame cycle
    assert c.last_tap_id() == w.id


def test_point_no_tap_if_moves_away_early():
    c = JarvisController()
    w = c.manager.create(x=0.3, y=0.3, width=0.3, height=0.3)
    # 2 frames in, then leave — not long enough.
    c.update(_frame(_absent(), _hand("point", 0.45, 0.45)))
    c.update(_frame(_absent(), _hand("point", 0.45, 0.45)))
    c.update(_frame(_absent(), _hand("point", 0.1, 0.1)))
    unchanged = c.manager.get(w.id)
    assert unchanged is not None
    assert unchanged.color_idx == w.color_idx
```

### Step 2: Implement `JarvisController` in `jarvis.py` (append)

```python
from dataclasses import dataclass, field, replace  # already imported above; ignore dup
from typing import Literal

from handspring.types import FrameResult, HandState, Side


_TAP_HOVER_FRAMES = 5        # 5 frames @ 30fps ≈ 150 ms
_TAP_COOLDOWN_SECONDS = 0.4
_MIN_WINDOW_DIAGONAL = 0.1
_PINCH_ON_THRESHOLD = 0.85


@dataclass
class _GrabState:
    window_id: int
    side: Side
    last_palm_x: float
    last_palm_y: float


@dataclass
class _CreateState:
    start_left: tuple[float, float]
    start_right: tuple[float, float]


class JarvisController:
    def __init__(self) -> None:
        self.manager = WindowManager()
        self._grab: _GrabState | None = None
        self._create: _CreateState | None = None
        self._hover_frames: dict[Side, int] = {"left": 0, "right": 0}
        self._last_hover_window: dict[Side, int | None] = {"left": None, "right": None}
        self._tap_cooldown_by_window: dict[int, float] = {}
        self._last_tap_id: int | None = None
        self._last_tap_frame_consumed = False
        self._events_out: list[tuple[str, int]] = []

    # ---- Events consumed by __main__ to emit OSC ----

    def pop_events(self) -> list[tuple[str, int]]:
        """Return events accumulated during update() and clear the buffer.

        Event tuples: (kind, window_id). Kinds: "created" | "tap".
        """
        out = self._events_out
        self._events_out = []
        return out

    def last_tap_id(self) -> int | None:
        return self._last_tap_id

    # ---- Main update loop ----

    def update(self, frame: FrameResult, now: float | None = None) -> None:
        # Use a synthetic monotonic timestamp if caller didn't pass one.
        if now is None:
            now = 0.0

        # Re-allow last_tap_id each frame (transient: caller reads right after update).
        self._last_tap_id = None

        self._handle_create(frame, now)
        self._handle_grab(frame, now)
        self._handle_tap(frame, now)

    # ---- 1. Create: both-hand pinch + pull apart ----

    def _handle_create(self, frame: FrameResult, _now: float) -> None:
        left = frame.left
        right = frame.right
        both_pinching = (
            left.present and right.present
            and left.features is not None and right.features is not None
            and left.features.pinch >= _PINCH_ON_THRESHOLD
            and right.features.pinch >= _PINCH_ON_THRESHOLD
        )
        if both_pinching:
            if self._create is None:
                self._create = _CreateState(
                    start_left=(left.features.x, left.features.y),   # type: ignore[union-attr]
                    start_right=(right.features.x, right.features.y),  # type: ignore[union-attr]
                )
            return

        # If we were in a create gesture and now lost pinch on either hand, commit.
        if self._create is not None and left.features is not None and right.features is not None:
            # Use current (at-release) positions as the rectangle corners.
            x1, y1 = left.features.x, left.features.y
            x2, y2 = right.features.x, right.features.y
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            w = x_max - x_min
            h = y_max - y_min
            diag = (w * w + h * h) ** 0.5
            if diag >= _MIN_WINDOW_DIAGONAL:
                # Clamp aspect ratio to [0.5, 2.0]
                if w > 0 and h > 0:
                    ar = w / h
                    if ar < 0.5:
                        h = w / 0.5
                    elif ar > 2.0:
                        h = w / 2.0
                created = self.manager.create(x=x_min, y=y_min, width=w, height=h)
                self._events_out.append(("created", created.id))
        self._create = None

    # ---- 2. Grab/drag: open → fist while palm is over a window ----

    def _handle_grab(self, frame: FrameResult, _now: float) -> None:
        if self._grab is None:
            # Look for either hand transitioning to fist while over a window.
            for side in ("left", "right"):  # left has priority
                state = getattr(frame, side)
                if not state.present or state.features is None:
                    continue
                if state.gesture != "fist":
                    continue
                target = self.manager.topmost_at(state.features.x, state.features.y)
                if target is None:
                    continue
                self.manager.promote(target.id)
                self._grab = _GrabState(
                    window_id=target.id,
                    side=side,  # type: ignore[arg-type]
                    last_palm_x=state.features.x,
                    last_palm_y=state.features.y,
                )
                return
            return

        # Already grabbing: track motion, release on open.
        grab = self._grab
        state = getattr(frame, grab.side)
        if not state.present or state.features is None or state.gesture != "fist":
            self._grab = None
            return
        dx = state.features.x - grab.last_palm_x
        dy = state.features.y - grab.last_palm_y
        self.manager.move(grab.window_id, dx=dx, dy=dy)
        self._grab = _GrabState(
            window_id=grab.window_id,
            side=grab.side,
            last_palm_x=state.features.x,
            last_palm_y=state.features.y,
        )

    # ---- 3. Tap: point + hover on window for 150 ms ----

    def _handle_tap(self, frame: FrameResult, now: float) -> None:
        for side in ("left", "right"):  # type: ignore[assignment]
            state = getattr(frame, side)
            if not state.present or state.features is None or state.gesture != "point":
                self._hover_frames[side] = 0  # type: ignore[index]
                self._last_hover_window[side] = None  # type: ignore[index]
                continue
            target = self.manager.topmost_at(state.features.index_x, state.features.index_y)
            if target is None:
                self._hover_frames[side] = 0  # type: ignore[index]
                self._last_hover_window[side] = None  # type: ignore[index]
                continue
            if self._last_hover_window[side] == target.id:  # type: ignore[index]
                self._hover_frames[side] += 1  # type: ignore[index]
            else:
                self._hover_frames[side] = 1  # type: ignore[index]
                self._last_hover_window[side] = target.id  # type: ignore[index]

            if self._hover_frames[side] >= _TAP_HOVER_FRAMES:  # type: ignore[index]
                last_tap = self._tap_cooldown_by_window.get(target.id, -1.0)
                if now - last_tap >= _TAP_COOLDOWN_SECONDS:
                    self.manager.cycle_color(target.id)
                    self.manager.promote(target.id)
                    self._tap_cooldown_by_window[target.id] = now
                    self._events_out.append(("tap", target.id))
                    self._last_tap_id = target.id
                # Reset hover counter so a continuous hover doesn't retap until cooldown.
                self._hover_frames[side] = 0  # type: ignore[index]
```

### Step 3: Verify + commit

```bash
pytest tests/test_jarvis.py -v
ruff check . && ruff format --check . && mypy src/
git add src/handspring/jarvis.py tests/test_jarvis.py
git commit -m "feat(jarvis): create/grab/tap gesture state machines"
```

---

## Task 6: Jarvis renderer (semi-transparent windows + title bar + highlight)

**Files:**
- Modify: `src/handspring/preview.py`

Add a `_render_jarvis` function plus a mode badge drawn at the top of every frame. Extend `Preview.render()` signature to accept the current `AppMode` + optional `JarvisController`.

### Step 1: Extend Preview.render

Add imports:
```python
from handspring.app_mode import AppMode
from handspring.jarvis import JarvisController, WINDOW_COLORS
```

Extend `Preview.render()` signature:
```python
    def render(
        self,
        bgr_frame: NDArray[np.uint8],
        hand_landmark_lists: list[Any],
        face_landmark_lists: list[Any],
        pose_landmarks: Any | None,
        frame_result: FrameResult,
        osc_target: str,
        synth_snapshot: SynthSnapshot | None,
        synth_hint: UiHint | None,
        app_mode: AppMode,
        jarvis: JarvisController | None,
    ) -> bool:
```

Inside `render` (after drawing status but before returning), branch on mode:

```python
        # Mode badge (always visible)
        _draw_mode_badge(display, app_mode)

        if app_mode == "jarvis" and jarvis is not None:
            _draw_jarvis(display, jarvis, mirrored=self._mirror)
        # Synth panel only in synth mode
        if app_mode == "synth" and synth_snapshot is not None:
            _draw_synth_panel(display, synth_snapshot)
        if app_mode == "synth" and synth_hint is not None and synth_hint.kind != "none":
            _draw_synth_hint(display, synth_hint, mirrored=self._mirror)
```

Add new module-level helpers:

```python
_JARVIS_HINT_TEXT = "pinch-open to spawn - grab to drag - point to tap"


def _draw_mode_badge(frame: NDArray[np.uint8], mode: AppMode) -> None:
    h, w = frame.shape[:2]
    text = "JARVIS" if mode == "jarvis" else "SYNTH"
    color = (136, 255, 0) if mode == "jarvis" else (230, 230, 230)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cx = (w - tw) // 2
    cy = 44
    # Background pill
    pad = 14
    cv2.rectangle(
        frame,
        (cx - pad, cy - th - 8),
        (cx + tw + pad, cy + 10),
        (20, 20, 20),
        -1,
    )
    cv2.rectangle(
        frame,
        (cx - pad, cy - th - 8),
        (cx + tw + pad, cy + 10),
        color,
        2,
    )
    cv2.putText(frame, text, (cx, cy), font, scale, color, thick, cv2.LINE_AA)
    # Hint line (jarvis only)
    if mode == "jarvis":
        (hw, hh), _ = cv2.getTextSize(_JARVIS_HINT_TEXT, font, 0.45, 1)
        hcx = (w - hw) // 2
        cv2.putText(
            frame, _JARVIS_HINT_TEXT, (hcx, cy + 28),
            font, 0.45, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, _JARVIS_HINT_TEXT, (hcx, cy + 28),
            font, 0.45, (180, 220, 180), 1, cv2.LINE_AA,
        )


def _draw_jarvis(frame: NDArray[np.uint8], jarvis: JarvisController, *, mirrored: bool) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    alpha = 0.35
    highlight_color = (136, 255, 0)  # neon green

    for win in jarvis.manager.windows():
        # Translate normalized coords to pixels; apply mirror if needed.
        if mirrored:
            x0_n = 1.0 - (win.x + win.width)
        else:
            x0_n = win.x
        x0 = int(x0_n * w)
        y0 = int(win.y * h)
        x1 = int((x0_n + win.width) * w)
        y1 = int((win.y + win.height) * h)

        fill = WINDOW_COLORS[win.color_idx]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1)

    # Composite overlay onto frame with alpha.
    import numpy as np
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    # Borders, titles drawn on top (fully opaque).
    for win in jarvis.manager.windows():
        if mirrored:
            x0_n = 1.0 - (win.x + win.width)
        else:
            x0_n = win.x
        x0 = int(x0_n * w)
        y0 = int(win.y * h)
        x1 = int((x0_n + win.width) * w)
        y1 = int((win.y + win.height) * h)

        border = WINDOW_COLORS[win.color_idx]
        cv2.rectangle(frame, (x0, y0), (x1, y1), border, 2)
        # Title bar
        cv2.rectangle(frame, (x0, y0), (x1, min(y0 + 22, y1)), border, -1)
        cv2.putText(
            frame, f"Window {win.id}", (x0 + 8, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
        )
```

### Step 2: Verify

```bash
ruff check src/handspring/preview.py
ruff format --check src/handspring/preview.py
mypy src/
python -c "from handspring.preview import Preview; Preview(); print('ok')"
pytest
```

The test suite may still pass because tests don't touch preview.render() directly. If `__main__.py` breaks due to the new render signature, that's expected — Task 7 fixes it.

### Step 3: Commit

```bash
git add src/handspring/preview.py
git commit -m "feat(preview): mode badge + jarvis window rendering"
```

---

## Task 7: OSC — app mode + jarvis events

**Files:**
- Modify: `src/handspring/osc_out.py`
- Modify: `tests/test_osc_out.py`

### Step 1: Tests (append)

```python
def test_app_mode_change_emitted():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit_app_mode("synth")
    emitter.emit_app_mode("synth")
    emitter.emit_app_mode("jarvis")
    modes = [v for a, v in fake.sent if a == "/app/mode"]
    assert modes == ["synth", "jarvis"]


def test_window_events_emitted():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit_jarvis_events([("created", 5), ("tap", 5), ("tap", 7)], window_count=2)
    assert ("/jarvis/window_created", 5) in fake.sent
    assert ("/jarvis/window_tap", 5) in fake.sent
    assert ("/jarvis/window_tap", 7) in fake.sent
    assert ("/jarvis/window_count", 2) in fake.sent


def test_window_count_only_on_change():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit_jarvis_events([], window_count=3)
    emitter.emit_jarvis_events([], window_count=3)
    emitter.emit_jarvis_events([], window_count=4)
    counts = [v for a, v in fake.sent if a == "/jarvis/window_count"]
    assert counts == [3, 4]
```

### Step 2: Implementation

Add to imports:
```python
from handspring.app_mode import AppMode
```

Add to `__init__`:
```python
        self._last_app_mode: AppMode | None = None
        self._last_window_count: int | None = None
```

Add two new public methods:
```python
    def emit_app_mode(self, mode: AppMode) -> None:
        if mode != self._last_app_mode:
            self._client.send_message("/app/mode", mode)
            self._last_app_mode = mode

    def emit_jarvis_events(
        self, events: list[tuple[str, int]], *, window_count: int
    ) -> None:
        for kind, window_id in events:
            if kind == "created":
                self._client.send_message("/jarvis/window_created", int(window_id))
            elif kind == "tap":
                self._client.send_message("/jarvis/window_tap", int(window_id))
        if window_count != self._last_window_count:
            self._client.send_message("/jarvis/window_count", int(window_count))
            self._last_window_count = window_count
```

### Step 3: Verify + commit

```bash
pytest
ruff check . && ruff format --check . && mypy src/
git add src/handspring/osc_out.py tests/test_osc_out.py
git commit -m "feat(osc_out): emit /app/mode and /jarvis/* events"
```

---

## Task 8: Wire into `__main__.py`

**Files:**
- Modify: `src/handspring/__main__.py`

### Step 1: Add imports + controllers

Top of file, near other handspring imports:
```python
from handspring.app_mode import AppModeController
from handspring.jarvis import JarvisController
```

### Step 2: Construct controllers

In `main()`, after `synth_controller = SynthController(synth_params)`:

```python
    app_mode_controller = AppModeController()
    jarvis = JarvisController()
```

### Step 3: Per-frame loop updates

After `synth_controller.update(result)`, add:

```python
            # App-level mode toggle via mouth open.
            now = time.monotonic()
            mouth_open_val = (
                result.face.features.mouth_open
                if result.face.present and result.face.features is not None
                else 0.0
            )
            app_mode_controller.update(
                mouth_open=mouth_open_val,
                face_present=result.face.present,
                now=now,
            )
            mode = app_mode_controller.mode()

            # In Jarvis mode: mute synth by zeroing the volume target.
            # In Synth mode: restore volume to its last-edited value.
            if mode == "jarvis":
                synth_params.set_volume(0.0)
                jarvis.update(result, now=now)

            # Emit mode + jarvis events.
            emitter.emit_app_mode(mode)
            if mode == "jarvis":
                emitter.emit_jarvis_events(
                    jarvis.pop_events(),
                    window_count=len(jarvis.manager.windows()),
                )
```

**Important:** the "restore volume" behavior — we don't want every frame to clobber the user's last volume. Pattern: track the last non-zero volume before entering Jarvis; on mode transition back to synth, restore it. Simple implementation: store `last_user_volume` on mode change. Add this via the `on_change` callback hook:

Replace the construction line:
```python
    last_user_volume = {"v": synth_params.snapshot().volume}

    def _on_app_mode_change(new_mode):
        if new_mode == "jarvis":
            last_user_volume["v"] = synth_params.snapshot().volume
            synth_params.set_volume(0.0)
        else:
            synth_params.set_volume(last_user_volume["v"])

    app_mode_controller = AppModeController(on_change=_on_app_mode_change)
```

Remove the per-frame `if mode == "jarvis": synth_params.set_volume(0.0)` line (the callback handles it).

### Step 4: Pass through to preview

Update the `preview.render(...)` call to pass `mode` + `jarvis`:

```python
                if not preview.render(
                    bgr,
                    hand_landmarks,
                    face_landmarks,
                    pose_landmarks,
                    result,
                    f"{args.host}:{args.port}",
                    snap_for_preview,
                    hint_for_preview,
                    mode,
                    jarvis,
                ):
                    break
```

### Step 5: Verify + commit

```bash
pytest
ruff check .
ruff format --check .
mypy src/
python -m handspring --version
python -m handspring --help
```

All green. `--help` unchanged (no new flags).

```bash
git add src/handspring/__main__.py
git commit -m "feat(main): wire app-mode controller + jarvis + mute-on-jarvis"
```

---

## Task 9: README + bump version + tag v0.5.0

**Files:**
- Modify: `README.md`
- Modify: `src/handspring/__init__.py` — `0.4.0` → `0.5.0`
- Modify: `pyproject.toml` — `0.4.0` → `0.5.0`

### Step 1: README additions

Add a new top-level section after "Built-in synth":

````markdown
## JARVIS mode

handspring hosts multiple interaction modes. Open your mouth wide for about
half a second to toggle between **SYNTH** (default) and **JARVIS**. Your audio
cuts out on entering Jarvis and resumes on returning.

In Jarvis mode:

- **Create** — pinch both thumb-index pairs together, then pull your hands
  apart. A semi-transparent blue rectangle follows your hands. Release either
  pinch to commit. Tiny movements are ignored.
- **Grab** — hold an open hand over a window, then close it. The window
  follows your palm. Open your hand to drop.
- **Tap** — point with an index finger. Move your fingertip over a window.
  After ~150 ms, the window cycles colour and an OSC tap event fires.

Up to 8 windows. Older ones get evicted first.

### Jarvis OSC

| Address | Type | Notes |
|---|---|---|
| `/app/mode` | string | `synth` / `jarvis` — on change |
| `/jarvis/window_count` | int | on change |
| `/jarvis/window_created` | int | window id, one-shot |
| `/jarvis/window_tap` | int | window id, one-shot |

Also new per-hand: `/hand/<side>/index_x` and `/hand/<side>/index_y` (both
float 0..1) for the index fingertip.
````

### Step 2: Version bumps

Both `__init__.py` and `pyproject.toml` → `0.5.0`.

### Step 3: Full check + commit + tag

```bash
ruff check .
ruff format --check .
mypy src/
pytest
python -m handspring --version   # → 0.5.0

git add README.md src/handspring/__init__.py pyproject.toml
git commit -m "docs: document JARVIS mode + mouth toggle + index tip OSC"
git tag -a v0.5.0 -m "handspring v0.5.0 — JARVIS mode + app-level toggle"
```

Do NOT push.

### Step 4: Report

- Commits since v0.4.0
- Total test count (expect ≥ 110)
- Any manual tweaks
- Tag SHA

---

## Execution notes

- **`Window` frozen means `move()` replaces the whole record.** Keep all mutations going through `WindowManager._replace`.
- **Hit-testing `topmost_at` returns the window, not the id** — call sites unwrap.
- **Mirror handling:** WindowManager stores normalized positions in "user-perspective" (same convention as hand features x/y already use). The preview handles the mirror by flipping x at render time. Every gesture/controller call uses un-mirrored coords.
- **If the user's pinch feature never reaches 0.85 in real tests, the threshold may need to drop.** Don't adjust yet — first verify by checking `/hand/*/pinch` values from OSC during a real pinch. The threshold is central in multiple places; change carefully.
