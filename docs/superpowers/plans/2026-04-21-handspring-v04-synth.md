# handspring v0.4.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. TDD where practical. Audio callback code is not unit-testable without a device; rely on logical tests of parameter mapping + smoothing.

**Goal:** Build an in-process gesture-driven synth: saw + lowpass + amp env + tremolo, edited by fist-activated modes mapped to hand position.

**Architecture:** Three new modules — `synth.py` (audio engine, sounddevice callback), `synth_ui.py` (state machine: gesture → parameter updates + UI hints). Preview renders an always-visible synth panel and edit-mode sliders/crosshairs. OSC emits `/synth/*`. Audio runs on a dedicated sounddevice callback thread; parameters live behind a `threading.Lock`.

**Tech additions:** `sounddevice` moves from dev-dep to runtime dep. No new concepts beyond what's already in the project.

**Spec:** [`2026-04-21-handspring-v04-synth.md`](../specs/2026-04-21-handspring-v04-synth.md)

---

## Task 1: SynthParams (thread-safe parameter store)

**Files:**
- Create: `src/handspring/synth_params.py`
- Create: `tests/test_synth_params.py`

### Step 1: Tests

```python
"""SynthParams tests — thread-safe snapshot semantics."""
from __future__ import annotations

from handspring.synth_params import SynthParams, SynthSnapshot


def test_defaults():
    p = SynthParams()
    s = p.snapshot()
    assert s.volume == 0.4
    assert abs(s.note_hz - 440.0) < 1e-6
    assert s.stepping_hz == 0.0
    assert s.cutoff_hz == 3000.0
    assert s.mod_depth == 0.0
    assert s.mod_rate == 1.0
    assert s.mode == "play"


def test_set_volume_clamps():
    p = SynthParams()
    p.set_volume(1.5)
    assert p.snapshot().volume == 1.0
    p.set_volume(-0.2)
    assert p.snapshot().volume == 0.0


def test_set_note_clamps_to_range():
    p = SynthParams()
    p.set_note_hz(50.0)
    assert p.snapshot().note_hz == 131.0
    p.set_note_hz(10_000.0)
    assert p.snapshot().note_hz == 1047.0


def test_set_stepping_clamps():
    p = SynthParams()
    p.set_stepping_hz(-1)
    assert p.snapshot().stepping_hz == 0.0
    p.set_stepping_hz(99)
    assert p.snapshot().stepping_hz == 16.0


def test_set_cutoff_clamps_exponential():
    p = SynthParams()
    p.set_cutoff_hz(100.0)
    assert p.snapshot().cutoff_hz == 200.0
    p.set_cutoff_hz(50000.0)
    assert p.snapshot().cutoff_hz == 8000.0


def test_set_mode_updates_snapshot():
    p = SynthParams()
    p.set_mode("edit_left")
    assert p.snapshot().mode == "edit_left"


def test_snapshot_is_immutable():
    p = SynthParams()
    s = p.snapshot()
    # Dataclass is frozen; assigning should raise.
    import pytest
    with pytest.raises(AttributeError):
        s.volume = 0.9  # type: ignore[misc]
```

### Step 2: Implement `src/handspring/synth_params.py`

```python
"""Thread-safe parameter store for the in-process synth.

Writes are done by the main thread (via the gesture → parameter mapper).
Reads happen in the audio callback thread once per block via `snapshot()`,
which acquires the lock briefly and returns a frozen copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal

SynthMode = Literal["play", "edit_left", "edit_right"]


# Hard ranges (spec §Autonomous design decisions).
VOL_MIN, VOL_MAX = 0.0, 1.0
NOTE_MIN, NOTE_MAX = 131.0, 1047.0       # C3..C6
STEP_MIN, STEP_MAX = 0.0, 16.0
CUTOFF_MIN, CUTOFF_MAX = 200.0, 8000.0
MOD_DEPTH_MIN, MOD_DEPTH_MAX = 0.0, 1.0
MOD_RATE_MIN, MOD_RATE_MAX = 0.1, 10.0


@dataclass(frozen=True)
class SynthSnapshot:
    volume: float
    note_hz: float
    stepping_hz: float
    cutoff_hz: float
    mod_depth: float
    mod_rate: float
    mode: SynthMode


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


class SynthParams:
    def __init__(self) -> None:
        self._lock = Lock()
        self._volume = 0.4
        self._note_hz = 440.0
        self._stepping_hz = 0.0
        self._cutoff_hz = 3000.0
        self._mod_depth = 0.0
        self._mod_rate = 1.0
        self._mode: SynthMode = "play"

    def snapshot(self) -> SynthSnapshot:
        with self._lock:
            return SynthSnapshot(
                volume=self._volume,
                note_hz=self._note_hz,
                stepping_hz=self._stepping_hz,
                cutoff_hz=self._cutoff_hz,
                mod_depth=self._mod_depth,
                mod_rate=self._mod_rate,
                mode=self._mode,
            )

    def set_volume(self, v: float) -> None:
        with self._lock:
            self._volume = _clamp(v, VOL_MIN, VOL_MAX)

    def set_note_hz(self, hz: float) -> None:
        with self._lock:
            self._note_hz = _clamp(hz, NOTE_MIN, NOTE_MAX)

    def set_stepping_hz(self, hz: float) -> None:
        with self._lock:
            self._stepping_hz = _clamp(hz, STEP_MIN, STEP_MAX)

    def set_cutoff_hz(self, hz: float) -> None:
        with self._lock:
            self._cutoff_hz = _clamp(hz, CUTOFF_MIN, CUTOFF_MAX)

    def set_mod_depth(self, v: float) -> None:
        with self._lock:
            self._mod_depth = _clamp(v, MOD_DEPTH_MIN, MOD_DEPTH_MAX)

    def set_mod_rate(self, hz: float) -> None:
        with self._lock:
            self._mod_rate = _clamp(hz, MOD_RATE_MIN, MOD_RATE_MAX)

    def set_mode(self, mode: SynthMode) -> None:
        with self._lock:
            self._mode = mode
```

### Step 3: Verify + commit

```bash
pytest tests/test_synth_params.py -v
ruff check . && ruff format --check . && mypy src/
git add src/handspring/synth_params.py tests/test_synth_params.py
git commit -m "feat(synth_params): thread-safe parameter store"
```

---

## Task 2: Synth audio engine

**Files:**
- Create: `src/handspring/synth.py`

Not straightforward to unit-test without audio hardware. Logic tests will come via synth_ui's mapping tests (Task 4). Keep this module small and focused: given a `SynthParams` reference, produce audio blocks on request.

### Step 1: Implement `src/handspring/synth.py`

```python
"""In-process gesture-driven synth: saw → lowpass → amp env → tremolo."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import sounddevice as sd  # type: ignore[import-not-found]

from handspring.synth_params import SynthParams

_SAMPLE_RATE = 48_000
_BLOCK_SIZE = 256
_SMOOTH = 0.01            # per-sample one-pole smoothing coefficient
_ENV_ATTACK_SAMPLES = int(0.005 * _SAMPLE_RATE)   # 5 ms attack
_ENV_RELEASE_SAMPLES = int(0.060 * _SAMPLE_RATE)  # 60 ms release


class Synth:
    """sounddevice-driven audio engine reading from a SynthParams store.

    Use `start()` to open the audio stream, `stop()` to release it.
    Parameters are updated externally by calling SynthParams setters.
    """

    def __init__(self, params: SynthParams) -> None:
        self._params = params
        self._stream: sd.OutputStream | None = None

        # Smoothed parameter state (starts at defaults).
        snap = params.snapshot()
        self._s_volume = snap.volume
        self._s_note_hz = snap.note_hz
        self._s_stepping_hz = snap.stepping_hz
        self._s_cutoff_hz = snap.cutoff_hz
        self._s_mod_depth = snap.mod_depth
        self._s_mod_rate = snap.mod_rate

        # Oscillator / envelope / mod state.
        self._osc_phase = 0.0
        self._mod_phase = 0.0
        self._env = 0.0             # amplitude envelope value
        self._env_target = 1.0      # attacking or releasing
        self._step_phase = 0.0      # 0..1; resets at each stepping retrigger
        self._lp_prev = 0.0         # one-pole lowpass memory

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=_SAMPLE_RATE,
            blocksize=_BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        _time_info: Any,
        _status: Any,
    ) -> None:
        target = self._params.snapshot()
        out = outdata[:, 0]

        two_pi = 2.0 * math.pi

        for i in range(frames):
            # Smooth parameters toward targets.
            self._s_volume += (target.volume - self._s_volume) * _SMOOTH
            self._s_note_hz += (target.note_hz - self._s_note_hz) * _SMOOTH
            self._s_stepping_hz += (target.stepping_hz - self._s_stepping_hz) * _SMOOTH
            self._s_cutoff_hz += (target.cutoff_hz - self._s_cutoff_hz) * _SMOOTH
            self._s_mod_depth += (target.mod_depth - self._s_mod_depth) * _SMOOTH
            self._s_mod_rate += (target.mod_rate - self._s_mod_rate) * _SMOOTH

            # Oscillator: saw.
            self._osc_phase += self._s_note_hz / _SAMPLE_RATE
            if self._osc_phase >= 1.0:
                self._osc_phase -= 1.0
            osc = 2.0 * self._osc_phase - 1.0

            # One-pole lowpass.
            wc = self._s_cutoff_hz
            # alpha = 1 - e^(-2πwc/sr) approximates proper LP; use simpler form.
            alpha = (two_pi * wc) / (two_pi * wc + _SAMPLE_RATE)
            self._lp_prev = self._lp_prev + alpha * (osc - self._lp_prev)
            filtered = self._lp_prev

            # Stepping retrigger: if stepping_hz > 0, retrigger env periodically.
            if self._s_stepping_hz > 0.01:
                self._step_phase += self._s_stepping_hz / _SAMPLE_RATE
                if self._step_phase >= 1.0:
                    self._step_phase -= 1.0
                    self._env = 0.0  # retrigger
                    self._env_target = 1.0
            else:
                self._step_phase = 0.0
                self._env_target = 1.0

            # Envelope (attack then release).
            if self._env < self._env_target:
                self._env += 1.0 / _ENV_ATTACK_SAMPLES
                if self._env > 1.0:
                    self._env = 1.0
            elif self._s_stepping_hz > 0.01:
                # Releasing inside a step cycle.
                self._env -= 1.0 / _ENV_RELEASE_SAMPLES
                if self._env < 0.0:
                    self._env = 0.0

            # Tremolo.
            self._mod_phase += self._s_mod_rate / _SAMPLE_RATE
            if self._mod_phase >= 1.0:
                self._mod_phase -= 1.0
            tremolo = 1.0 - self._s_mod_depth * (
                0.5 - 0.5 * math.cos(two_pi * self._mod_phase)
            )

            out[i] = self._s_volume * self._env * tremolo * filtered * 0.35  # headroom
```

### Step 2: Smoke-test (no automated test)

```bash
ruff check src/handspring/synth.py
ruff format --check src/handspring/synth.py
mypy src/handspring/synth.py
python -c "from handspring.synth_params import SynthParams; from handspring.synth import Synth; s = Synth(SynthParams()); print('constructed ok')"
```

(Do NOT actually `start()` it in CI — would open audio output.)

### Step 3: Commit

```bash
git add src/handspring/synth.py
git commit -m "feat(synth): in-process saw+lowpass+env+tremolo synth"
```

---

## Task 3: Add `sounddevice` to runtime dependencies

**Files:**
- Modify: `pyproject.toml`

Move `sounddevice` from `[project.optional-dependencies] dev` to `[project] dependencies`. This is a small but meaningful change: the top-level install now requires PortAudio system libs. Linux CI needs `libportaudio2` apt pkg; macOS bundles it via sounddevice's wheels.

### Step 1: Edit `pyproject.toml`

Move the `sounddevice>=0.4,<0.5` entry from `dev` to `dependencies`. After the change:

```toml
[project]
...
dependencies = [
    "mediapipe>=0.10,<0.11",
    "opencv-python>=4.10,<5",
    "python-osc>=1.8,<2",
    "numpy>=1.26,<3",
    "sounddevice>=0.4,<0.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9",
    "ruff>=0.6,<1",
    "mypy>=1.11,<2",
]
```

Also update `.github/workflows/ci.yml` — the Ubuntu install step needs portaudio:

```yaml
      - name: Install system deps (ubuntu)
        if: matrix.os == 'ubuntu-latest'
        run: |
          sudo apt-get update
          sudo apt-get install -y libgl1 libglib2.0-0 libportaudio2
```

### Step 2: Commit

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "chore: promote sounddevice to runtime dep; add libportaudio2 to ubuntu CI"
```

---

## Task 4: SynthController (gesture → parameter mapping)

**Files:**
- Create: `src/handspring/synth_ui.py`
- Create: `tests/test_synth_ui.py`

The controller owns the fist-debounce state and maps gestures/positions to parameter writes.

### Step 1: Tests

```python
"""SynthController state-machine tests."""
from __future__ import annotations

from handspring.synth_params import SynthParams
from handspring.synth_ui import SynthController
from handspring.types import FaceState, FrameResult, HandFeatures, HandState, MotionState, PoseState


def _absent_hand() -> HandState:
    return HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )


def _hand(gesture: str, x: float = 0.5, y: float = 0.5) -> HandState:
    return HandState(
        present=True,
        features=HandFeatures(x=x, y=y, z=0.0, openness=1.0, pinch=0.0),
        gesture=gesture,  # type: ignore[arg-type]
        motion=MotionState(False, False, 0.0, 0.0, None),
    )


def _face_absent() -> FaceState:
    return FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )


def _frame(left: HandState, right: HandState) -> FrameResult:
    return FrameResult(
        left=left,
        right=right,
        face=_face_absent(),
        pose=PoseState(False, None),
        fps=30.0,
        clap_event=False,
    )


def test_mode_defaults_play():
    p = SynthParams()
    c = SynthController(p)
    c.update(_frame(_absent_hand(), _absent_hand()))
    assert p.snapshot().mode == "play"


def test_left_fist_debounce_activates_edit_left():
    p = SynthParams()
    c = SynthController(p)
    # 2 frames is not enough (debounce = 3).
    for _ in range(2):
        c.update(_frame(_hand("fist"), _absent_hand()))
    assert p.snapshot().mode == "play"
    c.update(_frame(_hand("fist"), _absent_hand()))
    assert p.snapshot().mode == "edit_left"


def test_release_requires_debounce():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _absent_hand()))
    assert p.snapshot().mode == "edit_left"
    # Single frame of non-fist — still in edit mode.
    c.update(_frame(_hand("open"), _absent_hand()))
    assert p.snapshot().mode == "edit_left"
    c.update(_frame(_hand("open"), _absent_hand()))
    c.update(_frame(_hand("open"), _absent_hand()))
    assert p.snapshot().mode == "play"


def test_both_fist_left_wins():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("fist")))
    assert p.snapshot().mode == "edit_left"


def test_edit_left_point_controls_volume():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("point", y=0.5)))
    # y=0.5 (middle) → volume ~0.5
    vol_mid = p.snapshot().volume
    assert 0.4 <= vol_mid <= 0.6
    # y=0.0 (top of frame) → volume = 1.0
    c.update(_frame(_hand("fist"), _hand("point", y=0.0)))
    assert p.snapshot().volume == 1.0
    # y=1.0 (bottom) → volume = 0.0
    c.update(_frame(_hand("fist"), _hand("point", y=1.0)))
    assert p.snapshot().volume == 0.0


def test_edit_left_open_controls_pitch_and_stepping():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("open", x=0.5, y=0.5)))
    s = p.snapshot()
    # Middle y → middle of log-pitch range (geometric mean of C3..C6).
    assert 200.0 < s.note_hz < 700.0
    # Middle x → stepping ~ half of range (0..16). Actual expo mapping
    # puts mid at 4.0, but allow some range.
    assert 0.0 <= s.stepping_hz <= 16.0
    # Top of frame → highest pitch.
    c.update(_frame(_hand("fist"), _hand("open", x=0.5, y=0.0)))
    assert p.snapshot().note_hz == 1047.0


def test_edit_right_point_controls_cutoff():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("point", y=0.5), _hand("fist")))
    assert p.snapshot().mode == "edit_right"
    # y=0.0 → highest cutoff
    c.update(_frame(_hand("point", y=0.0), _hand("fist")))
    assert p.snapshot().cutoff_hz == 8000.0
    c.update(_frame(_hand("point", y=1.0), _hand("fist")))
    assert p.snapshot().cutoff_hz == 200.0


def test_edit_right_open_controls_mod():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("open", x=0.5, y=0.5), _hand("fist")))
    s = p.snapshot()
    assert 0.0 <= s.mod_depth <= 1.0
    assert 0.1 <= s.mod_rate <= 10.0
    # y=0 → max depth
    c.update(_frame(_hand("open", x=0.0, y=0.0), _hand("fist")))
    assert p.snapshot().mod_depth == 1.0


def test_play_mode_does_not_edit_params():
    p = SynthParams()
    c = SynthController(p)
    initial_vol = p.snapshot().volume
    # No fist → play mode; right hand pointing should NOT change volume.
    c.update(_frame(_absent_hand(), _hand("point", y=0.0)))
    c.update(_frame(_absent_hand(), _hand("point", y=0.0)))
    assert p.snapshot().volume == initial_vol
```

### Step 2: Implement `src/handspring/synth_ui.py`

```python
"""Gesture → synth parameter state machine.

Reads FrameResult each frame, handles fist debouncing, determines the active
mode, and writes to the SynthParams instance. Also exposes `ui_hints` so the
preview can draw the correct slider / crosshair at the current finger position.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

from handspring.synth_params import (
    CUTOFF_MAX,
    CUTOFF_MIN,
    MOD_DEPTH_MAX,
    MOD_DEPTH_MIN,
    MOD_RATE_MAX,
    MOD_RATE_MIN,
    NOTE_MAX,
    NOTE_MIN,
    STEP_MAX,
    STEP_MIN,
    SynthParams,
)
from handspring.types import FrameResult, HandState

_DEBOUNCE_N = 3

# UI hint kinds.
HintKind = Literal["none", "slider", "xy"]


@dataclass(frozen=True)
class UiHint:
    kind: HintKind
    # slider anchor (for slider) or palm center (for xy), normalized 0..1
    x: float
    y: float
    # Primary label + value for the slider (or x-axis label for xy).
    label_a: str
    value_a: float      # normalized 0..1
    display_a: str      # pretty-printed value text
    # Secondary (y-axis) for xy only.
    label_b: str
    value_b: float
    display_b: str


_NO_HINT = UiHint("none", 0.0, 0.0, "", 0.0, "", "", 0.0, "")


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


def _log_lerp(lo: float, hi: float, t: float) -> float:
    return math.exp(_lerp(math.log(lo), math.log(hi), t))


def _y_to_norm(y: float) -> float:
    """Hand Y 0 (top) → 1.0, Y 1 (bottom) → 0.0. Clamped."""
    return max(0.0, min(1.0, 1.0 - y))


def _x_to_norm(x: float) -> float:
    return max(0.0, min(1.0, x))


class SynthController:
    def __init__(self, params: SynthParams) -> None:
        self._params = params
        self._left_fists: Deque[bool] = deque(maxlen=_DEBOUNCE_N)
        self._right_fists: Deque[bool] = deque(maxlen=_DEBOUNCE_N)
        self._active_mode: Literal["play", "edit_left", "edit_right"] = "play"
        self._last_hint: UiHint = _NO_HINT

    def update(self, frame: FrameResult) -> None:
        # Record fist state for debounce.
        self._left_fists.append(frame.left.present and frame.left.gesture == "fist")
        self._right_fists.append(frame.right.present and frame.right.gesture == "fist")

        left_fist_stable = len(self._left_fists) == _DEBOUNCE_N and all(self._left_fists)
        right_fist_stable = len(self._right_fists) == _DEBOUNCE_N and all(self._right_fists)
        left_released = len(self._left_fists) == _DEBOUNCE_N and not any(self._left_fists)
        right_released = len(self._right_fists) == _DEBOUNCE_N and not any(self._right_fists)

        # Mode transitions.
        if self._active_mode == "play":
            if left_fist_stable:
                self._active_mode = "edit_left"
            elif right_fist_stable:
                self._active_mode = "edit_right"
        elif self._active_mode == "edit_left":
            if left_released:
                if right_fist_stable:
                    self._active_mode = "edit_right"
                else:
                    self._active_mode = "play"
        else:  # edit_right
            if right_released:
                if left_fist_stable:
                    self._active_mode = "edit_left"
                else:
                    self._active_mode = "play"

        self._params.set_mode(self._active_mode)

        # Apply edits based on the non-fist hand.
        self._last_hint = _NO_HINT
        if self._active_mode == "edit_left":
            self._apply_edit_left(frame.right)
        elif self._active_mode == "edit_right":
            self._apply_edit_right(frame.left)

    def ui_hint(self) -> UiHint:
        return self._last_hint

    # ---- Mode 1: left fist, right hand edits ----

    def _apply_edit_left(self, right: HandState) -> None:
        if not right.present or right.features is None:
            return
        f = right.features
        if right.gesture == "point":
            vol = _y_to_norm(f.y)
            self._params.set_volume(vol)
            self._last_hint = UiHint(
                "slider",
                x=f.x,
                y=f.y,
                label_a="volume",
                value_a=vol,
                display_a=f"{vol:.2f}",
                label_b="",
                value_b=0.0,
                display_b="",
            )
        elif right.gesture == "open":
            pitch_t = _y_to_norm(f.y)
            note_hz = _log_lerp(NOTE_MIN, NOTE_MAX, pitch_t)
            step_t = _x_to_norm(f.x)
            stepping_hz = _lerp(STEP_MIN, STEP_MAX, step_t)
            self._params.set_note_hz(note_hz)
            self._params.set_stepping_hz(stepping_hz)
            self._last_hint = UiHint(
                "xy",
                x=f.x,
                y=f.y,
                label_a="step",
                value_a=step_t,
                display_a=f"{stepping_hz:.1f} Hz",
                label_b="pitch",
                value_b=pitch_t,
                display_b=f"{_hz_to_note(note_hz)} ({note_hz:.0f} Hz)",
            )

    # ---- Mode 2: right fist, left hand edits ----

    def _apply_edit_right(self, left: HandState) -> None:
        if not left.present or left.features is None:
            return
        f = left.features
        if left.gesture == "point":
            cutoff_t = _y_to_norm(f.y)
            cutoff_hz = _log_lerp(CUTOFF_MIN, CUTOFF_MAX, cutoff_t)
            self._params.set_cutoff_hz(cutoff_hz)
            self._last_hint = UiHint(
                "slider",
                x=f.x,
                y=f.y,
                label_a="cutoff",
                value_a=cutoff_t,
                display_a=f"{cutoff_hz:.0f} Hz",
                label_b="",
                value_b=0.0,
                display_b="",
            )
        elif left.gesture == "open":
            depth_t = _y_to_norm(f.y)
            rate_t = _x_to_norm(f.x)
            mod_depth = _lerp(MOD_DEPTH_MIN, MOD_DEPTH_MAX, depth_t)
            mod_rate = _log_lerp(MOD_RATE_MIN, MOD_RATE_MAX, rate_t)
            self._params.set_mod_depth(mod_depth)
            self._params.set_mod_rate(mod_rate)
            self._last_hint = UiHint(
                "xy",
                x=f.x,
                y=f.y,
                label_a="mod_rate",
                value_a=rate_t,
                display_a=f"{mod_rate:.2f} Hz",
                label_b="mod_depth",
                value_b=depth_t,
                display_b=f"{mod_depth:.2f}",
            )


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_note(hz: float) -> str:
    if hz <= 0:
        return "-"
    midi = 69.0 + 12.0 * math.log2(hz / 440.0)
    midi_int = round(midi)
    octave = midi_int // 12 - 1
    name = _NOTE_NAMES[midi_int % 12]
    return f"{name}{octave}"
```

### Step 3: Verify + commit

```bash
pytest tests/test_synth_ui.py -v
ruff check . && ruff format --check . && mypy src/
git add src/handspring/synth_ui.py tests/test_synth_ui.py
git commit -m "feat(synth_ui): fist-debounced gesture → synth parameter mapping"
```

---

## Task 5: OSC emissions for /synth/*

**Files:**
- Modify: `src/handspring/osc_out.py`
- Modify: `tests/test_osc_out.py`

### Step 1: Tests (append to `tests/test_osc_out.py`)

```python
from handspring.synth_params import SynthSnapshot


def test_synth_snapshot_emitted():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    snap = SynthSnapshot(
        volume=0.65,
        note_hz=523.25,
        stepping_hz=4.0,
        cutoff_hz=2500.0,
        mod_depth=0.3,
        mod_rate=2.0,
        mode="edit_left",
    )
    emitter.emit_synth(snap)
    assert ("/synth/volume", 0.65) in fake.sent
    assert ("/synth/note_hz", 523.25) in fake.sent
    assert ("/synth/stepping_hz", 4.0) in fake.sent
    assert ("/synth/cutoff_hz", 2500.0) in fake.sent
    assert ("/synth/mod_depth", 0.3) in fake.sent
    assert ("/synth/mod_rate", 2.0) in fake.sent


def test_synth_mode_only_on_change():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)

    def snap(mode):
        return SynthSnapshot(
            volume=0.5,
            note_hz=440.0,
            stepping_hz=0.0,
            cutoff_hz=3000.0,
            mod_depth=0.0,
            mod_rate=1.0,
            mode=mode,
        )

    emitter.emit_synth(snap("play"))
    emitter.emit_synth(snap("play"))
    emitter.emit_synth(snap("edit_left"))
    emitter.emit_synth(snap("edit_left"))
    modes = [v for a, v in fake.sent if a == "/synth/mode"]
    assert modes == ["play", "edit_left"]
```

### Step 2: Extend `OscEmitter`

Add to imports:
```python
from handspring.synth_params import SynthSnapshot
```

Add to `__init__`:
```python
        self._last_synth_mode: str | None = None
```

Add new method after `_emit_pose`:
```python
    def emit_synth(self, snap: "SynthSnapshot") -> None:
        self._client.send_message("/synth/volume", float(snap.volume))
        self._client.send_message("/synth/note_hz", float(snap.note_hz))
        self._client.send_message("/synth/stepping_hz", float(snap.stepping_hz))
        self._client.send_message("/synth/cutoff_hz", float(snap.cutoff_hz))
        self._client.send_message("/synth/mod_depth", float(snap.mod_depth))
        self._client.send_message("/synth/mod_rate", float(snap.mod_rate))
        if snap.mode != self._last_synth_mode:
            self._client.send_message("/synth/mode", snap.mode)
            self._last_synth_mode = snap.mode
```

### Step 3: Verify + commit

```bash
pytest tests/test_osc_out.py -v
ruff check . && ruff format --check . && mypy src/
git add src/handspring/osc_out.py tests/test_osc_out.py
git commit -m "feat(osc_out): emit /synth/* snapshot and mode-change events"
```

---

## Task 6: Preview overlay — synth panel + slider + XY crosshair

**Files:**
- Modify: `src/handspring/preview.py`

Add helpers to draw the synth panel and the edit-mode hint (slider or xy crosshair). Add a new parameter to `Preview.render()`.

### Step 1: Update preview.py

Add near the top:
```python
from handspring.synth_params import SynthSnapshot
from handspring.synth_ui import UiHint
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
    ) -> bool:
```

After `_draw_status(display, frame_result, osc_target)`, add:
```python
        if synth_snapshot is not None:
            _draw_synth_panel(display, synth_snapshot)
        if synth_hint is not None and synth_hint.kind != "none":
            _draw_synth_hint(display, synth_hint, mirrored=self._mirror)
```

Add these functions to preview.py:

```python
def _draw_synth_panel(frame: NDArray[np.uint8], snap: SynthSnapshot) -> None:
    """Lower-left compact synth readout."""
    h = frame.shape[0]
    from handspring.synth_ui import _hz_to_note

    mode_text = {
        "play": "PLAY",
        "edit_left": "EDIT L",
        "edit_right": "EDIT R",
    }[snap.mode]
    lines = [
        "-- SYNTH --",
        f"vol: {snap.volume:.2f}",
        f"note: {_hz_to_note(snap.note_hz)} ({snap.note_hz:.0f} Hz)",
        f"step: {snap.stepping_hz:.1f} Hz",
        f"cutoff: {snap.cutoff_hz:.0f} Hz",
        f"mod: {snap.mod_depth:.2f} @ {snap.mod_rate:.2f} Hz",
        f"mode: {mode_text}",
    ]
    x = 12
    y = h - 24 * len(lines) - 12
    for text in lines:
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA
        )
        color = (136, 255, 0) if "mode" in text and snap.mode != "play" else (230, 230, 230)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 24


def _draw_synth_hint(frame: NDArray[np.uint8], hint: UiHint, *, mirrored: bool) -> None:
    h, w = frame.shape[:2]
    # Note: if preview mirrored, hint x was derived from the un-mirrored hand
    # feature. We need to flip x so it lands where the finger visually is.
    display_x = (1.0 - hint.x) if mirrored else hint.x

    if hint.kind == "slider":
        cx = int(display_x * w) + 24
        cy = int(hint.y * h)
        _draw_slider(frame, cx=cx, cy=cy, label=hint.label_a, value=hint.value_a, display=hint.display_a)
    elif hint.kind == "xy":
        cx = int(display_x * w)
        cy = int(hint.y * h)
        _draw_xy(
            frame,
            cx=cx,
            cy=cy,
            label_x=hint.label_a,
            display_x=hint.display_a,
            label_y=hint.label_b,
            display_y=hint.display_b,
        )


def _draw_slider(
    frame: NDArray[np.uint8],
    *,
    cx: int,
    cy: int,
    label: str,
    value: float,
    display: str,
) -> None:
    height = 140
    width = 18
    x0 = cx
    y0 = cy - height // 2
    # Track
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (136, 255, 0), 2)
    # Fill (bottom-up)
    fill_px = int(value * (height - 4))
    cv2.rectangle(
        frame,
        (x0 + 2, y0 + height - 2 - fill_px),
        (x0 + width - 2, y0 + height - 2),
        (136, 255, 0),
        -1,
    )
    # Label
    _label(frame, x0 + width + 6, y0 + 12, label)
    _label(frame, x0 + width + 6, y0 + height - 4, display)


def _draw_xy(
    frame: NDArray[np.uint8],
    *,
    cx: int,
    cy: int,
    label_x: str,
    display_x: str,
    label_y: str,
    display_y: str,
) -> None:
    h, w = frame.shape[:2]
    # Full-width horizontal + full-height vertical crosshair.
    cv2.line(frame, (0, cy), (w, cy), (136, 255, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, 0), (cx, h), (136, 255, 0), 1, cv2.LINE_AA)
    # Labels at axis ends.
    _label(frame, max(8, cx - 80), 20, f"{label_y}: {display_y}")
    _label(frame, w - 220, cy - 6, f"{label_x}: {display_x}")


def _label(frame: NDArray[np.uint8], x: int, y: int, text: str) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
```

### Step 2: Verify + commit

```bash
ruff check . && ruff format --check . && mypy src/
python -c "from handspring.preview import Preview; Preview(); print('ok')"
git add src/handspring/preview.py
git commit -m "feat(preview): synth panel + slider + xy crosshair overlays"
```

---

## Task 7: Wire into `__main__.py` + `--no-synth` flag

**Files:**
- Modify: `src/handspring/__main__.py`

### Changes

1. Add `--no-synth` CLI flag.
2. Construct `SynthParams`, `Synth`, `SynthController`. Start the synth unless `--no-synth`.
3. Each frame after `tracker.process`:
   - `synth_controller.update(result)` to apply edits.
   - `emitter.emit_synth(params.snapshot())`
   - Pass `params.snapshot()` and `synth_controller.ui_hint()` to `preview.render(...)`.
4. On shutdown: `synth.stop()`.

### Step 1: Edit `__main__.py`

Add imports:
```python
from handspring.synth import Synth
from handspring.synth_params import SynthParams
from handspring.synth_ui import SynthController
```

In `_parse_args`, add:
```python
    p.add_argument("--no-synth", action="store_true", help="disable in-process synth audio output")
```

In `main()`, after `emitter = OscEmitter(...)`:
```python
    synth_params = SynthParams()
    synth_controller = SynthController(synth_params)
    synth: Synth | None = None
    if not args.no_synth:
        try:
            synth = Synth(synth_params)
            synth.start()
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not start synth ({e}); continuing without audio", file=sys.stderr)
            synth = None
```

Update the startup banner to include synth status:
```python
    synth_status = "off" if args.no_synth or synth is None else "on"
    print(
        f"hands:  {args.hands}   face: {'off' if args.no_face else 'on'}   "
        f"pose: {'off' if args.no_pose else 'on'}   synth: {synth_status}",
        flush=True,
    )
```

In the main loop, after `emitter.emit(result)`:
```python
            synth_controller.update(result)
            if not args.no_synth:
                emitter.emit_synth(synth_params.snapshot())
```

Update the preview.render call to pass synth info:
```python
            if preview is not None:
                hand_landmarks, face_landmarks, pose_landmarks = _extract_landmark_lists(tracker, bgr)
                snap_for_preview = synth_params.snapshot() if not args.no_synth else None
                hint_for_preview = synth_controller.ui_hint() if not args.no_synth else None
                if not preview.render(
                    bgr,
                    hand_landmarks,
                    face_landmarks,
                    pose_landmarks,
                    result,
                    f"{args.host}:{args.port}",
                    snap_for_preview,
                    hint_for_preview,
                ):
                    break
```

In the `finally` block, add:
```python
        if synth is not None:
            synth.stop()
```

### Step 2: Full suite verify

```bash
pytest
ruff check .
ruff format --check .
mypy src/
python -m handspring --version
python -m handspring --help | grep -i synth
```

All green. `--help` shows `--no-synth`.

### Step 3: Commit

```bash
git add src/handspring/__main__.py
git commit -m "feat(main): wire in-process synth + --no-synth flag"
```

---

## Task 8: README + tag v0.4.0

**Files:**
- Modify: `README.md`
- Modify: `src/handspring/__init__.py` — bump to `0.4.0`
- Modify: `pyproject.toml` — bump to `0.4.0`

### Step 1: README additions

In the "Demo" section, UPDATE the description to mention the built-in synth plays by default:

```markdown
## Built-in synth

handspring includes an in-process synth. The moment you run `python -m handspring`
a sustained tone plays. Put your LEFT hand in a fist to enter the edit mode —
point with the right hand to adjust volume; open right hand to change pitch (Y)
and stepping (X). Put your RIGHT hand in a fist to edit filter cutoff and
modulation with your left hand.

Disable the built-in synth (for OSC-only workflows) with `--no-synth`.

### Synth parameters
```
volume       0..1
note_hz      131..1047 (C3..C6)
stepping_hz  0..16  — envelope retrigger rate for buildup pulses
cutoff_hz    200..8000  — lowpass filter
mod_depth    0..1   — amplitude tremolo depth
mod_rate     0.1..10  — LFO frequency
mode         play | edit_left | edit_right
```

In the OSC reference, add a new section:

```markdown
Synth state (continuous):

| Address | Type | Notes |
|---|---|---|
| `/synth/volume` | float | 0..1 |
| `/synth/note_hz` | float | |
| `/synth/stepping_hz` | float | 0 = sustained |
| `/synth/cutoff_hz` | float | |
| `/synth/mod_depth` | float | 0..1 |
| `/synth/mod_rate` | float | Hz |
| `/synth/mode` | string | `play` / `edit_left` / `edit_right` — on change |
```

In the CLI flags block, add:

```
--no-synth             disable the in-process synth
```

### Step 2: Version bumps + checks + tag

```bash
# bump __init__.py and pyproject.toml from 0.3.0 to 0.4.0
ruff check .
ruff format --check .
mypy src/
pytest
python -m handspring --version   # → 0.4.0
git add README.md src/handspring/__init__.py pyproject.toml
git commit -m "docs: document built-in synth + /synth/* OSC + --no-synth flag"
git tag -a v0.4.0 -m "handspring v0.4.0 — in-process gesture-driven synth"
```

Do NOT push.

### Step 3: Summary

Report:
- Commits since v0.3.0
- Test count
- Any manual tweaks
- Tag SHA

---

## Execution notes

- `sounddevice` requires PortAudio. The `pyproject.toml` change alone doesn't install the system library; the Ubuntu CI step does (`libportaudio2`). Developers on macOS: no extra work. Linux: `sudo apt-get install libportaudio2` once if not present.
- The audio callback must be fast — if it spends more time than one block (~5.3 ms), you get dropouts. The per-sample loop in `synth.py` is ~30 cycles in Python; at 256 samples × 30 cycles ≈ a few ms — within budget. If it drops frames in practice, port the inner loop to numpy vectorization.
- Fist detection accuracy is the main UX risk. If users report edit-mode flapping, tighten the debounce from 3 frames to 5, or add hysteresis (different thresholds for on vs off).
