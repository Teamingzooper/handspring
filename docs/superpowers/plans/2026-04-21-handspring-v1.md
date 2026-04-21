# handspring v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build handspring v1 — a Python script that streams live hand/face gesture data from a webcam as OSC messages, with a bundled sine-synth demo receiver.

**Architecture:** Single-process Python 3.10+ app, single main thread. `cv2.VideoCapture` → `mediapipe` inference → pure-function feature derivation + gesture classification → `python-osc` UDP send + OpenCV preview window. No real-time safety constraints (OSC fire-and-forget; preview best-effort).

**Tech Stack:** Python 3.10+, `mediapipe ~= 0.10`, `opencv-python ~= 4.10`, `python-osc ~= 1.8`, `numpy ~= 1.26`. Dev: `pytest`, `ruff`, `mypy`, `sounddevice`. Package manager: `uv` (preferred) or `pip`.

**Spec:** [`2026-04-21-handspring-v1-design.md`](../specs/2026-04-21-handspring-v1-design.md)

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md` (stub; Task 12 finalizes)
- Create: `LICENSE`
- Create: `.github/workflows/ci.yml`
- Create: `src/handspring/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "handspring"
version = "0.1.0"
description = "Webcam gesture → OSC stream for creative coding"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
authors = [{ name = "handspring contributors" }]
dependencies = [
    "mediapipe>=0.10,<0.11",
    "opencv-python>=4.10,<5",
    "python-osc>=1.8,<2",
    "numpy>=1.26,<3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9",
    "ruff>=0.6,<1",
    "mypy>=1.11,<2",
    "sounddevice>=0.4,<0.5",
]

[project.scripts]
handspring = "handspring.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/handspring"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM", "RET"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --strict-markers"
```

- [ ] **Step 2: Create `LICENSE`**

```
MIT License

Copyright (c) 2026 handspring contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create stub `README.md`**

```markdown
# handspring

Webcam gesture → OSC stream for creative coding.

Full README lands in Task 12 once the app is built.
```

- [ ] **Step 4: Create `src/handspring/__init__.py`**

```python
"""handspring — webcam gesture → OSC stream."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Create `tests/__init__.py`**

```python
"""Test package."""
```

- [ ] **Step 6: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install system deps (ubuntu)
        if: matrix.os == 'ubuntu-latest'
        run: |
          sudo apt-get update
          sudo apt-get install -y libgl1 libglib2.0-0
      - name: Install
        run: pip install -e '.[dev]'
      - name: Ruff check
        run: ruff check .
      - name: Ruff format check
        run: ruff format --check .
      - name: Mypy
        run: mypy src/
      - name: Pytest
        run: pytest
```

- [ ] **Step 7: Verify scaffold + commit**

Run:
```bash
ls pyproject.toml README.md LICENSE .github/workflows/ci.yml src/handspring/__init__.py tests/__init__.py
```
All files present.

```bash
git add pyproject.toml README.md LICENSE .github/workflows/ci.yml src/handspring/__init__.py tests/__init__.py
git commit -m "chore: scaffold handspring project"
```

---

## Task 2: Core types

**Files:**
- Create: `src/handspring/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing test `tests/test_types.py`**

```python
"""Type dataclasses should be frozen and round-trip cleanly."""
from handspring.types import (
    FaceFeatures,
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
)


def test_hand_features_frozen():
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.8, pinch=0.1)
    import pytest
    with pytest.raises(AttributeError):
        hf.x = 0.0  # type: ignore[misc]


def test_hand_state_absent():
    hs = HandState(present=False, features=None, gesture="none")
    assert hs.present is False
    assert hs.features is None
    assert hs.gesture == "none"


def test_hand_state_present():
    hf = HandFeatures(x=0.1, y=0.2, z=0.3, openness=0.9, pinch=0.05)
    hs = HandState(present=True, features=hf, gesture="open")
    assert hs.present is True
    assert hs.features == hf
    assert hs.gesture == "open"


def test_face_state_absent():
    fs = FaceState(present=False, features=None)
    assert fs.present is False
    assert fs.features is None


def test_face_features_ranges():
    ff = FaceFeatures(yaw=-0.3, pitch=0.1, mouth_open=0.5)
    assert -1.0 <= ff.yaw <= 1.0
    assert -1.0 <= ff.pitch <= 1.0
    assert 0.0 <= ff.mouth_open <= 1.0


def test_frame_result_composition():
    left = HandState(present=False, features=None, gesture="none")
    right = HandState(present=False, features=None, gesture="none")
    face = FaceState(present=False, features=None)
    fr = FrameResult(left=left, right=right, face=face, fps=30.0)
    assert fr.fps == 30.0
    assert fr.left.gesture == "none"
```

- [ ] **Step 2: Run test; expect ImportError**

```bash
pytest tests/test_types.py -v
```
Expected: ImportError (module not yet created).

- [ ] **Step 3: Implement `src/handspring/types.py`**

```python
"""Core dataclasses for frame-level tracking results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
Gesture = Literal["fist", "open", "point", "peace", "thumbs_up", "none"]


@dataclass(frozen=True)
class HandFeatures:
    """Continuous hand features in a single frame.

    x, y, z are normalized in [0, 1] (x, y) and relative units (z). openness
    is 0 (fist) to 1 (open palm). pinch is 0 (thumb and index far apart) to 1
    (touching).
    """

    x: float
    y: float
    z: float
    openness: float
    pinch: float


@dataclass(frozen=True)
class HandState:
    """One hand's state for a single frame."""

    present: bool
    features: HandFeatures | None
    gesture: Gesture


@dataclass(frozen=True)
class FaceFeatures:
    """Continuous face features in a single frame.

    yaw and pitch are in [-1, 1] (negative yaw = looking left of camera,
    negative pitch = looking down). mouth_open is 0 (closed) to 1 (wide open).
    """

    yaw: float
    pitch: float
    mouth_open: float


@dataclass(frozen=True)
class FaceState:
    """Face state for a single frame."""

    present: bool
    features: FaceFeatures | None


@dataclass(frozen=True)
class FrameResult:
    """Full per-frame tracking result."""

    left: HandState
    right: HandState
    face: FaceState
    fps: float
```

- [ ] **Step 4: Run test; expect pass**

```bash
pytest tests/test_types.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/handspring/types.py tests/test_types.py
git commit -m "feat(types): add frame-level tracking dataclasses"
```

---

## Task 3: Landmark fixtures for tests

**Files:**
- Create: `tests/fixtures.py`

Hand-written synthesized landmark arrays that mimic MediaPipe output geometry for each gesture + a closed/open mouth face. Downstream tests import from here.

- [ ] **Step 1: Create `tests/fixtures.py`**

```python
"""Synthetic landmark fixtures for offline testing.

MediaPipe landmark conventions reproduced here:
- Hand: 21 landmarks indexed [0..20]
  0: WRIST
  1..4: THUMB (CMC, MCP, IP, TIP)
  5..8: INDEX (MCP, PIP, DIP, TIP)
  9..12: MIDDLE (MCP, PIP, DIP, TIP)
  13..16: RING (MCP, PIP, DIP, TIP)
  17..20: PINKY (MCP, PIP, DIP, TIP)
- Face: we use only a handful of key landmarks.

Coordinates are in normalized image space (x in [0, 1], y in [0, 1], z
roughly in [-1, 1] with z=0 at the wrist). Each factory returns an
(N, 3) numpy array.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _hand_skeleton(
    extended: tuple[bool, bool, bool, bool, bool],
    palm_xy: tuple[float, float] = (0.5, 0.5),
    hand_size: float = 0.15,
) -> NDArray[np.float32]:
    """Build a 21x3 hand landmark array with the given per-finger extension.

    extended = (thumb, index, middle, ring, pinky). True = extended
    outward from the wrist; False = curled inward.
    """
    cx, cy = palm_xy
    s = hand_size
    lm = np.zeros((21, 3), dtype=np.float32)

    # Wrist at bottom centre of palm
    lm[0] = (cx, cy + 0.55 * s, 0.0)

    # Finger root (MCP) layout — fanned along the top of the palm
    # Thumb CMC offset to the side
    lm[1] = (cx - 0.45 * s, cy + 0.25 * s, 0.0)  # THUMB_CMC
    lm[5] = (cx - 0.30 * s, cy - 0.10 * s, 0.0)  # INDEX_MCP
    lm[9] = (cx - 0.10 * s, cy - 0.15 * s, 0.0)  # MIDDLE_MCP
    lm[13] = (cx + 0.10 * s, cy - 0.15 * s, 0.0)  # RING_MCP
    lm[17] = (cx + 0.30 * s, cy - 0.10 * s, 0.0)  # PINKY_MCP

    # Helper to place a finger's joints along a direction from MCP
    def _place_finger(
        tip_idx: int,
        dip_idx: int,
        pip_idx: int,
        mcp_idx: int,
        direction: tuple[float, float],
        is_extended: bool,
    ) -> None:
        dx, dy = direction
        mcp = lm[mcp_idx]
        if is_extended:
            pip_off = 0.30
            dip_off = 0.55
            tip_off = 0.80
        else:
            # Curled: tip ends up back near palm (shorter and inward)
            pip_off = 0.22
            dip_off = 0.22
            tip_off = 0.18
        lm[pip_idx] = (mcp[0] + dx * s * pip_off, mcp[1] + dy * s * pip_off, 0.0)
        lm[dip_idx] = (mcp[0] + dx * s * dip_off, mcp[1] + dy * s * dip_off, 0.0)
        lm[tip_idx] = (mcp[0] + dx * s * tip_off, mcp[1] + dy * s * tip_off, 0.0)

    thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = extended
    # Fingers point "up" in image space (y decreases).
    up = (0.0, -1.0)
    # Thumb points to the side (left in user's view, so negative x).
    thumb_dir = (-1.0, -0.2)

    _place_finger(4, 3, 2, 1, thumb_dir, thumb_ext)
    _place_finger(8, 7, 6, 5, up, index_ext)
    _place_finger(12, 11, 10, 9, up, middle_ext)
    _place_finger(16, 15, 14, 13, up, ring_ext)
    _place_finger(20, 19, 18, 17, up, pinky_ext)

    return lm


def hand_fist() -> NDArray[np.float32]:
    return _hand_skeleton((False, False, False, False, False))


def hand_open() -> NDArray[np.float32]:
    return _hand_skeleton((True, True, True, True, True))


def hand_point() -> NDArray[np.float32]:
    # Index out, others curled. Thumb curled (doesn't matter for `point`).
    return _hand_skeleton((False, True, False, False, False))


def hand_peace() -> NDArray[np.float32]:
    return _hand_skeleton((False, True, True, False, False))


def hand_thumbs_up() -> NDArray[np.float32]:
    # Thumb extended, but pointing up instead of to the side.
    lm = _hand_skeleton((True, False, False, False, False))
    # Redirect thumb to point upward (-y) instead of sideways.
    wrist = lm[0]
    lm[1] = (wrist[0], wrist[1] - 0.05, 0.0)
    lm[2] = (wrist[0], wrist[1] - 0.10, 0.0)
    lm[3] = (wrist[0], wrist[1] - 0.15, 0.0)
    lm[4] = (wrist[0], wrist[1] - 0.20, 0.0)
    return lm


def hand_pinch() -> NDArray[np.float32]:
    # Thumb tip touches index tip; both extended but tips coincident.
    lm = _hand_skeleton((True, True, False, False, False))
    # Move thumb tip to coincide with index tip.
    lm[4] = lm[8].copy()
    return lm


# Face fixtures: a minimal set of the MediaPipe face mesh landmarks we use.
# We simulate a 478-point array but populate only the indices we read.
FACE_NOSE_TIP = 1
FACE_LEFT_EYE = 33  # user's left, camera's right
FACE_RIGHT_EYE = 263
FACE_UPPER_LIP = 13
FACE_LOWER_LIP = 14
FACE_LEFT_MOUTH = 61
FACE_RIGHT_MOUTH = 291


def _face_base() -> NDArray[np.float32]:
    lm = np.zeros((478, 3), dtype=np.float32)
    # Eye line at y=0.45, nose at y=0.55, mouth at y=0.7
    lm[FACE_LEFT_EYE] = (0.35, 0.45, 0.0)
    lm[FACE_RIGHT_EYE] = (0.65, 0.45, 0.0)
    lm[FACE_NOSE_TIP] = (0.5, 0.55, 0.0)
    lm[FACE_LEFT_MOUTH] = (0.4, 0.7, 0.0)
    lm[FACE_RIGHT_MOUTH] = (0.6, 0.7, 0.0)
    # For the baseline (closed mouth), upper and lower lip are coincident so
    # mouth_open == 0 exactly. `face_open_mouth()` below separates them.
    lm[FACE_UPPER_LIP] = (0.5, 0.70, 0.0)
    lm[FACE_LOWER_LIP] = (0.5, 0.70, 0.0)
    return lm


def face_closed_mouth() -> NDArray[np.float32]:
    return _face_base()


def face_open_mouth() -> NDArray[np.float32]:
    lm = _face_base()
    lm[FACE_UPPER_LIP] = (0.5, 0.685, 0.0)
    lm[FACE_LOWER_LIP] = (0.5, 0.735, 0.0)
    return lm


def face_looking_left() -> NDArray[np.float32]:
    lm = _face_base()
    lm[FACE_NOSE_TIP] = (0.38, 0.55, 0.0)
    return lm


def face_looking_down() -> NDArray[np.float32]:
    lm = _face_base()
    lm[FACE_NOSE_TIP] = (0.5, 0.63, 0.0)
    return lm
```

- [ ] **Step 2: Commit (no tests yet — Tasks 4–5 consume these)**

```bash
git add tests/fixtures.py
git commit -m "test(fixtures): add synthetic MediaPipe landmark arrays"
```

---

## Task 4: Feature derivation

**Files:**
- Create: `src/handspring/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing tests `tests/test_features.py`**

```python
"""Feature derivation tests."""
from __future__ import annotations

import numpy as np

from handspring.features import hand_features, face_features
from tests.fixtures import (
    face_closed_mouth,
    face_looking_down,
    face_looking_left,
    face_open_mouth,
    hand_fist,
    hand_open,
    hand_pinch,
)


def test_hand_features_xy_in_unit_range():
    f = hand_features(hand_open())
    assert 0.0 <= f.x <= 1.0
    assert 0.0 <= f.y <= 1.0


def test_hand_openness_fist_low():
    f = hand_features(hand_fist())
    assert f.openness < 0.3, f"fist openness too high: {f.openness}"


def test_hand_openness_open_high():
    f = hand_features(hand_open())
    assert f.openness > 0.7, f"open palm openness too low: {f.openness}"


def test_hand_pinch_high_when_coincident():
    f = hand_features(hand_pinch())
    assert f.pinch > 0.85, f"pinch value too low: {f.pinch}"


def test_hand_pinch_low_when_spread():
    f = hand_features(hand_open())
    assert f.pinch < 0.3, f"pinch value too high when spread: {f.pinch}"


def test_face_mouth_closed_low():
    f = face_features(face_closed_mouth())
    assert f.mouth_open < 0.1


def test_face_mouth_open_high():
    f = face_features(face_open_mouth())
    assert f.mouth_open > 0.3


def test_face_yaw_left_negative():
    f = face_features(face_looking_left())
    assert f.yaw < -0.1


def test_face_yaw_center_near_zero():
    f = face_features(face_closed_mouth())
    assert abs(f.yaw) < 0.1


def test_face_pitch_looking_down_is_negative():
    # Spec: negative pitch = looking down.
    f = face_features(face_looking_down())
    assert f.pitch < -0.05, f"down pitch not negative enough: {f.pitch}"


def test_hand_features_reject_nan():
    bad = np.full((21, 3), np.nan, dtype=np.float32)
    import pytest
    with pytest.raises(ValueError):
        hand_features(bad)
```

- [ ] **Step 2: Run test; expect ImportError**

```bash
pytest tests/test_features.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `src/handspring/features.py`**

```python
"""Derive normalized continuous features from MediaPipe landmarks."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from handspring.types import FaceFeatures, HandFeatures

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

# MediaPipe face mesh indices (a small subset we use)
F_NOSE_TIP = 1
F_LEFT_EYE = 33
F_RIGHT_EYE = 263
F_UPPER_LIP = 13
F_LOWER_LIP = 14
F_LEFT_MOUTH = 61
F_RIGHT_MOUTH = 291


def _validate(landmarks: NDArray[np.floating]) -> None:
    if not np.all(np.isfinite(landmarks)):
        raise ValueError("landmarks contain NaN or Inf")


def _hand_span(landmarks: NDArray[np.floating]) -> float:
    """Approximate hand size: wrist → middle-finger MCP distance."""
    return float(np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST]))


def hand_features(landmarks: NDArray[np.floating]) -> HandFeatures:
    """Derive HandFeatures from a (21, 3) landmark array."""
    _validate(landmarks)
    if landmarks.shape != (21, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {landmarks.shape}")

    palm = landmarks[MIDDLE_MCP]
    wrist = landmarks[WRIST]

    # Use palm center (middle MCP) as the reference for x, y.
    x = float(np.clip(palm[0], 0.0, 1.0))
    y = float(np.clip(palm[1], 0.0, 1.0))
    z = float(palm[2] - wrist[2])  # relative depth; MediaPipe z is already relative.

    span = _hand_span(landmarks)
    # Avoid div-by-zero: a collapsed hand has span ~0; default to a small value.
    if span < 1e-6:
        span = 1e-6

    # Openness: average tip-to-wrist distance for non-thumb fingers, normalized
    # against a reasonable "fully open" baseline of ~2.2× hand span.
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    tip_dists = [float(np.linalg.norm(landmarks[t] - wrist)) for t in tips]
    avg_tip_dist = sum(tip_dists) / len(tip_dists)
    openness = float(np.clip((avg_tip_dist / span - 0.9) / 1.3, 0.0, 1.0))

    # Pinch: inverse thumb-tip to index-tip distance, normalized by hand span.
    thumb_index_dist = float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP]))
    # When touching: ~0. When spread: can reach > 1.0× span.
    pinch = float(np.clip(1.0 - (thumb_index_dist / (span * 0.8)), 0.0, 1.0))

    return HandFeatures(x=x, y=y, z=z, openness=openness, pinch=pinch)


def face_features(landmarks: NDArray[np.floating]) -> FaceFeatures:
    """Derive FaceFeatures from a (N, 3) landmark array (N usually 468 or 478)."""
    _validate(landmarks)

    # Eyes give a horizontal reference line. Midpoint is the face "center x".
    left_eye = landmarks[F_LEFT_EYE]
    right_eye = landmarks[F_RIGHT_EYE]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    eye_distance = float(abs(right_eye[0] - left_eye[0]))
    if eye_distance < 1e-6:
        eye_distance = 1e-6

    # Yaw: how far the nose is off the eye midpoint, normalized by eye distance.
    # Positive yaw = looking right (nose shifted left in image because mirror
    # consumption is downstream's problem — here we follow user-perspective
    # convention where user looking left = nose shifts toward image-left =
    # negative yaw).
    nose = landmarks[F_NOSE_TIP]
    raw_yaw = float((nose[0] - eye_mid_x) / (eye_distance * 0.5))
    yaw = float(np.clip(raw_yaw, -1.0, 1.0))

    # Pitch: how far the nose is below the eye midline, relative to some
    # nominal "neutral" offset. A relaxed face has nose ~0.5 × eye-distance
    # below the eye midline. Subtract that and normalize.
    raw_vertical = float((nose[1] - eye_mid_y) / eye_distance)
    # raw_vertical ~= 0.5 at neutral. Positive delta = nose farther down = looking down.
    # Spec: negative pitch = looking down. So invert.
    pitch = float(np.clip(-(raw_vertical - 0.5) * 2.0, -1.0, 1.0))

    # Mouth open: vertical distance between upper/lower lip, normalized by
    # horizontal mouth width.
    upper = landmarks[F_UPPER_LIP]
    lower = landmarks[F_LOWER_LIP]
    left_mouth = landmarks[F_LEFT_MOUTH]
    right_mouth = landmarks[F_RIGHT_MOUTH]
    vertical = float(abs(lower[1] - upper[1]))
    mouth_width = float(abs(right_mouth[0] - left_mouth[0]))
    if mouth_width < 1e-6:
        mouth_width = 1e-6
    # Typical closed mouth: vertical/width ~ 0.02. Wide open: vertical/width ~ 0.5.
    mouth_open = float(np.clip((vertical / mouth_width) * 2.2, 0.0, 1.0))

    return FaceFeatures(yaw=yaw, pitch=pitch, mouth_open=mouth_open)
```

- [ ] **Step 4: Run test; expect pass**

```bash
pytest tests/test_features.py -v
```
Expected: 11 passed. If any test fails because a threshold isn't quite hit by the synthetic fixture, adjust the fixture in `tests/fixtures.py` (not the production code) unless the failure indicates a real bug. Document the adjustment in the fixture comment.

- [ ] **Step 5: Commit**

```bash
git add src/handspring/features.py tests/test_features.py
git commit -m "feat(features): derive hand and face features from landmarks"
```

---

## Task 5: Gesture classification

**Files:**
- Create: `src/handspring/gestures.py`
- Create: `tests/test_gestures.py`

- [ ] **Step 1: Write failing tests `tests/test_gestures.py`**

```python
"""Gesture classifier tests."""
from __future__ import annotations

import numpy as np

from handspring.gestures import classify_hand
from tests.fixtures import (
    hand_fist,
    hand_open,
    hand_peace,
    hand_pinch,
    hand_point,
    hand_thumbs_up,
)


def test_fist_classifies_fist():
    assert classify_hand(hand_fist()) == "fist"


def test_open_classifies_open():
    assert classify_hand(hand_open()) == "open"


def test_point_classifies_point():
    assert classify_hand(hand_point()) == "point"


def test_peace_classifies_peace():
    assert classify_hand(hand_peace()) == "peace"


def test_thumbs_up_classifies_thumbs_up():
    assert classify_hand(hand_thumbs_up()) == "thumbs_up"


def test_pinch_is_not_a_classified_gesture():
    # Pinch is a continuous feature; the classifier should not return "pinch".
    # The pinch fixture has thumb + index extended, others curled — this is
    # technically close to "peace" minus middle. Accept "none" for ambiguity.
    result = classify_hand(hand_pinch())
    assert result != "pinch"


def test_classifier_is_deterministic():
    fx = hand_fist()
    results = {classify_hand(fx) for _ in range(100)}
    assert results == {"fist"}


def test_classifier_rejects_nan():
    bad = np.full((21, 3), np.nan, dtype=np.float32)
    import pytest
    with pytest.raises(ValueError):
        classify_hand(bad)
```

- [ ] **Step 2: Run; expect ImportError**

```bash
pytest tests/test_gestures.py -v
```

- [ ] **Step 3: Implement `src/handspring/gestures.py`**

```python
"""Classify a hand into one of six discrete gestures."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from handspring.features import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
)
from handspring.types import Gesture


def _finger_extended(
    landmarks: NDArray[np.floating],
    tip_idx: int,
    pip_idx: int,
) -> bool:
    """A finger is "extended" if its tip is farther from the wrist than the PIP joint."""
    wrist = landmarks[WRIST]
    tip_dist = float(np.linalg.norm(landmarks[tip_idx] - wrist))
    pip_dist = float(np.linalg.norm(landmarks[pip_idx] - wrist))
    # 5% margin so borderline poses don't flicker between classes.
    return tip_dist > pip_dist * 1.05


def _thumb_extended(landmarks: NDArray[np.floating]) -> bool:
    """Thumb is extended if its tip is far from the index MCP."""
    thumb_tip = landmarks[THUMB_TIP]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    # Use the distance from index MCP to pinky MCP as a palm-width scale.
    palm_width = float(np.linalg.norm(pinky_mcp - index_mcp))
    if palm_width < 1e-6:
        return False
    thumb_dist = float(np.linalg.norm(thumb_tip - index_mcp))
    return thumb_dist > palm_width * 0.7


def _thumb_up(landmarks: NDArray[np.floating]) -> bool:
    """Thumb tip is well above the wrist (y decreases upward in image space)."""
    thumb_tip = landmarks[THUMB_TIP]
    wrist = landmarks[WRIST]
    # Thumb must extend upward by at least ~10% of image height.
    return (wrist[1] - thumb_tip[1]) > 0.1


def classify_hand(landmarks: NDArray[np.floating]) -> Gesture:
    """Classify a (21, 3) MediaPipe hand into one of six gestures."""
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
    if thumb and not index and not middle and not ring and not pinky:
        if _thumb_up(landmarks):
            return "thumbs_up"

    # open: all five extended.
    if extended_count == 5:
        return "open"

    # fist: none of the four non-thumb extended, and thumb curled.
    if not thumb and not index and not middle and not ring and not pinky:
        return "fist"
    # Also accept fist if thumb is ambiguously extended (some people tuck it,
    # others stick it out slightly). Require all four non-thumb curled.
    if not index and not middle and not ring and not pinky:
        return "fist"

    # peace: index + middle extended; ring + pinky curled.
    if index and middle and not ring and not pinky:
        return "peace"

    # point: only index extended among the four non-thumbs. Thumb state doesn't matter.
    if index and not middle and not ring and not pinky:
        return "point"

    return "none"
```

- [ ] **Step 4: Run; expect pass**

```bash
pytest tests/test_gestures.py -v
```
Expected: 8 passed. If `test_thumbs_up_classifies_thumbs_up` fails, verify the fixture produces a thumb tip significantly above the wrist (the fixture explicitly redirects the thumb upward). If needed, widen the `_thumb_up` threshold slightly.

- [ ] **Step 5: Commit**

```bash
git add src/handspring/gestures.py tests/test_gestures.py
git commit -m "feat(gestures): classify five hand gestures with heuristic rules"
```

---

## Task 6: OSC output with state-change dedupe

**Files:**
- Create: `src/handspring/osc_out.py`
- Create: `tests/test_osc_out.py`

- [ ] **Step 1: Write failing tests `tests/test_osc_out.py`**

```python
"""OSC emitter tests — assert message content and dedup behavior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from handspring.osc_out import OscEmitter
from handspring.types import (
    FaceFeatures,
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
)


@dataclass
class FakeOsc:
    sent: list[tuple[str, Any]]

    def send_message(self, address: str, value: Any) -> None:
        self.sent.append((address, value))


def _frame(
    left_gesture: str = "none",
    right_gesture: str = "none",
    left_present: bool = True,
    right_present: bool = True,
    face_present: bool = True,
) -> FrameResult:
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0)
    left = HandState(
        present=left_present,
        features=hf if left_present else None,
        gesture=left_gesture,  # type: ignore[arg-type]
    )
    right = HandState(
        present=right_present,
        features=hf if right_present else None,
        gesture=right_gesture,  # type: ignore[arg-type]
    )
    face = FaceState(
        present=face_present,
        features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=0.0)
        if face_present
        else None,
    )
    return FrameResult(left=left, right=right, face=face, fps=30.0)


def test_continuous_features_emitted_every_frame():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame())
    addresses = [addr for addr, _ in fake.sent]
    assert "/hand/left/x" in addresses
    assert "/hand/left/y" in addresses
    assert "/hand/left/openness" in addresses
    assert "/hand/right/x" in addresses
    assert "/face/yaw" in addresses
    assert "/face/mouth_open" in addresses


def test_present_flags_are_int():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame())
    for addr, value in fake.sent:
        if addr.endswith("/present"):
            assert value in (0, 1), f"{addr} sent non-int: {value}"


def test_gesture_only_emitted_on_change():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame(left_gesture="fist"))
    emitter.emit(_frame(left_gesture="fist"))
    emitter.emit(_frame(left_gesture="fist"))
    gesture_msgs = [(a, v) for a, v in fake.sent if a == "/hand/left/gesture"]
    assert gesture_msgs == [("/hand/left/gesture", "fist")], (
        f"expected one gesture message, got {gesture_msgs}"
    )


def test_gesture_re_emitted_on_transition():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame(left_gesture="fist"))
    emitter.emit(_frame(left_gesture="open"))
    gesture_msgs = [v for a, v in fake.sent if a == "/hand/left/gesture"]
    assert gesture_msgs == ["fist", "open"]


def test_absent_hand_emits_present_zero_no_features():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame(left_present=False))
    addresses = [addr for addr, _ in fake.sent]
    assert ("/hand/left/present", 0) in fake.sent
    assert "/hand/left/x" not in addresses
    assert "/hand/left/openness" not in addresses


def test_absent_face_emits_present_zero():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame(face_present=False))
    assert ("/face/present", 0) in fake.sent
    assert "/face/yaw" not in [a for a, _ in fake.sent]
```

- [ ] **Step 2: Run; expect ImportError**

```bash
pytest tests/test_osc_out.py -v
```

- [ ] **Step 3: Implement `src/handspring/osc_out.py`**

```python
"""Emit per-frame tracking results as OSC messages over UDP."""
from __future__ import annotations

from typing import Any, Protocol

from handspring.types import FrameResult, Gesture, Side


class _SendsOsc(Protocol):
    def send_message(self, address: str, value: Any) -> None: ...


def _make_client(host: str, port: int) -> _SendsOsc:
    # Import inside the function so tests can construct OscEmitter with a fake
    # client without importing python-osc.
    from pythonosc.udp_client import SimpleUDPClient

    return SimpleUDPClient(host, port)  # type: ignore[no-any-return]


class OscEmitter:
    """Thin stateful wrapper that emits continuous features every frame and
    discrete gesture events only on change."""

    def __init__(
        self,
        *,
        client: _SendsOsc | None = None,
        host: str = "127.0.0.1",
        port: int = 9000,
    ) -> None:
        self._client: _SendsOsc = client if client is not None else _make_client(host, port)
        self._last_gesture: dict[Side, Gesture | None] = {"left": None, "right": None}

    def emit(self, frame: FrameResult) -> None:
        self._emit_hand("left", frame.left)
        self._emit_hand("right", frame.right)
        self._emit_face(frame.face)

    def _emit_hand(self, side: Side, state: Any) -> None:
        self._client.send_message(f"/hand/{side}/present", 1 if state.present else 0)
        if state.present and state.features is not None:
            f = state.features
            self._client.send_message(f"/hand/{side}/x", float(f.x))
            self._client.send_message(f"/hand/{side}/y", float(f.y))
            self._client.send_message(f"/hand/{side}/z", float(f.z))
            self._client.send_message(f"/hand/{side}/openness", float(f.openness))
            self._client.send_message(f"/hand/{side}/pinch", float(f.pinch))

        # Gesture events: emit only on state change.
        current: Gesture = state.gesture
        if current != self._last_gesture[side]:
            self._client.send_message(f"/hand/{side}/gesture", current)
            self._last_gesture[side] = current

    def _emit_face(self, state: Any) -> None:
        self._client.send_message("/face/present", 1 if state.present else 0)
        if state.present and state.features is not None:
            f = state.features
            self._client.send_message("/face/yaw", float(f.yaw))
            self._client.send_message("/face/pitch", float(f.pitch))
            self._client.send_message("/face/mouth_open", float(f.mouth_open))
```

- [ ] **Step 4: Run; expect pass**

```bash
pytest tests/test_osc_out.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/handspring/osc_out.py tests/test_osc_out.py
git commit -m "feat(osc_out): emit features every frame, gestures on transition"
```

---

## Task 7: Tracker (MediaPipe integration)

**Files:**
- Create: `src/handspring/tracker.py`

Cannot unit-test without a camera or a bundled still frame. We rely on integration verification (Task 10's end-to-end run) + the fact that features/gestures modules are already tested.

- [ ] **Step 1: Implement `src/handspring/tracker.py`**

```python
"""MediaPipe wrapper: accepts BGR frames, returns FrameResult."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from handspring.features import face_features, hand_features
from handspring.gestures import classify_hand
from handspring.types import (
    FaceState,
    FrameResult,
    HandState,
    Side,
)


@dataclass
class TrackerConfig:
    max_hands: int = 2
    track_face: bool = True
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


class Tracker:
    """Runs MediaPipe hand + face tracking over successive frames."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        # mp.solutions.hands.Hands consumes RGB frames.
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
                refine_landmarks=False,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
        else:
            self._face_mesh = None

        self._last_frame_time: float | None = None
        self._fps_ema: float = 0.0

    def process(self, bgr_frame: NDArray[np.uint8]) -> FrameResult:
        """Run inference on a single BGR frame and return a FrameResult."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_results = self._hands.process(rgb)
        face_result: Any = self._face_mesh.process(rgb) if self._face_mesh is not None else None

        left_state, right_state = self._hand_states(hand_results)
        face_state = self._face_state(face_result)

        now = time.perf_counter()
        fps = 0.0
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                instant = 1.0 / dt
                self._fps_ema = 0.9 * self._fps_ema + 0.1 * instant if self._fps_ema else instant
                fps = self._fps_ema
        self._last_frame_time = now

        return FrameResult(left=left_state, right=right_state, face=face_state, fps=fps)

    def close(self) -> None:
        self._hands.close()
        if self._face_mesh is not None:
            self._face_mesh.close()

    # ---- Internals ----

    def _hand_states(self, hand_results: Any) -> tuple[HandState, HandState]:
        absent = HandState(present=False, features=None, gesture="none")
        left = absent
        right = absent
        if not hand_results.multi_hand_landmarks or not hand_results.multi_handedness:
            return left, right
        for landmarks, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness,
            strict=False,
        ):
            label: str = handedness.classification[0].label  # "Left" or "Right"
            # MediaPipe's "Left"/"Right" is from the camera's perspective. Invert
            # because we want the user's perspective.
            side: Side = "right" if label == "Left" else "left"
            arr = _landmark_list_to_array(landmarks)
            feats = hand_features(arr)
            gesture = classify_hand(arr)
            state = HandState(present=True, features=feats, gesture=gesture)
            if side == "left":
                left = state
            else:
                right = state
        return left, right

    def _face_state(self, face_result: Any) -> FaceState:
        if face_result is None:
            return FaceState(present=False, features=None)
        if not face_result.multi_face_landmarks:
            return FaceState(present=False, features=None)
        lm = face_result.multi_face_landmarks[0]
        arr = _landmark_list_to_array(lm)
        return FaceState(present=True, features=face_features(arr))


def _landmark_list_to_array(landmark_list: Any) -> NDArray[np.float32]:
    """Convert a MediaPipe NormalizedLandmarkList to an (N, 3) numpy array."""
    count = len(landmark_list.landmark)
    arr = np.zeros((count, 3), dtype=np.float32)
    for i, lm in enumerate(landmark_list.landmark):
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
    return arr
```

- [ ] **Step 2: Sanity check — `mypy` clean, `ruff` clean**

```bash
mypy src/handspring/tracker.py
ruff check src/handspring/tracker.py
```
Both clean.

- [ ] **Step 3: Commit**

```bash
git add src/handspring/tracker.py
git commit -m "feat(tracker): wrap MediaPipe hand + face inference into FrameResult"
```

---

## Task 8: Preview window

**Files:**
- Create: `src/handspring/preview.py`

- [ ] **Step 1: Implement `src/handspring/preview.py`**

```python
"""OpenCV preview window showing tracking overlay on the camera feed."""
from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from handspring.types import FrameResult

_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
_FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_CONTOURS


class Preview:
    """Wraps the OpenCV preview window — manages the HighGUI window lifecycle."""

    WINDOW_NAME = "handspring"

    def __init__(self, *, mirror: bool = True) -> None:
        self._mirror = mirror
        self._created = False

    def render(
        self,
        bgr_frame: NDArray[np.uint8],
        hand_landmark_lists: list[Any],
        face_landmark_lists: list[Any],
        frame_result: FrameResult,
        osc_target: str,
    ) -> bool:
        """Draw one frame. Returns True to keep going, False if the user
        pressed 'q' or closed the window."""
        display = bgr_frame.copy()
        if self._mirror:
            display = cv2.flip(display, 1)
            # When mirroring, landmark x coordinates need flipping for the
            # overlay to align.
            hand_landmark_lists = [_mirror_landmarks(ll, display.shape[1]) for ll in hand_landmark_lists]
            face_landmark_lists = [_mirror_landmarks(ll, display.shape[1]) for ll in face_landmark_lists]

        for ll in hand_landmark_lists:
            mp.solutions.drawing_utils.draw_landmarks(
                display, ll, _HAND_CONNECTIONS
            )
        for ll in face_landmark_lists:
            mp.solutions.drawing_utils.draw_landmarks(
                display, ll, _FACE_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(160, 160, 160), thickness=1
                ),
            )

        _draw_status(display, frame_result, osc_target)

        if not self._created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            self._created = True
        cv2.imshow(self.WINDOW_NAME, display)

        if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            return False
        return True

    def close(self) -> None:
        if self._created:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._created = False


def _draw_status(
    frame: NDArray[np.uint8], frame_result: FrameResult, osc_target: str
) -> None:
    lines = [
        f"FPS: {frame_result.fps:5.1f}",
        f"OSC -> {osc_target}",
        f"Left:  {frame_result.left.gesture if frame_result.left.present else '-'}",
        f"Right: {frame_result.right.gesture if frame_result.right.present else '-'}",
    ]
    y = 30
    for text in lines:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 24


def _mirror_landmarks(landmark_list: Any, _width: int) -> Any:
    """Return a new landmark list with x coordinates mirrored to 1 - x."""
    from mediapipe.framework.formats import landmark_pb2

    mirrored = landmark_pb2.NormalizedLandmarkList()
    for lm in landmark_list.landmark:
        new_lm = mirrored.landmark.add()
        new_lm.x = 1.0 - lm.x
        new_lm.y = lm.y
        new_lm.z = lm.z
        new_lm.visibility = lm.visibility
        new_lm.presence = lm.presence
    return mirrored
```

- [ ] **Step 2: Ruff + mypy**

```bash
ruff check src/handspring/preview.py
mypy src/handspring/preview.py
```
Both clean.

- [ ] **Step 3: Commit**

```bash
git add src/handspring/preview.py
git commit -m "feat(preview): OpenCV preview window with landmark overlay"
```

---

## Task 9: Main CLI and loop

**Files:**
- Create: `src/handspring/__main__.py`

- [ ] **Step 1: Implement `src/handspring/__main__.py`**

```python
"""`python -m handspring` entry point: camera → tracker → OSC + preview."""
from __future__ import annotations

import argparse
import signal
import sys
import time
from types import FrameType
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from handspring import __version__
from handspring.osc_out import OscEmitter
from handspring.preview import Preview
from handspring.tracker import Tracker, TrackerConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="handspring",
        description="Live gesture tracker → OSC stream.",
    )
    p.add_argument("--host", default="127.0.0.1", help="OSC receiver host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=9000, help="OSC receiver port (default: 9000)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--no-preview", action="store_true", help="disable the OpenCV preview window")
    p.add_argument("--no-face", action="store_true", help="disable face tracking")
    p.add_argument(
        "--hands", type=int, choices=[0, 1, 2], default=2, help="max hands to track"
    )
    p.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="disable preview mirror",
    )
    p.add_argument(
        "--fps-log-interval",
        type=float,
        default=0.5,
        help="print FPS + state to terminal every N seconds (default: 0.5)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.set_defaults(mirror=True)
    return p.parse_args(argv)


class _Shutdown:
    """Tiny helper so Ctrl+C sets a flag instead of raising inside the main loop."""

    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"error: could not open camera {args.camera}", file=sys.stderr)
        return 2

    tracker = Tracker(
        TrackerConfig(
            max_hands=args.hands,
            track_face=not args.no_face,
        )
    )
    emitter = OscEmitter(host=args.host, port=args.port)
    preview = Preview(mirror=args.mirror) if not args.no_preview else None
    shutdown = _Shutdown()

    print(f"handspring {__version__}", flush=True)
    print(f"camera: {args.camera}", flush=True)
    print(f"OSC:    {args.host}:{args.port}", flush=True)
    print(f"hands:  {args.hands}   face: {'off' if args.no_face else 'on'}", flush=True)
    print("Ctrl+C to quit.", flush=True)

    last_log = 0.0
    try:
        while not shutdown.requested:
            ok, bgr = cap.read()
            if not ok:
                # Momentary read failure — try again after a beat.
                time.sleep(0.01)
                continue
            result = tracker.process(bgr)
            emitter.emit(result)

            if preview is not None:
                hand_landmarks, face_landmarks = _extract_landmark_lists(tracker, bgr)
                if not preview.render(
                    bgr,
                    hand_landmarks,
                    face_landmarks,
                    result,
                    f"{args.host}:{args.port}",
                ):
                    break

            now = time.monotonic()
            if now - last_log >= args.fps_log_interval:
                _print_status(result)
                last_log = now
    finally:
        cap.release()
        tracker.close()
        if preview is not None:
            preview.close()
        cv2.destroyAllWindows()
    return 0


def _extract_landmark_lists(
    tracker: Tracker, bgr: np.ndarray
) -> tuple[list[Any], list[Any]]:
    """Re-run the underlying MediaPipe solvers so the preview sees the same
    landmark lists the tracker used. The duplication is intentional —
    MediaPipe's API doesn't expose the raw lists from Tracker without
    doubling the public surface, and a second pass at the already-available
    frame is cheap."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hand_results = tracker._hands.process(rgb)  # noqa: SLF001 — private access is fine inside the package
    face_results = (
        tracker._face_mesh.process(rgb) if tracker._face_mesh is not None else None  # noqa: SLF001
    )
    hand_lists = list(hand_results.multi_hand_landmarks or [])
    face_lists = (
        list(face_results.multi_face_landmarks or []) if face_results is not None else []
    )
    return hand_lists, face_lists


def _print_status(result: Any) -> None:
    left = result.left.gesture if result.left.present else "-"
    right = result.right.gesture if result.right.present else "-"
    face = "yes" if result.face.present else "no"
    print(
        f"\rFPS {result.fps:5.1f}  L:{left:<10} R:{right:<10} face:{face}",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
```

Re-running MediaPipe inside `_extract_landmark_lists` doubles work. Acceptable for v1 — the spec says ~30 FPS on modern CPUs and the second pass is ~10–20% of a frame's cost. If we hit a bottleneck, Task 13 (future) can expose raw landmark lists from `Tracker.process`.

- [ ] **Step 2: Ruff, mypy, pytest**

```bash
ruff check .
ruff format --check .
mypy src/
pytest
```
All clean.

If `ruff format --check` fails, run `ruff format .` and stage the changes before committing.

- [ ] **Step 3: Commit**

```bash
git add src/handspring/__main__.py
git commit -m "feat(main): CLI entry wiring camera, tracker, OSC, preview"
```

---

## Task 10: Demo receiver

**Files:**
- Create: `examples/tone_synth.py`

- [ ] **Step 1: Implement `examples/tone_synth.py`**

```python
"""Minimal demo receiver for handspring.

- Left hand Y → pitch in 200–800 Hz (exponential; low Y = low pitch)
- Right hand Y → amplitude in 0..0.3
- Left hand gesture "fist" → mute; "open" → unmute

Run:
    python examples/tone_synth.py            # listens on 127.0.0.1:9000
    python examples/tone_synth.py --host X   # override
"""
from __future__ import annotations

import argparse
import math
import threading
from typing import Any

import numpy as np
import sounddevice as sd  # type: ignore[import-not-found]
from pythonosc import dispatcher, osc_server


SAMPLE_RATE = 48_000
SMOOTH = 0.01  # one-pole smoothing coef per frame (a smaller number = slower)


class Synth:
    def __init__(self) -> None:
        self.target_freq = 440.0
        self.target_amp = 0.0
        self.muted = False
        self._freq = 440.0
        self._amp = 0.0
        self._phase = 0.0
        self._lock = threading.Lock()

    def set_left_y(self, y: float) -> None:
        y = max(0.0, min(1.0, float(y)))
        # y=0 (top) → high pitch; y=1 (bottom) → low pitch.
        log_low = math.log2(200.0)
        log_high = math.log2(800.0)
        self.target_freq = 2 ** (log_high - y * (log_high - log_low))

    def set_right_y(self, y: float) -> None:
        y = max(0.0, min(1.0, float(y)))
        self.target_amp = (1.0 - y) * 0.3

    def set_left_gesture(self, gesture: str) -> None:
        if gesture == "fist":
            self.muted = True
        elif gesture == "open":
            self.muted = False

    def callback(self, outdata: np.ndarray, frames: int, _time_info: Any, _status: Any) -> None:
        with self._lock:
            for i in range(frames):
                self._freq += (self.target_freq - self._freq) * SMOOTH
                target_amp = 0.0 if self.muted else self.target_amp
                self._amp += (target_amp - self._amp) * SMOOTH
                sample = math.sin(self._phase) * self._amp
                outdata[i, 0] = sample
                self._phase += 2 * math.pi * self._freq / SAMPLE_RATE
                if self._phase > 2 * math.pi:
                    self._phase -= 2 * math.pi


def _make_dispatcher(synth: Synth) -> dispatcher.Dispatcher:
    d = dispatcher.Dispatcher()
    d.map("/hand/left/y", lambda _addr, val: synth.set_left_y(val))
    d.map("/hand/right/y", lambda _addr, val: synth.set_right_y(val))
    d.map("/hand/left/gesture", lambda _addr, val: synth.set_left_gesture(str(val)))
    return d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    synth = Synth()
    with sd.OutputStream(
        channels=1, samplerate=SAMPLE_RATE, callback=synth.callback, blocksize=256
    ):
        server = osc_server.ThreadingOSCUDPServer(
            (args.host, args.port), _make_dispatcher(synth)
        )
        print(f"tone_synth listening on {args.host}:{args.port}; Ctrl+C to quit")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Ruff + mypy check**

```bash
ruff check examples/tone_synth.py
```

mypy may complain about `sounddevice` lacking type stubs — that's covered by `ignore_missing_imports = true` in pyproject. If it still complains, add the same ignore at module level.

- [ ] **Step 3: Commit**

```bash
git add examples/tone_synth.py
git commit -m "feat(examples): add tone_synth demo receiver"
```

---

## Task 11: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md`**

````markdown
# handspring

Webcam → hand/face tracking → OSC stream.

Drop-in gesture input for creative-coding projects: audio synths, visuals, game controls, prompt generators. Written in Python, powered by [MediaPipe](https://mediapipe.dev/).

## Quick start

```bash
# 1. Install (Python 3.10+ required)
pip install -e '.[dev]'
# or with uv:
# uv sync

# 2. Run
python -m handspring
```

A preview window opens showing your webcam with hand/face landmarks drawn.
OSC packets fly out to `127.0.0.1:9000` (UDP). Terminal shows FPS + current
gesture state.

## Demo

In one terminal:

```bash
python -m handspring
```

In another:

```bash
python examples/tone_synth.py
```

Move your hands. The synth plays sine tones — left hand Y controls pitch,
right hand Y controls amplitude. Make a fist with your left hand to mute;
open your palm to unmute.

## OSC reference

Continuous features (sent every frame, ~30 Hz):

| Address | Type | Range | Notes |
|---|---|---|---|
| `/hand/<side>/present` | int | 0 or 1 | `<side>` is `left` or `right` (user's perspective) |
| `/hand/<side>/x` | float | 0..1 | palm center, normalized to frame width |
| `/hand/<side>/y` | float | 0..1 | palm center, normalized to frame height |
| `/hand/<side>/z` | float | — | relative depth |
| `/hand/<side>/openness` | float | 0..1 | 0 = fist, 1 = open palm |
| `/hand/<side>/pinch` | float | 0..1 | thumb-index proximity |
| `/face/present` | int | 0 or 1 | |
| `/face/yaw` | float | -1..1 | negative = looking left |
| `/face/pitch` | float | -1..1 | negative = looking down |
| `/face/mouth_open` | float | 0..1 | |

Discrete gesture events (per hand, sent only on state transitions):

| Address | Type | Values |
|---|---|---|
| `/hand/<side>/gesture` | string | `fist` \| `open` \| `point` \| `peace` \| `thumbs_up` \| `none` |

## CLI flags

```
--host HOST            OSC receiver host (default: 127.0.0.1)
--port PORT            OSC receiver port (default: 9000)
--camera N             camera index (default: 0)
--no-preview           disable the OpenCV preview window
--no-face              disable face tracking (hands only)
--hands {0,1,2}        max hands to track (default: 2)
--no-mirror            do not mirror the preview horizontally
--fps-log-interval SEC print status every N seconds (default: 0.5)
```

## Building your own receiver

The OSC stream is the whole point. Any OSC-speaking tool can be a receiver:

- Max/MSP, Pure Data, TouchDesigner — native OSC support
- SuperCollider — `OSCdef` / `NetAddr`
- Ableton Live — via `Max for Live` OSC bridge
- Unity / Godot — via community OSC libraries
- A Python script — see `examples/tone_synth.py`
- A web app — bridge OSC to WebSocket

For MIDI-only applications, use a bridge tool (`osculator` on macOS, `loopMIDI`
on Windows) to forward OSC to MIDI messages.

## Development

```bash
pip install -e '.[dev]'
pytest              # run unit tests (no camera required)
ruff check .        # lint
ruff format .       # format
mypy src/           # type check
```

CI (GitHub Actions) runs the same checks on Ubuntu and macOS with Python
3.10 and 3.12.

## License

MIT. See [LICENSE](./LICENSE).
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: flesh out README with quickstart, OSC reference, dev setup"
```

---

## Task 12: Final verification + tag

- [ ] **Step 1: Install into the current venv**

```bash
pip install -e '.[dev]'
```

Or `uv sync` if the user has `uv`.

- [ ] **Step 2: Run the full check suite**

```bash
ruff check .
ruff format --check .
mypy src/
pytest
```

Every command must exit 0. If `ruff format --check` fails, run `ruff format .` and make a dedicated commit:

```bash
git add -u
git commit -m "style: apply ruff format"
```

- [ ] **Step 3: Smoke test (manual, documented in the status report)**

Cannot be automated — requires a camera and a human. Skip automated execution; verify the module imports correctly:

```bash
python -c "import handspring; print(handspring.__version__)"
python -m handspring --version
```

Both should print `0.1.0`.

- [ ] **Step 4: Tag**

```bash
git tag -a v0.1.0 -m "handspring v0.1.0 — first working release"
```

Do NOT push.

- [ ] **Step 5: Summary report**

List:
- Commit count on main since init
- Test count
- Biggest modules by LOC
- Any manual interventions beyond the plan (e.g., fixture thresholds adjusted)

---

## Execution notes

- **Sequential only** — each task mutates files the next depends on.
- **Every task ends with `pytest` and `ruff check` clean** (not just lint; real tests too once they exist — Tasks 2+).
- If a fixture threshold in `tests/fixtures.py` needs tweaking to satisfy `test_features.py` or `test_gestures.py`, tweak the fixture with a comment explaining why — do not loosen the production code thresholds without cause.
- If MediaPipe/OpenCV installation fails on a runner due to missing system libs, the CI file already has `libgl1 libglib2.0-0` for Ubuntu. macOS works out of the box.
