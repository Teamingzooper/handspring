# handspring v0.2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. TDD. Checkbox progress.

**Goal:** Add MediaPipe Pose tracking + unified neon-green skeleton overlay across hand/face/pose in the preview; emit 8 upper-body joints as OSC.

**Architecture:** One new MediaPipe solver added to `Tracker`. New `PoseState`/`PoseLandmark` types in `types.py`. Preview gets a single `DrawingSpec` used across all three overlays. CLI adds `--no-pose`.

**Tech additions:** `mp.solutions.pose` (already shipped with MediaPipe 0.10). No new dependencies.

**Spec:** [`2026-04-21-handspring-v02-pose.md`](../specs/2026-04-21-handspring-v02-pose.md)

---

## Task 1: New types — `PoseLandmark`, `PoseState`, `Joint`, updated `FrameResult`

**Files:**
- Modify: `src/handspring/types.py`
- Modify: `tests/test_types.py`

### Step 1: Extend `tests/test_types.py`

Append these tests to the existing file:

```python
from handspring.types import Joint, PoseLandmark, PoseState


def test_pose_landmark_frozen():
    pl = PoseLandmark(x=0.4, y=0.5, z=-0.1, visible=True)
    import pytest
    with pytest.raises(AttributeError):
        pl.x = 0.0  # type: ignore[misc]


def test_pose_state_absent():
    ps = PoseState(present=False, joints=None)
    assert ps.present is False
    assert ps.joints is None


def test_pose_state_present_with_joints():
    pl = PoseLandmark(x=0.4, y=0.5, z=0.0, visible=True)
    ps = PoseState(present=True, joints={"shoulder_left": pl})
    assert ps.present is True
    assert ps.joints is not None
    assert ps.joints["shoulder_left"].x == 0.4


def test_frame_result_has_pose():
    from handspring.types import FaceState, FrameResult, HandState
    left = HandState(present=False, features=None, gesture="none")
    right = HandState(present=False, features=None, gesture="none")
    face = FaceState(present=False, features=None)
    pose = PoseState(present=False, joints=None)
    fr = FrameResult(left=left, right=right, face=face, pose=pose, fps=30.0)
    assert fr.pose.present is False
```

### Step 2: Run — expect ImportError + 1 new-signature failure

```bash
pytest tests/test_types.py -v
```

### Step 3: Update `src/handspring/types.py`

Add the new types and grow `FrameResult`:

```python
"""Core dataclasses for frame-level tracking results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
Gesture = Literal["fist", "open", "point", "peace", "thumbs_up", "none"]
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
class PoseLandmark:
    """One upper-body joint position in a single frame.

    x, y are normalized in [0, 1]. z is relative depth. `visible` mirrors
    MediaPipe's per-landmark visibility score thresholded at 0.5; joints
    with `visible=False` have unreliable x/y/z and should be ignored by
    downstream consumers.
    """

    x: float
    y: float
    z: float
    visible: bool


@dataclass(frozen=True)
class PoseState:
    """Body pose state for a single frame.

    `joints` maps the 8 tracked joint names to their landmarks when present;
    None when MediaPipe didn't detect a body at all.
    """

    present: bool
    joints: dict[Joint, PoseLandmark] | None


@dataclass(frozen=True)
class FrameResult:
    """Full per-frame tracking result."""

    left: HandState
    right: HandState
    face: FaceState
    pose: PoseState
    fps: float
```

### Step 4: Run — expect 10 passed (6 old + 4 new)

```bash
pytest tests/test_types.py -v
```

### Step 5: Lint, format, mypy

```bash
ruff check src/handspring/types.py tests/test_types.py
ruff format --check src/handspring/types.py tests/test_types.py
mypy src/handspring/types.py
```

Run `ruff format` if needed and include in commit.

### Step 6: Commit

```bash
git add src/handspring/types.py tests/test_types.py
git commit -m "feat(types): add PoseLandmark, PoseState, Joint; grow FrameResult"
```

---

## Task 2: Tracker integration — run Pose alongside Hands + FaceMesh

**Files:**
- Modify: `src/handspring/tracker.py`

The existing `FrameResult` constructors in `tracker.py` will now fail to compile because `FrameResult` requires a `pose` field. Fix by threading pose inference through.

### Step 1: Rewrite `src/handspring/tracker.py`

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
    Joint,
    PoseLandmark,
    PoseState,
    Side,
)

# MediaPipe PoseLandmark indices for the 8 joints we emit.
_POSE_JOINTS: dict[Joint, int] = {
    "shoulder_left": 12,   # MediaPipe's camera-perspective RIGHT_SHOULDER
    "shoulder_right": 11,  # MediaPipe LEFT_SHOULDER
    "elbow_left": 14,
    "elbow_right": 13,
    "wrist_left": 16,
    "wrist_right": 15,
    "hip_left": 24,
    "hip_right": 23,
}

_VISIBILITY_THRESHOLD = 0.5


@dataclass
class TrackerConfig:
    max_hands: int = 2
    track_face: bool = True
    track_pose: bool = True
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


class Tracker:
    """Runs MediaPipe hand + face + pose tracking over successive frames."""

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
                refine_landmarks=False,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
        else:
            self._face_mesh = None

        if self._config.track_pose:
            self._pose: Any = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # full model (not light, not heavy)
                smooth_landmarks=True,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
        else:
            self._pose = None

        self._last_frame_time: float | None = None
        self._fps_ema: float = 0.0

    def process(self, bgr_frame: NDArray[np.uint8]) -> FrameResult:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_results = self._hands.process(rgb)
        face_result: Any = self._face_mesh.process(rgb) if self._face_mesh is not None else None
        pose_result: Any = self._pose.process(rgb) if self._pose is not None else None

        left_state, right_state = self._hand_states(hand_results)
        face_state = self._face_state(face_result)
        pose_state = self._pose_state(pose_result)

        now = time.perf_counter()
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
        )

    def close(self) -> None:
        self._hands.close()
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._pose is not None:
            self._pose.close()

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
            label: str = handedness.classification[0].label
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
        if face_result is None or not face_result.multi_face_landmarks:
            return FaceState(present=False, features=None)
        lm = face_result.multi_face_landmarks[0]
        arr = _landmark_list_to_array(lm)
        return FaceState(present=True, features=face_features(arr))

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

### Step 2: Lint + typecheck

```bash
ruff check src/handspring/tracker.py
ruff format --check src/handspring/tracker.py
mypy src/handspring/tracker.py
```

### Step 3: Smoke test

```bash
python -c "from handspring.tracker import Tracker, TrackerConfig; t = Tracker(TrackerConfig(track_face=False, track_pose=True, max_hands=0)); t.close(); print('ok')"
```

Expected: `ok`. Pose solver loads without crashing.

### Step 4: Run full tests (existing + new types tests)

```bash
pytest
```

Expected: 35 passed (31 prior + 4 new types tests).

### Step 5: Commit

```bash
git add src/handspring/tracker.py
git commit -m "feat(tracker): add MediaPipe Pose and emit PoseState in FrameResult"
```

---

## Task 3: OSC emission for pose

**Files:**
- Modify: `src/handspring/osc_out.py`
- Modify: `tests/test_osc_out.py`

### Step 1: Extend `tests/test_osc_out.py`

Add a helper for building frames with pose, and add tests:

```python
# Add to imports at top of file:
from handspring.types import Joint, PoseLandmark, PoseState


def _pose(joints: dict[Joint, PoseLandmark] | None) -> PoseState:
    return PoseState(present=joints is not None, joints=joints)


def _frame_with_pose(pose: PoseState) -> FrameResult:
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0)
    left = HandState(present=True, features=hf, gesture="none")
    right = HandState(present=True, features=hf, gesture="none")
    face = FaceState(
        present=True,
        features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=0.0),
    )
    return FrameResult(left=left, right=right, face=face, pose=pose, fps=30.0)


# Also update the existing _frame() helper in this file to include pose=_pose(None)
# when constructing FrameResult. Find the line `return FrameResult(left=left, right=right, face=face, fps=30.0)`
# and change to `return FrameResult(left=left, right=right, face=face, pose=PoseState(present=False, joints=None), fps=30.0)`.


def test_pose_present_emits_joint_messages():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    joints: dict[Joint, PoseLandmark] = {
        "shoulder_left": PoseLandmark(x=0.3, y=0.4, z=0.0, visible=True),
        "shoulder_right": PoseLandmark(x=0.7, y=0.4, z=0.0, visible=True),
        "elbow_left": PoseLandmark(x=0.25, y=0.55, z=0.0, visible=True),
        "elbow_right": PoseLandmark(x=0.75, y=0.55, z=0.0, visible=True),
        "wrist_left": PoseLandmark(x=0.2, y=0.7, z=0.0, visible=True),
        "wrist_right": PoseLandmark(x=0.8, y=0.7, z=0.0, visible=True),
        "hip_left": PoseLandmark(x=0.4, y=0.85, z=0.0, visible=True),
        "hip_right": PoseLandmark(x=0.6, y=0.85, z=0.0, visible=True),
    }
    emitter.emit(_frame_with_pose(_pose(joints)))
    addresses = [addr for addr, _ in fake.sent]
    assert ("/pose/present", 1) in fake.sent
    for joint in joints:
        assert f"/pose/{joint}/visible" in addresses
        assert f"/pose/{joint}/x" in addresses
        assert f"/pose/{joint}/y" in addresses
        assert f"/pose/{joint}/z" in addresses


def test_pose_absent_emits_only_present_zero():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame_with_pose(_pose(None)))
    pose_messages = [a for a, _ in fake.sent if a.startswith("/pose/")]
    assert pose_messages == ["/pose/present"]
    assert ("/pose/present", 0) in fake.sent


def test_pose_joint_invisible_suppresses_xyz():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    joints: dict[Joint, PoseLandmark] = {
        "shoulder_left": PoseLandmark(x=0.3, y=0.4, z=0.0, visible=True),
        "shoulder_right": PoseLandmark(x=0.7, y=0.4, z=0.0, visible=False),
        "elbow_left": PoseLandmark(x=0.25, y=0.55, z=0.0, visible=True),
        "elbow_right": PoseLandmark(x=0.75, y=0.55, z=0.0, visible=True),
        "wrist_left": PoseLandmark(x=0.2, y=0.7, z=0.0, visible=True),
        "wrist_right": PoseLandmark(x=0.8, y=0.7, z=0.0, visible=True),
        "hip_left": PoseLandmark(x=0.4, y=0.85, z=0.0, visible=True),
        "hip_right": PoseLandmark(x=0.6, y=0.85, z=0.0, visible=True),
    }
    emitter.emit(_frame_with_pose(_pose(joints)))
    assert ("/pose/shoulder_right/visible", 0) in fake.sent
    addresses = [addr for addr, _ in fake.sent]
    assert "/pose/shoulder_right/x" not in addresses
    assert "/pose/shoulder_right/y" not in addresses
    assert "/pose/shoulder_right/z" not in addresses
    # Visible joints still emit
    assert "/pose/shoulder_left/x" in addresses
```

### Step 2: Update existing `_frame()` helper

In the same file, find the `_frame` helper (near the top) and update its return to include pose. Change:

```python
return FrameResult(left=left, right=right, face=face, fps=30.0)
```

to:

```python
return FrameResult(
    left=left,
    right=right,
    face=face,
    pose=PoseState(present=False, joints=None),
    fps=30.0,
)
```

Also import `PoseState` at the top if not already there.

### Step 3: Update `src/handspring/osc_out.py`

Modify `emit` to call a new `_emit_pose`, and add the method:

```python
# Near the imports:
from handspring.types import (
    FaceState,
    FrameResult,
    Gesture,
    HandState,
    PoseState,
    Side,
)


# In OscEmitter.emit:
def emit(self, frame: FrameResult) -> None:
    self._emit_hand("left", frame.left)
    self._emit_hand("right", frame.right)
    self._emit_face(frame.face)
    self._emit_pose(frame.pose)


# New method on OscEmitter:
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

### Step 4: Run tests

```bash
pytest
```

Expected: 38 passed (35 after Task 2 + 3 new pose tests).

### Step 5: Lint + format + mypy

```bash
ruff check src/handspring/osc_out.py tests/test_osc_out.py
ruff format --check .
mypy src/
```

### Step 6: Commit

```bash
git add src/handspring/osc_out.py tests/test_osc_out.py
git commit -m "feat(osc_out): emit /pose/* joint positions with per-joint visibility"
```

---

## Task 4: Unified skeleton preview — hand, face, pose drawn in neon green

**Files:**
- Modify: `src/handspring/preview.py`
- Modify: `src/handspring/__main__.py` (extract pose landmark list alongside hand/face)

### Step 1: Rewrite `src/handspring/preview.py`

```python
"""OpenCV preview window showing a unified neon-green skeleton overlay."""
from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from handspring.types import FrameResult

# MediaPipe connection sets (tuples of (src_idx, dst_idx)).
_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
_FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_CONTOURS
_POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# Shared "neon-green skeleton" drawing specs. Colors are BGR (OpenCV order).
_DOT_SPEC = mp.solutions.drawing_utils.DrawingSpec(
    color=(102, 204, 153),   # BGR for #99CC66 — desaturated green dots
    thickness=-1,
    circle_radius=3,
)
_LINE_SPEC = mp.solutions.drawing_utils.DrawingSpec(
    color=(136, 255, 0),     # BGR for #00FF88 — bright green bones
    thickness=2,
)


class Preview:
    """OpenCV preview window with landmark-skeleton overlay."""

    WINDOW_NAME = "handspring"

    def __init__(self, *, mirror: bool = True) -> None:
        self._mirror = mirror
        self._created = False

    def render(
        self,
        bgr_frame: NDArray[np.uint8],
        hand_landmark_lists: list[Any],
        face_landmark_lists: list[Any],
        pose_landmarks: Any | None,
        frame_result: FrameResult,
        osc_target: str,
    ) -> bool:
        display = bgr_frame.copy()
        if self._mirror:
            display = cv2.flip(display, 1)
            hand_landmark_lists = [
                _mirror_landmarks(ll) for ll in hand_landmark_lists
            ]
            face_landmark_lists = [
                _mirror_landmarks(ll) for ll in face_landmark_lists
            ]
            if pose_landmarks is not None:
                pose_landmarks = _mirror_landmarks(pose_landmarks)

        # Draw pose first (background layer), then face, then hands on top.
        if pose_landmarks is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                display,
                pose_landmarks,
                _POSE_CONNECTIONS,
                landmark_drawing_spec=_DOT_SPEC,
                connection_drawing_spec=_LINE_SPEC,
            )
        for ll in face_landmark_lists:
            mp.solutions.drawing_utils.draw_landmarks(
                display,
                ll,
                _FACE_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=_LINE_SPEC,
            )
        for ll in hand_landmark_lists:
            mp.solutions.drawing_utils.draw_landmarks(
                display,
                ll,
                _HAND_CONNECTIONS,
                landmark_drawing_spec=_DOT_SPEC,
                connection_drawing_spec=_LINE_SPEC,
            )

        _draw_status(display, frame_result, osc_target)

        if not self._created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            self._created = True
        cv2.imshow(self.WINDOW_NAME, display)

        if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
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
        f"Pose:  {'on' if frame_result.pose.present else '-'}",
    ]
    y = 30
    for text in lines:
        cv2.putText(
            frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA
        )
        y += 24


def _mirror_landmarks(landmark_list: Any) -> Any:
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

### Step 2: Update `src/handspring/__main__.py` to extract pose landmarks + pass to preview

Find the existing `_extract_landmark_lists` function. Rename and rewrite to return three lists (hands, faces, pose):

```python
def _extract_landmark_lists(
    tracker: Tracker, bgr: np.ndarray
) -> tuple[list[Any], list[Any], Any | None]:
    """Re-run the underlying MediaPipe solvers so the preview sees the same
    landmark lists the tracker used."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hand_results = tracker._hands.process(rgb)  # noqa: SLF001
    face_results = (
        tracker._face_mesh.process(rgb) if tracker._face_mesh is not None else None  # noqa: SLF001
    )
    pose_results = (
        tracker._pose.process(rgb) if tracker._pose is not None else None  # noqa: SLF001
    )
    hand_lists = list(hand_results.multi_hand_landmarks or [])
    face_lists = (
        list(face_results.multi_face_landmarks or []) if face_results is not None else []
    )
    pose_landmarks = pose_results.pose_landmarks if pose_results is not None else None
    return hand_lists, face_lists, pose_landmarks
```

Update the caller in `main()`:

```python
if preview is not None:
    hand_landmarks, face_landmarks, pose_landmarks = _extract_landmark_lists(tracker, bgr)
    if not preview.render(
        bgr,
        hand_landmarks,
        face_landmarks,
        pose_landmarks,
        result,
        f"{args.host}:{args.port}",
    ):
        break
```

### Step 3: Add `--no-pose` CLI flag in `_parse_args`

In `_parse_args`, add:

```python
p.add_argument("--no-pose", action="store_true", help="disable body/arm pose tracking")
```

And wire to `TrackerConfig` in `main()`:

```python
tracker = Tracker(
    TrackerConfig(
        max_hands=args.hands,
        track_face=not args.no_face,
        track_pose=not args.no_pose,
    )
)
```

Also update the startup banner print lines to include pose status:

```python
print(f"hands:  {args.hands}   face: {'off' if args.no_face else 'on'}   pose: {'off' if args.no_pose else 'on'}", flush=True)
```

### Step 4: Full check suite

```bash
ruff check .
ruff format --check .
mypy src/
pytest
```

All green. Fix any `ruff format` issues in the same commit.

### Step 5: Smoke imports

```bash
python -c "from handspring.preview import Preview; p = Preview(); p.close(); print('ok')"
python -m handspring --help | grep no-pose
python -m handspring --version
```

### Step 6: Commit

```bash
git add src/handspring/preview.py src/handspring/__main__.py
git commit -m "feat(preview): unified neon skeleton + pose overlay, --no-pose flag"
```

---

## Task 5: README update + tag v0.2.0

**Files:**
- Modify: `README.md`

### Step 1: Update README

Find the **OSC reference** section. After the discrete-gesture table, add a new table for pose:

```markdown
Body pose (continuous per frame when present):

| Address | Type | Range | Notes |
|---|---|---|---|
| `/pose/present` | int | 0 or 1 | 1 when a body is detected |
| `/pose/<joint>/visible` | int | 0 or 1 | per-joint visibility; `0` means unreliable |
| `/pose/<joint>/x` | float | 0..1 | sent only when `visible=1` |
| `/pose/<joint>/y` | float | 0..1 | |
| `/pose/<joint>/z` | float | — | relative depth |

`<joint>` is one of: `shoulder_left`, `shoulder_right`, `elbow_left`, `elbow_right`, `wrist_left`, `wrist_right`, `hip_left`, `hip_right`.
```

In the **CLI flags** code block, add one line:

```
--no-pose              disable body/arm pose tracking
```

### Step 2: Run the full check suite one more time

```bash
ruff check .
ruff format --check .
mypy src/
pytest
```

All green.

### Step 3: Commit + tag

```bash
git add README.md
git commit -m "docs: document /pose/* OSC messages and --no-pose flag"
git tag -a v0.2.0 -m "handspring v0.2.0 — body pose + unified skeleton overlay"
```

### Step 4: Report

Summarize:
- Commits on branch since the last tag (`git log --oneline v0.1.0..HEAD`)
- Test count delta (v0.1.0 → v0.2.0)
- Any manual interventions beyond the plan
