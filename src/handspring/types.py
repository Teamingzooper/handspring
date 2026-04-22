"""Core dataclasses for frame-level tracking results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["left", "right"]
Gesture = Literal["fist", "open", "point", "peace", "thumbs_up", "ok", "rock", "three", "none"]
Expression = Literal["smile", "frown", "surprise", "wink_left", "wink_right", "neutral"]
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
    index_x: float
    index_y: float
    thumb_x: float
    thumb_y: float
    # Palm orientation (radians). Computed from landmark geometry.
    # roll = rotation in image plane (wrist→middle-MCP angle, 0 = pointing up).
    # pitch = tilt forward/back (positive = fingers toward camera).
    # yaw = left/right tilt around palm axis (positive = pinky-side forward).
    palm_roll: float = 0.0
    palm_pitch: float = 0.0
    palm_yaw: float = 0.0


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
    """One hand's state for a single frame."""

    present: bool
    features: HandFeatures | None
    gesture: Gesture
    motion: MotionState


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
    expression: Expression
    eye_left_open: float
    eye_right_open: float


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
    clap_event: bool
