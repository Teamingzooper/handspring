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
