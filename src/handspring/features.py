"""Derive normalized continuous features from MediaPipe landmarks."""

from __future__ import annotations

from typing import Any

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


def _validate(landmarks: NDArray[np.floating[Any]]) -> None:
    if not np.all(np.isfinite(landmarks)):
        raise ValueError("landmarks contain NaN or Inf")


def _hand_span(landmarks: NDArray[np.floating[Any]]) -> float:
    """Approximate hand size: wrist → middle-finger MCP distance."""
    return float(np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST]))


def hand_features(landmarks: NDArray[np.floating[Any]]) -> HandFeatures:
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

    index_x = float(np.clip(landmarks[INDEX_TIP][0], 0.0, 1.0))
    index_y = float(np.clip(landmarks[INDEX_TIP][1], 0.0, 1.0))

    thumb_x = float(np.clip(landmarks[THUMB_TIP][0], 0.0, 1.0))
    thumb_y = float(np.clip(landmarks[THUMB_TIP][1], 0.0, 1.0))

    return HandFeatures(
        x=x,
        y=y,
        z=z,
        openness=openness,
        pinch=pinch,
        index_x=index_x,
        index_y=index_y,
        thumb_x=thumb_x,
        thumb_y=thumb_y,
    )


def face_features(landmarks: NDArray[np.floating[Any]]) -> FaceFeatures:
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


from handspring.types import HandState  # noqa: E402

# Raw distance threshold (frame-normalized). Matches the viz threshold in
# preview.py so "green pinch line" <=> is_pinching == True.
_IS_PINCHING_DISTANCE = 0.05


def is_pinching(state: HandState) -> bool:
    if not state.present or state.features is None:
        return False
    dx = state.features.thumb_x - state.features.index_x
    dy = state.features.thumb_y - state.features.index_y
    return bool((dx * dx + dy * dy) ** 0.5 < _IS_PINCHING_DISTANCE)
