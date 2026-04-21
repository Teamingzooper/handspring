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

_EYE_OPEN_MIN_RAW = 0.02  # eye width ratio at closed
_EYE_OPEN_MAX_RAW = 0.32  # eye width ratio at wide open


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


def _mouth_corner_delta(landmarks: NDArray[np.floating[Any]], eye_distance: float) -> float:
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


# Silence unused-import warning — F_NOSE_TIP is imported per the public API
# contract specified in the plan (other modules may import from here).
__all__ = [
    "classify_expression",
    "eye_open_values",
]
_ = F_NOSE_TIP  # keep import; may be needed by future callers via star-import
