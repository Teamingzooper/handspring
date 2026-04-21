"""Classify a hand into one of six discrete gestures."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from handspring.features import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
)
from handspring.types import Gesture


def _finger_extended(
    landmarks: NDArray[np.floating[Any]],
    tip_idx: int,
    pip_idx: int,
) -> bool:
    """A finger is "extended" if its tip is farther from the wrist than the PIP joint."""
    wrist = landmarks[WRIST]
    tip_dist = float(np.linalg.norm(landmarks[tip_idx] - wrist))
    pip_dist = float(np.linalg.norm(landmarks[pip_idx] - wrist))
    # 5% margin so borderline poses don't flicker between classes.
    return tip_dist > pip_dist * 1.05


def _thumb_extended(landmarks: NDArray[np.floating[Any]]) -> bool:
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


def _thumb_up(landmarks: NDArray[np.floating[Any]]) -> bool:
    """Thumb tip is well above the wrist (y decreases upward in image space)."""
    thumb_tip = landmarks[THUMB_TIP]
    wrist = landmarks[WRIST]
    # Thumb must extend upward by at least ~10% of image height.
    return bool((wrist[1] - thumb_tip[1]) > 0.1)


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


def classify_hand(landmarks: NDArray[np.floating[Any]]) -> Gesture:
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
    if thumb and not index and not middle and not ring and not pinky and _thumb_up(landmarks):
        return "thumbs_up"

    # ok: thumb + index tips touching; middle/ring/pinky extended.
    if _thumb_index_touching(landmarks) and middle and ring and pinky:
        return "ok"

    # open: all five extended.
    if extended_count == 5:
        return "open"

    # fist: all four non-thumb fingers curled. Accept thumb in any state since
    # some people tuck the thumb inside the fist, others rest it on the index.
    if not index and not middle and not ring and not pinky:
        return "fist"

    # peace: index + middle extended; ring + pinky curled.
    if index and middle and not ring and not pinky:
        return "peace"

    # rock: index + pinky extended, middle + ring curled.
    if index and not middle and not ring and pinky:
        return "rock"

    # three: index + middle + ring extended; pinky curled.
    if index and middle and ring and not pinky:
        return "three"

    # point: only index extended among the four non-thumbs. Thumb state doesn't matter.
    if index and not middle and not ring and not pinky:
        return "point"

    return "none"
