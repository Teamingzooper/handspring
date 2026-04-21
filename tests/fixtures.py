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
