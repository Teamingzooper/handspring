"""OpenCV preview window showing a unified neon-green skeleton overlay."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from handspring.app_mode import AppMode
from handspring.jarvis import WINDOW_COLORS, JarvisController
from handspring.synth_params import SynthSnapshot
from handspring.synth_ui import UiHint
from handspring.types import FrameResult

# MediaPipe connection sets (tuples of (src_idx, dst_idx)).
_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
_FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_CONTOURS
_POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# Shared "neon-green skeleton" colors (BGR, OpenCV order).
_BONE_COLOR = (136, 255, 0)  # #00FF88 bright green
_DOT_COLOR = (102, 204, 153)  # #99CC66 desaturated green
_SHADOW_COLOR = (0, 0, 0)

_BONE_THICKNESS = 3
_BONE_SHADOW_THICKNESS = 5  # draw shadow first for contrast on bright backgrounds
_DOT_RADIUS = 3

# Drawing specs for hand + face (MediaPipe drawing_utils path).
_DOT_SPEC = mp.solutions.drawing_utils.DrawingSpec(
    color=_DOT_COLOR, thickness=-1, circle_radius=_DOT_RADIUS
)
_LINE_SPEC = mp.solutions.drawing_utils.DrawingSpec(color=_BONE_COLOR, thickness=_BONE_THICKNESS)


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
        synth_snapshot: SynthSnapshot | None,
        synth_hint: UiHint | None,
        app_mode: AppMode,
        jarvis: JarvisController | None,
    ) -> bool:
        display = bgr_frame.copy()
        if self._mirror:
            display = cv2.flip(display, 1)  # type: ignore[assignment]
            hand_landmark_lists = [_mirror_landmarks(ll) for ll in hand_landmark_lists]
            face_landmark_lists = [_mirror_landmarks(ll) for ll in face_landmark_lists]
            if pose_landmarks is not None:
                pose_landmarks = _mirror_landmarks(pose_landmarks)

        # Draw pose first (background layer) using our own cv2-based path that
        # does NOT filter by MediaPipe's visibility threshold. Even partially-
        # visible bodies produce a readable skeleton.
        if pose_landmarks is not None:
            _draw_pose_skeleton(display, pose_landmarks)

        # Hand + face use MediaPipe's drawing_utils; it works fine because the
        # mirror helper below preserves HasField semantics on visibility and
        # presence (i.e., doesn't spuriously set them to 0.0 for landmarks
        # that didn't have them set originally).
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

        # Mode badge always visible.
        _draw_mode_badge(display, app_mode)

        if app_mode == "jarvis" and jarvis is not None:
            _draw_jarvis(display, jarvis, mirrored=self._mirror)
        # Synth panel only in synth mode.
        if app_mode == "synth" and synth_snapshot is not None:
            _draw_synth_panel(display, synth_snapshot)
        if app_mode == "synth" and synth_hint is not None and synth_hint.kind != "none":
            _draw_synth_hint(display, synth_hint, mirrored=self._mirror)

        if not self._created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            self._created = True
        cv2.imshow(self.WINDOW_NAME, display)

        if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)  # q or ESC

    def close(self) -> None:
        if self._created:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._created = False


def _draw_pose_skeleton(frame: NDArray[np.uint8], pose_landmarks: Any) -> None:
    """Draw pose connections directly with cv2 so visibility thresholds don't
    silently drop whole limbs. Each bone gets a dark shadow underneath for
    readability on bright backgrounds."""
    h, w = frame.shape[:2]
    lm = list(pose_landmarks.landmark)

    def _point(i: int) -> tuple[int, int]:
        p = lm[i]
        return int(p.x * w), int(p.y * h)

    # Shadow pass — thicker black line underneath everything.
    for a, b in _POSE_CONNECTIONS:
        cv2.line(frame, _point(a), _point(b), _SHADOW_COLOR, _BONE_SHADOW_THICKNESS, cv2.LINE_AA)
    # Bone pass — neon green on top.
    for a, b in _POSE_CONNECTIONS:
        cv2.line(frame, _point(a), _point(b), _BONE_COLOR, _BONE_THICKNESS, cv2.LINE_AA)
    # Joint dots on top of lines.
    for i in range(len(lm)):
        cv2.circle(frame, _point(i), _DOT_RADIUS, _DOT_COLOR, -1, cv2.LINE_AA)


def _draw_status(frame: NDArray[np.uint8], frame_result: FrameResult, osc_target: str) -> None:
    lines = [
        f"FPS: {frame_result.fps:5.1f}",
        f"OSC -> {osc_target}",
        f"Left:  {frame_result.left.gesture if frame_result.left.present else '-'}",
        f"Right: {frame_result.right.gesture if frame_result.right.present else '-'}",
        f"Pose:  {'on' if frame_result.pose.present else '-'}",
        f"Face:  {frame_result.face.expression if frame_result.face.present else '-'}",
    ]
    y = 30
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA
        )
        y += 24


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
            frame,
            _JARVIS_HINT_TEXT,
            (hcx, cy + 28),
            font,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            _JARVIS_HINT_TEXT,
            (hcx, cy + 28),
            font,
            0.45,
            (180, 220, 180),
            1,
            cv2.LINE_AA,
        )


def _draw_jarvis(frame: NDArray[np.uint8], jarvis: JarvisController, *, mirrored: bool) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    alpha = 0.35

    for win in jarvis.manager.windows():
        # Translate normalized coords to pixels; apply mirror if needed.
        x0_n = 1.0 - (win.x + win.width) if mirrored else win.x
        x0 = int(x0_n * w)
        y0 = int(win.y * h)
        x1 = int((x0_n + win.width) * w)
        y1 = int((win.y + win.height) * h)

        fill = WINDOW_COLORS[win.color_idx]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1)

    # Composite overlay onto frame with alpha.
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    # Borders, titles drawn on top (fully opaque).
    for win in jarvis.manager.windows():
        x0_n = 1.0 - (win.x + win.width) if mirrored else win.x
        x0 = int(x0_n * w)
        y0 = int(win.y * h)
        x1 = int((x0_n + win.width) * w)
        y1 = int((win.y + win.height) * h)

        border = WINDOW_COLORS[win.color_idx]
        cv2.rectangle(frame, (x0, y0), (x1, y1), border, 2)
        # Title bar
        cv2.rectangle(frame, (x0, y0), (x1, min(y0 + 22, y1)), border, -1)
        cv2.putText(
            frame,
            f"Window {win.id}",
            (x0 + 8, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


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
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
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
        _draw_slider(
            frame, cx=cx, cy=cy, label=hint.label_a, value=hint.value_a, display=hint.display_a
        )
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
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA
    )


def _mirror_landmarks(landmark_list: Any) -> Any:
    """Return a new landmark list with x coordinates mirrored to 1 - x.

    Preserves HasField semantics on visibility / presence — if the source
    landmark did NOT have these fields set (as is the case for hand and
    face landmarks), the mirrored landmark also leaves them unset. This
    prevents MediaPipe's draw_landmarks from spuriously skipping the
    landmark because of a 0.0 visibility score that the original didn't
    assert.
    """
    from mediapipe.framework.formats import landmark_pb2

    mirrored = landmark_pb2.NormalizedLandmarkList()
    for lm in landmark_list.landmark:
        new_lm = mirrored.landmark.add()
        new_lm.x = 1.0 - lm.x
        new_lm.y = lm.y
        new_lm.z = lm.z
        if lm.HasField("visibility"):
            new_lm.visibility = lm.visibility
        if lm.HasField("presence"):
            new_lm.presence = lm.presence
    return mirrored
