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

_PINCH_CLOSE_DISTANCE = 0.05  # geometric threshold for "pinching" visual
_CREATE_READY_DISTANCE = 0.08  # Jarvis create-entry threshold (matches jarvis.py)

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

        _draw_pinch_viz(display, hand_landmark_lists)

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
        # Top-right corner resize handle.
        resizing_id = jarvis.resizing_window_id()
        handle_color = (
            (136, 255, 0) if resizing_id == win.id else (100, 200, 255)
        )  # green when active, cyan otherwise
        handle_size = 10 if resizing_id == win.id else 8
        hx = x1
        hy = y0
        cv2.rectangle(
            frame,
            (hx - handle_size, hy),
            (hx, hy + handle_size),
            handle_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (hx - handle_size, hy),
            (hx, hy + handle_size),
            (30, 30, 30),
            1,
        )

    pending = jarvis.pending_rect()
    if pending is not None:
        px, py, pw, ph = pending
        x0_n = 1.0 - (px + pw) if mirrored else px
        x0 = int(x0_n * w)
        y0 = int(py * h)
        x1 = int((x0_n + pw) * w)
        y1 = int((py + ph) * h)

        # Semi-transparent fill (same overlay approach as real windows, but redo the blend locally).
        ghost = frame.copy()
        cv2.rectangle(ghost, (x0, y0), (x1, y1), WINDOW_COLORS[0], -1)
        cv2.addWeighted(ghost, 0.20, frame, 0.80, 0, dst=frame)

        # Dashed border using _dotted_line (already defined in preview.py).
        _dotted_line(frame, (x0, y0), (x1, y0), (136, 255, 0), thickness=2, gap=10)
        _dotted_line(frame, (x1, y0), (x1, y1), (136, 255, 0), thickness=2, gap=10)
        _dotted_line(frame, (x1, y1), (x0, y1), (136, 255, 0), thickness=2, gap=10)
        _dotted_line(frame, (x0, y1), (x0, y0), (136, 255, 0), thickness=2, gap=10)

        # "NEW" label near top-left.
        _label_with_shadow(frame, x0 + 8, y0 + 18, "NEW", (136, 255, 0))


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
        _draw_slider(
            frame,
            cx=cx,
            label=hint.label_a,
            value=hint.value_a,
            display=hint.display_a,
            live_y=hint.live_y,
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
    label: str,
    value: float,
    display: str,
    live_y: float,
) -> None:
    h, w = frame.shape[:2]
    width = 36
    y_top = int(0.10 * h)
    y_bottom = int(0.90 * h)
    x0 = cx
    # Keep the slider on-screen: clamp x0 so it doesn't run off the right edge.
    x0 = min(x0, w - width - 180)  # leave room for label on the right
    x0 = max(x0, 12)

    # Track (dark background)
    cv2.rectangle(frame, (x0, y_top), (x0 + width, y_bottom), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, y_top), (x0 + width, y_bottom), (136, 255, 0), 2)

    # Fill from y_bottom up to the fingertip Y (clamped into the track).
    live_y_clamped = max(0.0, min(1.0, live_y))
    fill_top_px = int(live_y_clamped * h)
    fill_top_px = max(y_top + 2, min(y_bottom - 2, fill_top_px))
    cv2.rectangle(
        frame,
        (x0 + 2, fill_top_px),
        (x0 + width - 2, y_bottom - 2),
        (136, 255, 0),
        -1,
    )

    # Horizontal "fingertip indicator" tick on the fill's top edge — extends a bit outside the track.
    cv2.line(
        frame,
        (x0 - 6, fill_top_px),
        (x0 + width + 6, fill_top_px),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Label (parameter name) + value text to the right of the slider.
    label_x = x0 + width + 12
    cv2.putText(
        frame,
        label,
        (label_x, y_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (label_x, y_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        display,
        (label_x, y_top + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        display,
        (label_x, y_top + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (136, 255, 0),
        1,
        cv2.LINE_AA,
    )


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


def _dotted_line(
    frame: NDArray[np.uint8],
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    gap: int = 8,
) -> None:
    """Draw a dotted line from pt1 to pt2 with short dashes separated by `gap` px."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = int(np.hypot(x2 - x1, y2 - y1))
    if dist == 0:
        return
    n_dots = max(dist // gap, 1)
    for i in range(n_dots):
        t = i / max(n_dots - 1, 1)
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        cv2.circle(frame, (x, y), thickness + 1, color, -1, cv2.LINE_AA)


def _label_with_shadow(
    frame: NDArray[np.uint8], x: int, y: int, text: str, color: tuple[int, int, int]
) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_pinch_viz(
    frame: NDArray[np.uint8],
    hand_landmark_lists: list[Any],  # already in display coords (mirroring already applied)
) -> None:
    """Draw per-hand thumb→index pinch lines and an inter-hand line when both pinching."""
    h, w = frame.shape[:2]
    close_index_tips: list[tuple[int, int]] = []

    for ll in hand_landmark_lists:
        thumb = ll.landmark[4]
        index = ll.landmark[8]
        tx_n, ty_n = thumb.x, thumb.y
        ix_n, iy_n = index.x, index.y
        dist = float(np.hypot(ix_n - tx_n, iy_n - ty_n))
        is_close = dist < _PINCH_CLOSE_DISTANCE

        tx_px = int(tx_n * w)
        ty_px = int(ty_n * h)
        ix_px = int(ix_n * w)
        iy_px = int(iy_n * h)

        if is_close:
            cv2.line(frame, (tx_px, ty_px), (ix_px, iy_px), (136, 255, 0), 3, cv2.LINE_AA)
            mid = ((tx_px + ix_px) // 2, (ty_px + iy_px) // 2)
            cv2.circle(frame, mid, 6, (136, 255, 0), -1, cv2.LINE_AA)
            close_index_tips.append((ix_px, iy_px))
        else:
            _dotted_line(frame, (tx_px, ty_px), (ix_px, iy_px), (150, 150, 150), 1, gap=10)

        # Distance label slightly above the midpoint.
        mx = (tx_px + ix_px) // 2
        my = (ty_px + iy_px) // 2 - 14
        text = f"{dist:.3f}"
        color = (136, 255, 0) if is_close else (200, 200, 200)
        _label_with_shadow(frame, mx, my, text, color)

    # Inter-hand line when both hands are geometrically "close" (pinching).
    if len(close_index_tips) == 2:
        p1, p2 = close_index_tips
        # Compute normalized distance using pixel distance / frame width.
        px_dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        norm_dist = px_dist / w
        ready = norm_dist < _CREATE_READY_DISTANCE
        line_color = (136, 255, 0) if ready else (0, 230, 230)
        if ready:
            cv2.line(frame, p1, p2, line_color, 3, cv2.LINE_AA)
        else:
            _dotted_line(frame, p1, p2, line_color, 1, gap=10)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 14)
        _label_with_shadow(frame, mid[0], mid[1], f"{norm_dist:.3f}", line_color)


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
