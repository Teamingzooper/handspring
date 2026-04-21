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
    ]
    y = 30
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA
        )
        y += 24


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
