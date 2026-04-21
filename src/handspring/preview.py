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
    color=(102, 204, 153),  # BGR for #99CC66 — desaturated green dots
    thickness=-1,
    circle_radius=3,
)
_LINE_SPEC = mp.solutions.drawing_utils.DrawingSpec(
    color=(136, 255, 0),  # BGR for #00FF88 — bright green bones
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
            display = cv2.flip(display, 1)  # type: ignore[assignment]
            hand_landmark_lists = [_mirror_landmarks(ll) for ll in hand_landmark_lists]
            face_landmark_lists = [_mirror_landmarks(ll) for ll in face_landmark_lists]
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
        return key not in (ord("q"), 27)  # q or ESC

    def close(self) -> None:
        if self._created:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._created = False


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
