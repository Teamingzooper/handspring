"""OpenCV preview window showing tracking overlay on the camera feed."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from handspring.types import FrameResult

_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
_FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_CONTOURS


class Preview:
    """Wraps the OpenCV preview window — manages the HighGUI window lifecycle."""

    WINDOW_NAME = "handspring"

    def __init__(self, *, mirror: bool = True) -> None:
        self._mirror = mirror
        self._created = False

    def render(
        self,
        bgr_frame: NDArray[np.uint8],
        hand_landmark_lists: list[Any],
        face_landmark_lists: list[Any],
        frame_result: FrameResult,
        osc_target: str,
    ) -> bool:
        """Draw one frame. Returns True to keep going, False if the user
        pressed 'q' or closed the window."""
        display: NDArray[np.uint8] = bgr_frame.copy()
        if self._mirror:
            display = cv2.flip(display, 1)  # type: ignore[assignment]
            # When mirroring, landmark x coordinates need flipping for the
            # overlay to align.
            hand_landmark_lists = [
                _mirror_landmarks(ll, display.shape[1]) for ll in hand_landmark_lists
            ]
            face_landmark_lists = [
                _mirror_landmarks(ll, display.shape[1]) for ll in face_landmark_lists
            ]

        for ll in hand_landmark_lists:
            mp.solutions.drawing_utils.draw_landmarks(display, ll, _HAND_CONNECTIONS)
        for ll in face_landmark_lists:
            mp.solutions.drawing_utils.draw_landmarks(
                display,
                ll,
                _FACE_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(160, 160, 160), thickness=1
                ),
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
    ]
    y = 30
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA
        )
        y += 24


def _mirror_landmarks(landmark_list: Any, _width: int) -> Any:
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
