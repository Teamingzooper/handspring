"""MediaPipe wrapper: accepts BGR frames, returns FrameResult."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from handspring.expressions import classify_expression, eye_open_values
from handspring.features import face_features, hand_features
from handspring.gestures import classify_hand
from handspring.history import HandHistory
from handspring.motion import MotionDetector, bi_hand_clap_detector
from handspring.types import (
    FaceState,
    FrameResult,
    HandState,
    Joint,
    MotionState,
    PoseLandmark,
    PoseState,
    Side,
)

_POSE_JOINTS: dict[Joint, int] = {
    "shoulder_left": 12,
    "shoulder_right": 11,
    "elbow_left": 14,
    "elbow_right": 13,
    "wrist_left": 16,
    "wrist_right": 15,
    "hip_left": 24,
    "hip_right": 23,
}

_VISIBILITY_THRESHOLD = 0.5
_HISTORY_CAPACITY = 30


@dataclass
class TrackerConfig:
    max_hands: int = 2
    track_face: bool = True
    track_pose: bool = True
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


class Tracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self._config.max_hands,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        if self._config.track_face:
            self._face_mesh: Any = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,  # NEW: needed for accurate eye/lip landmarks
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
        else:
            self._face_mesh = None

        if self._config.track_pose:
            self._pose: Any = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
        else:
            self._pose = None

        self._left_history = HandHistory(capacity=_HISTORY_CAPACITY)
        self._right_history = HandHistory(capacity=_HISTORY_CAPACITY)
        self._left_motion = MotionDetector()
        self._right_motion = MotionDetector()
        self._clap_detector = bi_hand_clap_detector()

        self._start_time = time.perf_counter()
        self._last_frame_time: float | None = None
        self._fps_ema: float = 0.0

    def process(self, bgr_frame: NDArray[np.uint8]) -> FrameResult:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        hand_results = self._hands.process(rgb)
        face_result: Any = self._face_mesh.process(rgb) if self._face_mesh is not None else None
        pose_result: Any = self._pose.process(rgb) if self._pose is not None else None

        now = time.perf_counter() - self._start_time

        left_state, right_state = self._hand_states(hand_results, now)
        face_state = self._face_state(face_result)
        pose_state = self._pose_state(pose_result)

        clap_event = False
        if (
            left_state.present
            and right_state.present
            and left_state.features
            and right_state.features
        ):
            dx = left_state.features.x - right_state.features.x
            dy = left_state.features.y - right_state.features.y
            distance = math.hypot(dx, dy)
            clap_event = self._clap_detector.update(distance, now)

        # FPS EMA
        fps = 0.0
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                instant = 1.0 / dt
                self._fps_ema = 0.9 * self._fps_ema + 0.1 * instant if self._fps_ema else instant
                fps = self._fps_ema
        self._last_frame_time = now

        return FrameResult(
            left=left_state,
            right=right_state,
            face=face_state,
            pose=pose_state,
            fps=fps,
            clap_event=clap_event,
        )

    def close(self) -> None:
        self._hands.close()
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._pose is not None:
            self._pose.close()

    def _hand_states(self, hand_results: Any, now: float) -> tuple[HandState, HandState]:
        absent_motion = MotionState(
            pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None
        )
        absent = HandState(present=False, features=None, gesture="none", motion=absent_motion)

        left = absent
        right = absent
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for landmarks, handedness in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
                strict=False,
            ):
                label: str = handedness.classification[0].label
                side: Side = "right" if label == "Left" else "left"
                arr = _landmark_list_to_array(landmarks)
                feats = hand_features(arr)
                gesture = classify_hand(arr)

                history = self._left_history if side == "left" else self._right_history
                detector = self._left_motion if side == "left" else self._right_motion
                history.push(feats, now)
                m = detector.update(history, now, gesture=gesture)
                motion = MotionState(
                    pinching=m.pinching,
                    dragging=m.dragging,
                    drag_dx=m.drag_dx,
                    drag_dy=m.drag_dy,
                    event=m.event,
                )
                state = HandState(present=True, features=feats, gesture=gesture, motion=motion)
                if side == "left":
                    left = state
                else:
                    right = state

        # Tick motion detectors even when a hand is absent so state doesn't stale.
        if not left.present:
            self._left_motion.update(self._left_history, now)
        if not right.present:
            self._right_motion.update(self._right_history, now)

        return left, right

    def _face_state(self, face_result: Any) -> FaceState:
        if face_result is None or not face_result.multi_face_landmarks:
            return FaceState(
                present=False,
                features=None,
                expression="neutral",
                eye_left_open=0.0,
                eye_right_open=0.0,
            )
        lm = face_result.multi_face_landmarks[0]
        arr = _landmark_list_to_array(lm)
        feats = face_features(arr)
        expression = classify_expression(arr)
        left_open, right_open = eye_open_values(arr)
        return FaceState(
            present=True,
            features=feats,
            expression=expression,
            eye_left_open=left_open,
            eye_right_open=right_open,
        )

    def _pose_state(self, pose_result: Any) -> PoseState:
        if pose_result is None or pose_result.pose_landmarks is None:
            return PoseState(present=False, joints=None)
        landmarks = pose_result.pose_landmarks.landmark
        joints: dict[Joint, PoseLandmark] = {}
        for joint_name, mp_idx in _POSE_JOINTS.items():
            lm = landmarks[mp_idx]
            joints[joint_name] = PoseLandmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visible=float(lm.visibility) >= _VISIBILITY_THRESHOLD,
            )
        return PoseState(present=True, joints=joints)


def _landmark_list_to_array(landmark_list: Any) -> NDArray[np.float32]:
    count = len(landmark_list.landmark)
    arr = np.zeros((count, 3), dtype=np.float32)
    for i, lm in enumerate(landmark_list.landmark):
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
    return arr
