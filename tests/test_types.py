"""Type dataclasses should be frozen and round-trip cleanly."""

from handspring.types import (
    FaceFeatures,
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
    MotionState,
    PoseLandmark,
    PoseState,
)


def test_hand_features_frozen():
    hf = HandFeatures(
        x=0.5,
        y=0.5,
        z=0.0,
        openness=0.8,
        pinch=0.1,
        index_x=0.5,
        index_y=0.5,
        thumb_x=0.5,
        thumb_y=0.5,
    )
    import pytest

    with pytest.raises(AttributeError):
        hf.x = 0.0  # type: ignore[misc]


def test_hand_state_absent():
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    hs = HandState(present=False, features=None, gesture="none", motion=m)
    assert hs.present is False
    assert hs.features is None
    assert hs.gesture == "none"


def test_hand_state_present():
    hf = HandFeatures(
        x=0.1,
        y=0.2,
        z=0.3,
        openness=0.9,
        pinch=0.05,
        index_x=0.1,
        index_y=0.2,
        thumb_x=0.1,
        thumb_y=0.2,
    )
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    hs = HandState(present=True, features=hf, gesture="open", motion=m)
    assert hs.present is True
    assert hs.features == hf
    assert hs.gesture == "open"


def test_face_state_absent():
    fs = FaceState(
        present=False, features=None, expression="neutral", eye_left_open=0.0, eye_right_open=0.0
    )
    assert fs.present is False
    assert fs.features is None


def test_face_features_ranges():
    ff = FaceFeatures(yaw=-0.3, pitch=0.1, mouth_open=0.5)
    assert -1.0 <= ff.yaw <= 1.0
    assert -1.0 <= ff.pitch <= 1.0
    assert 0.0 <= ff.mouth_open <= 1.0


def test_frame_result_composition():
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    left = HandState(present=False, features=None, gesture="none", motion=m)
    right = HandState(present=False, features=None, gesture="none", motion=m)
    face = FaceState(
        present=False, features=None, expression="neutral", eye_left_open=0.0, eye_right_open=0.0
    )
    pose = PoseState(present=False, joints=None)
    fr = FrameResult(left=left, right=right, face=face, pose=pose, fps=30.0, clap_event=False)
    assert fr.fps == 30.0
    assert fr.left.gesture == "none"


def test_pose_landmark_frozen():
    pl = PoseLandmark(x=0.4, y=0.5, z=-0.1, visible=True)
    import pytest

    with pytest.raises(AttributeError):
        pl.x = 0.0  # type: ignore[misc]


def test_pose_state_absent():
    ps = PoseState(present=False, joints=None)
    assert ps.present is False
    assert ps.joints is None


def test_pose_state_present_with_joints():
    pl = PoseLandmark(x=0.4, y=0.5, z=0.0, visible=True)
    ps = PoseState(present=True, joints={"shoulder_left": pl})
    assert ps.present is True
    assert ps.joints is not None
    assert ps.joints["shoulder_left"].x == 0.4


def test_frame_result_has_pose():
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    left = HandState(present=False, features=None, gesture="none", motion=m)
    right = HandState(present=False, features=None, gesture="none", motion=m)
    face = FaceState(
        present=False, features=None, expression="neutral", eye_left_open=0.0, eye_right_open=0.0
    )
    pose = PoseState(present=False, joints=None)
    fr = FrameResult(left=left, right=right, face=face, pose=pose, fps=30.0, clap_event=False)
    assert fr.pose.present is False


def test_motion_state_default():
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    assert m.pinching is False
    assert m.event is None


def test_motion_state_with_event():
    m = MotionState(pinching=True, dragging=False, drag_dx=0.0, drag_dy=0.0, event="pinch")
    assert m.event == "pinch"


def test_hand_state_has_motion():
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    hf = HandFeatures(
        x=0.5,
        y=0.5,
        z=0.0,
        openness=0.5,
        pinch=0.0,
        index_x=0.5,
        index_y=0.5,
        thumb_x=0.5,
        thumb_y=0.5,
    )
    hs = HandState(present=True, features=hf, gesture="open", motion=m)
    assert hs.motion.pinching is False


def test_face_state_has_expression_and_eye_open():
    fs = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    assert fs.expression == "neutral"
    assert fs.eye_left_open == 0.0


def test_frame_result_has_clap_event():
    m = MotionState(pinching=False, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    hs = HandState(present=False, features=None, gesture="none", motion=m)
    fs = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    ps = PoseState(present=False, joints=None)
    fr = FrameResult(left=hs, right=hs, face=fs, pose=ps, fps=30.0, clap_event=False)
    assert fr.clap_event is False
