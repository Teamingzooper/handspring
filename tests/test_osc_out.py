"""OSC emitter tests — assert message content and dedup behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from handspring.osc_out import OscEmitter
from handspring.synth_params import SynthSnapshot
from handspring.types import (
    FaceFeatures,
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
    Joint,
    MotionState,
    PoseLandmark,
    PoseState,
)


@dataclass
class FakeOsc:
    sent: list[tuple[str, Any]]

    def send_message(self, address: str, value: Any) -> None:
        self.sent.append((address, value))


def _frame(
    left_gesture: str = "none",
    right_gesture: str = "none",
    left_present: bool = True,
    right_present: bool = True,
    face_present: bool = True,
) -> FrameResult:
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0, index_x=0.5, index_y=0.5)
    _m = MotionState(False, False, 0.0, 0.0, None)
    left = HandState(
        present=left_present,
        features=hf if left_present else None,
        gesture=left_gesture,  # type: ignore[arg-type]
        motion=_m,
    )
    right = HandState(
        present=right_present,
        features=hf if right_present else None,
        gesture=right_gesture,  # type: ignore[arg-type]
        motion=_m,
    )
    face = FaceState(
        present=face_present,
        features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=0.0) if face_present else None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    return FrameResult(
        left=left,
        right=right,
        face=face,
        pose=PoseState(present=False, joints=None),
        fps=30.0,
        clap_event=False,
    )


def test_continuous_features_emitted_every_frame():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame())
    addresses = [addr for addr, _ in fake.sent]
    assert "/hand/left/x" in addresses
    assert "/hand/left/y" in addresses
    assert "/hand/left/openness" in addresses
    assert "/hand/right/x" in addresses
    assert "/face/yaw" in addresses
    assert "/face/mouth_open" in addresses


def test_present_flags_are_int():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame())
    for addr, value in fake.sent:
        if addr.endswith("/present"):
            assert value in (0, 1), f"{addr} sent non-int: {value}"


def test_gesture_only_emitted_on_change():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame(left_gesture="fist"))
    emitter.emit(_frame(left_gesture="fist"))
    emitter.emit(_frame(left_gesture="fist"))
    gesture_msgs = [(a, v) for a, v in fake.sent if a == "/hand/left/gesture"]
    assert gesture_msgs == [("/hand/left/gesture", "fist")], (
        f"expected one gesture message, got {gesture_msgs}"
    )


def test_gesture_re_emitted_on_transition():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame(left_gesture="fist"))
    emitter.emit(_frame(left_gesture="open"))
    gesture_msgs = [v for a, v in fake.sent if a == "/hand/left/gesture"]
    assert gesture_msgs == ["fist", "open"]


def test_absent_hand_emits_present_zero_no_features():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame(left_present=False))
    addresses = [addr for addr, _ in fake.sent]
    assert ("/hand/left/present", 0) in fake.sent
    assert "/hand/left/x" not in addresses
    assert "/hand/left/openness" not in addresses


def test_absent_face_emits_present_zero():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame(face_present=False))
    assert ("/face/present", 0) in fake.sent
    assert "/face/yaw" not in [a for a, _ in fake.sent]


def _pose(joints: dict[Joint, PoseLandmark] | None) -> PoseState:
    return PoseState(present=joints is not None, joints=joints)


def _frame_with_pose(pose: PoseState) -> FrameResult:
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0, index_x=0.5, index_y=0.5)
    _m = MotionState(False, False, 0.0, 0.0, None)
    left = HandState(present=True, features=hf, gesture="none", motion=_m)
    right = HandState(present=True, features=hf, gesture="none", motion=_m)
    face = FaceState(
        present=True,
        features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=0.0),
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    return FrameResult(left=left, right=right, face=face, pose=pose, fps=30.0, clap_event=False)


def test_pose_present_emits_joint_messages():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    joints: dict[Joint, PoseLandmark] = {
        "shoulder_left": PoseLandmark(x=0.3, y=0.4, z=0.0, visible=True),
        "shoulder_right": PoseLandmark(x=0.7, y=0.4, z=0.0, visible=True),
        "elbow_left": PoseLandmark(x=0.25, y=0.55, z=0.0, visible=True),
        "elbow_right": PoseLandmark(x=0.75, y=0.55, z=0.0, visible=True),
        "wrist_left": PoseLandmark(x=0.2, y=0.7, z=0.0, visible=True),
        "wrist_right": PoseLandmark(x=0.8, y=0.7, z=0.0, visible=True),
        "hip_left": PoseLandmark(x=0.4, y=0.85, z=0.0, visible=True),
        "hip_right": PoseLandmark(x=0.6, y=0.85, z=0.0, visible=True),
    }
    emitter.emit(_frame_with_pose(_pose(joints)))
    addresses = [addr for addr, _ in fake.sent]
    assert ("/pose/present", 1) in fake.sent
    for joint in joints:
        assert f"/pose/{joint}/visible" in addresses
        assert f"/pose/{joint}/x" in addresses
        assert f"/pose/{joint}/y" in addresses
        assert f"/pose/{joint}/z" in addresses


def test_pose_absent_emits_only_present_zero():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame_with_pose(_pose(None)))
    pose_messages = [a for a, _ in fake.sent if a.startswith("/pose/")]
    assert pose_messages == ["/pose/present"]
    assert ("/pose/present", 0) in fake.sent


def test_pose_joint_invisible_suppresses_xyz():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    joints: dict[Joint, PoseLandmark] = {
        "shoulder_left": PoseLandmark(x=0.3, y=0.4, z=0.0, visible=True),
        "shoulder_right": PoseLandmark(x=0.7, y=0.4, z=0.0, visible=False),
        "elbow_left": PoseLandmark(x=0.25, y=0.55, z=0.0, visible=True),
        "elbow_right": PoseLandmark(x=0.75, y=0.55, z=0.0, visible=True),
        "wrist_left": PoseLandmark(x=0.2, y=0.7, z=0.0, visible=True),
        "wrist_right": PoseLandmark(x=0.8, y=0.7, z=0.0, visible=True),
        "hip_left": PoseLandmark(x=0.4, y=0.85, z=0.0, visible=True),
        "hip_right": PoseLandmark(x=0.6, y=0.85, z=0.0, visible=True),
    }
    emitter.emit(_frame_with_pose(_pose(joints)))
    assert ("/pose/shoulder_right/visible", 0) in fake.sent
    addresses = [addr for addr, _ in fake.sent]
    assert "/pose/shoulder_right/x" not in addresses
    assert "/pose/shoulder_right/y" not in addresses
    assert "/pose/shoulder_right/z" not in addresses
    assert "/pose/shoulder_left/x" in addresses


def test_motion_continuous_state_emitted():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0, index_x=0.5, index_y=0.5)
    m = MotionState(pinching=True, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None)
    left = HandState(present=True, features=hf, gesture="none", motion=m)
    right = HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    face = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    pose = PoseState(present=False, joints=None)
    emitter.emit(
        FrameResult(left=left, right=right, face=face, pose=pose, fps=30.0, clap_event=False)
    )
    assert ("/hand/left/pinching", 1) in fake.sent
    assert ("/hand/left/dragging", 0) in fake.sent


def test_motion_event_fires():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0, index_x=0.5, index_y=0.5)
    m = MotionState(pinching=True, dragging=False, drag_dx=0.0, drag_dy=0.0, event="pinch")
    left = HandState(present=True, features=hf, gesture="none", motion=m)
    right = HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    face = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    pose = PoseState(present=False, joints=None)
    emitter.emit(FrameResult(left, right, face, pose, 30.0, False))
    assert ("/hand/left/event", "pinch") in fake.sent


def test_drag_dxdy_only_when_dragging():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0, index_x=0.5, index_y=0.5)
    m = MotionState(pinching=True, dragging=True, drag_dx=0.25, drag_dy=-0.1, event=None)
    left = HandState(present=True, features=hf, gesture="none", motion=m)
    right = HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    face = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    emitter.emit(FrameResult(left, right, face, PoseState(False, None), 30.0, False))
    assert ("/hand/left/drag_dx", 0.25) in fake.sent
    assert ("/hand/left/drag_dy", -0.1) in fake.sent


def test_clap_event_emits():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.5, pinch=0.0, index_x=0.5, index_y=0.5)
    m = MotionState(False, False, 0.0, 0.0, None)
    left = HandState(present=True, features=hf, gesture="none", motion=m)
    right = HandState(present=True, features=hf, gesture="none", motion=m)
    face = FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )
    emitter.emit(FrameResult(left, right, face, PoseState(False, None), 30.0, True))
    assert ("/motion/clap", 1) in fake.sent


def test_face_expression_event_on_change():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)

    def frame(expr):  # type: ignore[no-untyped-def]
        return FrameResult(
            left=HandState(
                present=False,
                features=None,
                gesture="none",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
            right=HandState(
                present=False,
                features=None,
                gesture="none",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
            face=FaceState(
                present=True,
                features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=0.0),
                expression=expr,
                eye_left_open=0.8,
                eye_right_open=0.8,
            ),
            pose=PoseState(False, None),
            fps=30.0,
            clap_event=False,
        )

    emitter.emit(frame("smile"))
    emitter.emit(frame("smile"))
    emitter.emit(frame("frown"))
    exprs = [v for a, v in fake.sent if a == "/face/expression"]
    assert exprs == ["smile", "frown"]


def test_face_eye_open_continuous():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    face = FaceState(
        present=True,
        features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=0.0),
        expression="neutral",
        eye_left_open=0.75,
        eye_right_open=0.92,
    )
    fr = FrameResult(
        left=HandState(False, None, "none", MotionState(False, False, 0.0, 0.0, None)),
        right=HandState(False, None, "none", MotionState(False, False, 0.0, 0.0, None)),
        face=face,
        pose=PoseState(False, None),
        fps=30.0,
        clap_event=False,
    )
    emitter.emit(fr)
    assert ("/face/eye_left_open", 0.75) in fake.sent
    assert ("/face/eye_right_open", 0.92) in fake.sent


def test_synth_snapshot_emitted():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    snap = SynthSnapshot(
        volume=0.65,
        note_hz=523.25,
        stepping_hz=4.0,
        cutoff_hz=2500.0,
        mod_depth=0.3,
        mod_rate=2.0,
        mode="edit_left",
    )
    emitter.emit_synth(snap)
    assert ("/synth/volume", 0.65) in fake.sent
    assert ("/synth/note_hz", 523.25) in fake.sent
    assert ("/synth/stepping_hz", 4.0) in fake.sent
    assert ("/synth/cutoff_hz", 2500.0) in fake.sent
    assert ("/synth/mod_depth", 0.3) in fake.sent
    assert ("/synth/mod_rate", 2.0) in fake.sent


def test_synth_mode_only_on_change():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)

    def snap(mode):  # type: ignore[no-untyped-def]
        return SynthSnapshot(
            volume=0.5,
            note_hz=440.0,
            stepping_hz=0.0,
            cutoff_hz=3000.0,
            mod_depth=0.0,
            mod_rate=1.0,
            mode=mode,
        )

    emitter.emit_synth(snap("play"))
    emitter.emit_synth(snap("play"))
    emitter.emit_synth(snap("edit_left"))
    emitter.emit_synth(snap("edit_left"))
    modes = [v for a, v in fake.sent if a == "/synth/mode"]
    assert modes == ["play", "edit_left"]


def test_index_tip_emitted():
    fake = FakeOsc(sent=[])
    emitter = OscEmitter(client=fake)
    emitter.emit(_frame())
    addresses = [addr for addr, _ in fake.sent]
    assert "/hand/left/index_x" in addresses
    assert "/hand/left/index_y" in addresses
    assert "/hand/right/index_x" in addresses
    assert "/hand/right/index_y" in addresses
