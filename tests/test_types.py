"""Type dataclasses should be frozen and round-trip cleanly."""

from handspring.types import (
    FaceFeatures,
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
)


def test_hand_features_frozen():
    hf = HandFeatures(x=0.5, y=0.5, z=0.0, openness=0.8, pinch=0.1)
    import pytest

    with pytest.raises(AttributeError):
        hf.x = 0.0  # type: ignore[misc]


def test_hand_state_absent():
    hs = HandState(present=False, features=None, gesture="none")
    assert hs.present is False
    assert hs.features is None
    assert hs.gesture == "none"


def test_hand_state_present():
    hf = HandFeatures(x=0.1, y=0.2, z=0.3, openness=0.9, pinch=0.05)
    hs = HandState(present=True, features=hf, gesture="open")
    assert hs.present is True
    assert hs.features == hf
    assert hs.gesture == "open"


def test_face_state_absent():
    fs = FaceState(present=False, features=None)
    assert fs.present is False
    assert fs.features is None


def test_face_features_ranges():
    ff = FaceFeatures(yaw=-0.3, pitch=0.1, mouth_open=0.5)
    assert -1.0 <= ff.yaw <= 1.0
    assert -1.0 <= ff.pitch <= 1.0
    assert 0.0 <= ff.mouth_open <= 1.0


def test_frame_result_composition():
    left = HandState(present=False, features=None, gesture="none")
    right = HandState(present=False, features=None, gesture="none")
    face = FaceState(present=False, features=None)
    fr = FrameResult(left=left, right=right, face=face, fps=30.0)
    assert fr.fps == 30.0
    assert fr.left.gesture == "none"
