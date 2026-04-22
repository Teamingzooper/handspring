"""Feature derivation tests."""

from __future__ import annotations

import numpy as np

from handspring.features import face_features, hand_features, is_pinching
from handspring.types import HandFeatures, HandState, MotionState
from tests.fixtures import (
    face_closed_mouth,
    face_looking_down,
    face_looking_left,
    face_open_mouth,
    hand_fist,
    hand_open,
    hand_pinch,
)


def test_hand_features_xy_in_unit_range():
    f = hand_features(hand_open())
    assert 0.0 <= f.x <= 1.0
    assert 0.0 <= f.y <= 1.0


def test_hand_openness_fist_low():
    f = hand_features(hand_fist())
    assert f.openness < 0.3, f"fist openness too high: {f.openness}"


def test_hand_openness_open_high():
    f = hand_features(hand_open())
    assert f.openness > 0.7, f"open palm openness too low: {f.openness}"


def test_hand_pinch_high_when_coincident():
    f = hand_features(hand_pinch())
    assert f.pinch > 0.85, f"pinch value too low: {f.pinch}"


def test_hand_pinch_low_when_spread():
    f = hand_features(hand_open())
    assert f.pinch < 0.3, f"pinch value too high when spread: {f.pinch}"


def test_face_mouth_closed_low():
    f = face_features(face_closed_mouth())
    assert f.mouth_open < 0.1


def test_face_mouth_open_high():
    f = face_features(face_open_mouth())
    assert f.mouth_open > 0.3


def test_face_yaw_left_negative():
    f = face_features(face_looking_left())
    assert f.yaw < -0.1


def test_face_yaw_center_near_zero():
    f = face_features(face_closed_mouth())
    assert abs(f.yaw) < 0.1


def test_face_pitch_looking_down_is_negative():
    # Spec: negative pitch = looking down.
    f = face_features(face_looking_down())
    assert f.pitch < -0.05, f"down pitch not negative enough: {f.pitch}"


def test_hand_features_reject_nan():
    bad = np.full((21, 3), np.nan, dtype=np.float32)
    import pytest

    with pytest.raises(ValueError):
        hand_features(bad)


def test_hand_features_index_tip_extracted():
    from tests.fixtures import hand_open

    lm = hand_open()
    lm[8] = (0.73, 0.21, 0.0)  # INDEX_TIP = landmark 8
    f = hand_features(lm)
    assert abs(f.index_x - 0.73) < 1e-6
    assert abs(f.index_y - 0.21) < 1e-6


# ---------------------------------------------------------------------------
# is_pinching helper tests
# ---------------------------------------------------------------------------


def test_is_pinching_uses_raw_distance_not_pinch_feature():
    # Hand with thumb and index far apart but span-normalized pinch at 0.95
    # is NOT pinching under the new raw-distance rule.
    h = HandState(
        present=True,
        features=HandFeatures(
            x=0.5,
            y=0.5,
            z=0.0,
            openness=0.9,
            pinch=0.95,
            index_x=0.5,
            index_y=0.5,
            thumb_x=0.6,
            thumb_y=0.5,  # 0.1 away from index — too far
        ),
        gesture="open",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    assert is_pinching(h) is False


def test_is_pinching_true_when_thumb_and_index_close():
    h = HandState(
        present=True,
        features=HandFeatures(
            x=0.5,
            y=0.5,
            z=0.0,
            openness=0.9,
            pinch=0.0,
            index_x=0.5,
            index_y=0.5,
            thumb_x=0.51,
            thumb_y=0.51,  # 0.014 away — well within 0.05
        ),
        gesture="open",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    assert is_pinching(h) is True


def test_is_pinching_works_even_if_gesture_is_fist():
    # Pinch with other fingers curled — classifier may say "fist" but it's
    # still a pinch geometrically.
    h = HandState(
        present=True,
        features=HandFeatures(
            x=0.5,
            y=0.5,
            z=0.0,
            openness=0.1,
            pinch=0.0,
            index_x=0.5,
            index_y=0.5,
            thumb_x=0.505,
            thumb_y=0.505,  # extremely close
        ),
        gesture="fist",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    assert is_pinching(h) is True  # No fist exclusion — raw distance rules.


def test_is_pinching_false_when_hand_absent():
    absent = HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )
    assert is_pinching(absent) is False
