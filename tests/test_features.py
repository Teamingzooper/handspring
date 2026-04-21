"""Feature derivation tests."""

from __future__ import annotations

import numpy as np

from handspring.features import face_features, hand_features
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
