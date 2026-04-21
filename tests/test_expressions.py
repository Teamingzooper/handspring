"""Expression classifier tests."""

from __future__ import annotations

from handspring.expressions import classify_expression, eye_open_values
from tests.fixtures import (
    face_frown,
    face_neutral,
    face_smile,
    face_surprise,
    face_wink_left,
    face_wink_right,
)


def test_neutral_classifies_neutral():
    assert classify_expression(face_neutral()) == "neutral"


def test_smile_classifies_smile():
    assert classify_expression(face_smile()) == "smile"


def test_frown_classifies_frown():
    assert classify_expression(face_frown()) == "frown"


def test_surprise_classifies_surprise():
    assert classify_expression(face_surprise()) == "surprise"


def test_wink_left_classifies_wink_left():
    assert classify_expression(face_wink_left()) == "wink_left"


def test_wink_right_classifies_wink_right():
    assert classify_expression(face_wink_right()) == "wink_right"


def test_eye_open_values_closed_low():
    left, right = eye_open_values(face_wink_left())
    assert left < 0.2
    assert right > 0.5


def test_eye_open_values_open_high():
    left, right = eye_open_values(face_neutral())
    assert left > 0.3
    assert right > 0.3
