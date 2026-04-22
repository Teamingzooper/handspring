"""Gesture classifier tests."""

from __future__ import annotations

import numpy as np

from handspring.gestures import classify_hand
from tests.fixtures import (
    hand_fist,
    hand_fist_thumb_poking_out,
    hand_ok,
    hand_open,
    hand_peace,
    hand_pinch,
    hand_point,
    hand_rock,
    hand_three,
    hand_thumbs_up,
)


def test_fist_classifies_fist():
    assert classify_hand(hand_fist()) == "fist"


def test_open_classifies_open():
    assert classify_hand(hand_open()) == "open"


def test_point_classifies_point():
    assert classify_hand(hand_point()) == "point"


def test_peace_classifies_peace():
    assert classify_hand(hand_peace()) == "peace"


def test_thumbs_up_classifies_thumbs_up():
    assert classify_hand(hand_thumbs_up()) == "thumbs_up"


def test_pinch_is_not_a_classified_gesture():
    # Pinch is a continuous feature; the classifier should not return "pinch".
    # The pinch fixture has thumb + index extended, others curled — this
    # technically matches "point" (index only, thumb state ignored). Accept
    # "point" or "none" but NOT "pinch".
    result = classify_hand(hand_pinch())
    assert result != "pinch"


def test_classifier_is_deterministic():
    fx = hand_fist()
    results = {classify_hand(fx) for _ in range(100)}
    assert results == {"fist"}


def test_classifier_rejects_nan():
    bad = np.full((21, 3), np.nan, dtype=np.float32)
    import pytest

    with pytest.raises(ValueError):
        classify_hand(bad)


def test_ok_classifies_ok():
    assert classify_hand(hand_ok()) == "ok"


def test_rock_classifies_rock():
    assert classify_hand(hand_rock()) == "rock"


def test_three_classifies_three():
    assert classify_hand(hand_three()) == "three"


def test_fist_with_thumb_poking_out_still_classifies_fist():
    assert classify_hand(hand_fist_thumb_poking_out()) == "fist"
