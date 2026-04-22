"""DesktopController gesture-state tests.

We patch os_control functions to capture calls rather than drive the real OS.
"""

from __future__ import annotations

from unittest.mock import patch

from handspring.desktop_controller import DesktopController
from handspring.types import (
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
    MotionState,
    PoseState,
)


def _hf(x: float, y: float, pinch: float = 0.0) -> HandFeatures:
    if pinch >= 0.85:
        tx, ty = x + 0.005, y + 0.005
    else:
        tx, ty = x + 0.1, y
    return HandFeatures(
        x=x,
        y=y,
        z=0.0,
        openness=1.0,
        pinch=pinch,
        index_x=x,
        index_y=y,
        thumb_x=tx,
        thumb_y=ty,
    )


def _hand(gesture: str, x: float, y: float, pinch: float = 0.0) -> HandState:
    return HandState(
        present=True,
        features=_hf(x, y, pinch),
        gesture=gesture,  # type: ignore[arg-type]
        motion=MotionState(
            pinching=pinch >= 0.85, dragging=False, drag_dx=0, drag_dy=0, event=None
        ),
    )


def _absent() -> HandState:
    return HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0, 0, None),
    )


def _frame(left: HandState, right: HandState) -> FrameResult:
    return FrameResult(
        left=left,
        right=right,
        face=FaceState(False, None, "neutral", 0, 0),
        pose=PoseState(False, None),
        fps=30.0,
        clap_event=False,
    )


def test_cursor_moves_to_right_index_tip():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor") as mv,
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        c.update(_frame(_absent(), _hand("open", 0.5, 0.5)), now=0.0)
        mv.assert_called_once()


def test_pinch_fires_mouse_down_then_up():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down") as md,
        patch("handspring.desktop_controller.os_control.mouse_drag"),
        patch("handspring.desktop_controller.os_control.mouse_up") as mu,
    ):
        # Begin pinch.
        c.update(_frame(_absent(), _hand("open", 0.5, 0.5, pinch=0.95)), now=0.0)
        # Release pinch.
        c.update(_frame(_absent(), _hand("open", 0.5, 0.5, pinch=0.1)), now=0.01)
        md.assert_called_once()
        mu.assert_called_once()


def test_both_fist_five_seconds_disables():
    c = DesktopController(mirrored=False)
    assert c.enabled()
    t = 0.0
    while t < 6.0:
        c.update(_frame(_hand("fist", 0.3, 0.5), _hand("fist", 0.7, 0.5)), now=t)
        t += 0.1
    assert not c.enabled()
    # And another 6 seconds re-enables.
    while t < 12.2:
        c.update(_frame(_hand("fist", 0.3, 0.5), _hand("fist", 0.7, 0.5)), now=t)
        t += 0.1
    assert c.enabled()


def test_failsafe_aborts_if_fist_released_before_5s():
    c = DesktopController(mirrored=False)
    # Hold 3s then release.
    for t in range(30):
        c.update(_frame(_hand("fist", 0.3, 0.5), _hand("fist", 0.7, 0.5)), now=t * 0.1)
    c.update(_frame(_hand("open", 0.3, 0.5), _hand("open", 0.7, 0.5)), now=3.1)
    # Resume fisting — should restart countdown, not immediately toggle.
    for t in range(20):
        c.update(_frame(_hand("fist", 0.3, 0.5), _hand("fist", 0.7, 0.5)), now=3.2 + t * 0.1)
    assert c.enabled()  # only ~2s of the second hold, not enough to toggle


def test_disabled_skips_cursor():
    c = DesktopController(mirrored=False)
    # Force disable via failsafe.
    t = 0.0
    while t < 6.0:
        c.update(_frame(_hand("fist", 0.3, 0.5), _hand("fist", 0.7, 0.5)), now=t)
        t += 0.1
    with patch("handspring.desktop_controller.os_control.move_cursor") as mv:
        c.update(_frame(_absent(), _hand("open", 0.5, 0.5)), now=t + 1.0)
        mv.assert_not_called()


def test_both_pinch_pull_apart_fires_new_finder_window():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_drag"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.new_finder_window") as nfw,
    ):
        # Arm: both pinching, hands close.
        c.update(
            _frame(
                _hand("open", 0.48, 0.5, pinch=0.95),
                _hand("open", 0.52, 0.5, pinch=0.95),
            ),
            now=0.0,
        )
        # Pull apart while still pinching — should fire.
        c.update(
            _frame(
                _hand("open", 0.2, 0.5, pinch=0.95),
                _hand("open", 0.8, 0.5, pinch=0.95),
            ),
            now=0.1,
        )
        nfw.assert_called_once()


def test_mirrored_flips_x():
    c = DesktopController(mirrored=True)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor") as mv,
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Right hand at raw x=0.2 → in mirrored display that's the right side
        # of the screen, so cursor should land at screen_w * 0.8, not 0.2.
        c.update(_frame(_absent(), _hand("open", 0.2, 0.5)), now=0.0)
        sx, sy = mv.call_args[0]
        sw, _ = c._screen_w, c._screen_h  # type: ignore[attr-defined]
        assert sx > sw * 0.5  # landed on the right half
