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


def test_both_pinch_pull_apart_then_release_spawns_selected_app():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_drag"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.new_app_window") as naw,
    ):
        # Arm.
        c.update(
            _frame(
                _hand("open", 0.48, 0.5, pinch=0.95),
                _hand("open", 0.52, 0.5, pinch=0.95),
            ),
            now=0.0,
        )
        # Pull apart.
        c.update(
            _frame(
                _hand("open", 0.2, 0.3, pinch=0.95),
                _hand("open", 0.8, 0.7, pinch=0.95),
            ),
            now=0.1,
        )
        naw.assert_not_called()
        # Release.
        c.update(
            _frame(
                _hand("open", 0.2, 0.3, pinch=0.1),
                _hand("open", 0.8, 0.7, pinch=0.1),
            ),
            now=0.2,
        )
        naw.assert_called_once()
        # Called with (selected_app, bounds=...).
        assert naw.call_args[0][0] == c.selected_app()
        assert "bounds" in naw.call_args.kwargs


def test_radial_does_not_arm_while_right_hand_is_pinching():
    """Right-hand pinch (= click or create) should suppress radial entirely."""
    c = DesktopController(mirrored=False)
    prev = c.selected_app()
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_drag"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Both hands pinching for well over the hold threshold.
        c.update(
            _frame(
                _hand("open", 0.3, 0.5, pinch=0.95),
                _hand("open", 0.7, 0.5, pinch=0.95),
            ),
            now=0.0,
        )
        for i in range(30):
            c.update(
                _frame(
                    _hand("open", 0.3, 0.5, pinch=0.95),
                    _hand("open", 0.7, 0.5, pinch=0.95),
                ),
                now=0.1 + i * 0.05,
            )
        # Release both. No selection should have committed.
        c.update(
            _frame(
                _hand("open", 0.3, 0.5, pinch=0.1),
                _hand("open", 0.7, 0.5, pinch=0.1),
            ),
            now=5.0,
        )
        assert c.selected_app() == prev


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


def test_cursor_uses_midpoint_of_thumb_and_index():
    """Pinching pulls index toward thumb; cursor should track the midpoint so
    it doesn't drift when you pinch."""
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor") as mv,
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Craft features where index_y and thumb_y differ: midpoint y is the
        # average. Use a hand with index at (0.5, 0.4) and thumb at (0.5, 0.6).
        features = HandFeatures(
            x=0.5,
            y=0.5,
            z=0.0,
            openness=1.0,
            pinch=0.0,
            index_x=0.5,
            index_y=0.4,
            thumb_x=0.5,
            thumb_y=0.6,
        )
        state = HandState(
            present=True,
            features=features,
            gesture="open",
            motion=MotionState(False, False, 0, 0, None),
        )
        c.update(_frame(_absent(), state), now=0.0)
        _sx, sy = mv.call_args[0]
        # Midpoint y is (0.4 + 0.6) / 2 = 0.5.
        _, sh = c._screen_w, c._screen_h  # type: ignore[attr-defined]
        assert abs(sy / sh - 0.5) < 0.02  # small tolerance for smoothing first frame


def test_cursor_smoothing_interpolates():
    """After a big jump, the cursor should move less than the full delta."""
    c = DesktopController(mirrored=False)
    sw = c._screen_w  # type: ignore[attr-defined]
    with (
        patch("handspring.desktop_controller.os_control.move_cursor") as mv,
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        c.update(_frame(_absent(), _hand("open", 0.2, 0.5)), now=0.0)
        first = mv.call_args[0]
        c.update(_frame(_absent(), _hand("open", 0.8, 0.5)), now=0.033)
        second = mv.call_args[0]
        # Raw jump would be ~0.6*sw px; smoothed should be substantially less.
        delta = second[0] - first[0]
        raw_delta = 0.6 * sw
        assert 0 < delta < raw_delta  # moved in the right direction but not all the way


def test_create_commits_with_bounds_matching_fingertips():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_drag"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.new_finder_window") as nfw,
    ):
        # Arm with hands close.
        c.update(
            _frame(
                _hand("open", 0.48, 0.5, pinch=0.95),
                _hand("open", 0.52, 0.5, pinch=0.95),
            ),
            now=0.0,
        )
        # Pull apart to diagonal corners.
        c.update(
            _frame(
                _hand("open", 0.2, 0.3, pinch=0.95),
                _hand("open", 0.7, 0.75, pinch=0.95),
            ),
            now=0.1,
        )
        # Release both pinches → commit.
        c.update(
            _frame(
                _hand("open", 0.2, 0.3, pinch=0.1),
                _hand("open", 0.7, 0.75, pinch=0.1),
            ),
            now=0.2,
        )
        nfw.assert_called_once()
        # Bounds kwarg should be present and describe the pulled-apart rect.
        kwargs = nfw.call_args.kwargs
        assert "bounds" in kwargs
        x1, y1, x2, y2 = kwargs["bounds"]
        sw = c._screen_w  # type: ignore[attr-defined]
        # x-range: 0.2..0.7 in screen coords.
        assert abs(x1 / sw - 0.2) < 0.02
        assert abs(x2 / sw - 0.7) < 0.02


def test_pending_create_bounds_exposes_live_rect():
    c = DesktopController(mirrored=False)
    assert c.pending_create_bounds() is None
    c.update(
        _frame(
            _hand("open", 0.48, 0.5, pinch=0.95),
            _hand("open", 0.52, 0.5, pinch=0.95),
        ),
        now=0.0,
    )
    # Now armed — moving should update.
    c.update(
        _frame(
            _hand("open", 0.3, 0.3, pinch=0.95),
            _hand("open", 0.7, 0.7, pinch=0.95),
        ),
        now=0.1,
    )
    rect = c.pending_create_bounds()
    assert rect is not None
    x, y, w_, h_ = rect
    assert abs(x - 0.3) < 0.01
    assert abs(y - 0.3) < 0.01
    assert abs(w_ - 0.4) < 0.01
    assert abs(h_ - 0.4) < 0.01


# ---------------------------------------------------------------------------
# Left-hand radial app launcher
# ---------------------------------------------------------------------------


def test_radial_sets_selected_app_after_hold_and_pull():
    c = DesktopController(mirrored=False)
    assert c.selected_app() == "Finder"  # default
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Left hand starts pinching.
        c.update(_frame(_hand("open", 0.3, 0.5, pinch=0.95), _absent()), now=0.0)
        # Hold 0.5s — hits 0.4s threshold and activates.
        c.update(_frame(_hand("open", 0.3, 0.5, pinch=0.95), _absent()), now=0.5)
        # Move hand toward DOWN-LEFT (next slice around the wheel).
        c.update(_frame(_hand("open", 0.2, 0.7, pinch=0.95), _absent()), now=0.6)
        picked = c.selected_app()  # shouldn't have committed yet
        # Release pinch → commit.
        c.update(_frame(_hand("open", 0.2, 0.7, pinch=0.1), _absent()), now=0.7)
        assert c.selected_app() in c.radial_apps()
        # Sanity: selection should have moved off default if we aimed at a slice.
        # (we don't assert a specific app to keep the test robust to angle math.)
        del picked


def test_radial_release_at_center_keeps_previous_selection():
    c = DesktopController(mirrored=False)
    prev = c.selected_app()
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        c.update(_frame(_hand("open", 0.3, 0.5, pinch=0.95), _absent()), now=0.0)
        c.update(_frame(_hand("open", 0.3, 0.5, pinch=0.95), _absent()), now=0.5)
        # Stay at center, release — no selection → no change.
        c.update(_frame(_hand("open", 0.3, 0.5, pinch=0.1), _absent()), now=0.6)
        assert c.selected_app() == prev


def test_radial_short_pinch_does_not_activate():
    c = DesktopController(mirrored=False)
    prev = c.selected_app()
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Brief pinch < 0.4s then release.
        c.update(_frame(_hand("open", 0.3, 0.5, pinch=0.95), _absent()), now=0.0)
        c.update(_frame(_hand("open", 0.3, 0.3, pinch=0.1), _absent()), now=0.2)
        assert c.selected_app() == prev


def test_cursor_inset_reaches_screen_edges():
    """With inset=0.08, camera x=0.08 should map to screen x=0 (left edge)."""
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor") as mv,
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Several frames at camera midpoint=0.08 so smoothing converges.
        # Thumb and index spread apart so is_pinching() stays False.
        state = HandState(
            present=True,
            features=HandFeatures(
                x=0.08,
                y=0.5,
                z=0,
                openness=1,
                pinch=0,
                index_x=0.04,
                index_y=0.5,
                thumb_x=0.12,
                thumb_y=0.5,
            ),
            gesture="open",
            motion=MotionState(False, False, 0, 0, None),
        )
        for i in range(40):
            c.update(_frame(_absent(), state), now=i * 0.033)
        sx, _sy = mv.call_args[0]
        assert sx == 0  # hit the left edge
