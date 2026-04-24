"""DesktopController gesture-state tests.

We patch os_control functions to capture calls rather than drive the real OS.
"""

from __future__ import annotations

from unittest.mock import patch

from handspring.desktop_controller import DesktopController
from handspring.types import (
    FaceFeatures,
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
    MotionState,
    PoseState,
)


def _face_facing(mouth_open: float = 0.0) -> FaceState:
    """Default face: present, looking at camera, mouth at given openness."""
    return FaceState(
        present=True,
        features=FaceFeatures(yaw=0.0, pitch=0.0, mouth_open=mouth_open),
        expression="neutral",
        eye_left_open=1.0,
        eye_right_open=1.0,
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


def _frame(
    left: HandState, right: HandState, face: FaceState | None = None
) -> FrameResult:
    # Default to a face-present frame so face-gating (on by default) doesn't
    # suppress hand gestures in tests. Pass an explicit ``face`` to override.
    return FrameResult(
        left=left,
        right=right,
        face=face if face is not None else _face_facing(),
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
        # Pull apart to diagonal corners — hold for many frames so EMA converges.
        for i in range(1, 40):
            c.update(
                _frame(
                    _hand("open", 0.2, 0.3, pinch=0.95),
                    _hand("open", 0.7, 0.75, pinch=0.95),
                ),
                now=i * 0.033,
            )
        # Release both pinches → commit.
        c.update(
            _frame(
                _hand("open", 0.2, 0.3, pinch=0.1),
                _hand("open", 0.7, 0.75, pinch=0.1),
            ),
            now=40 * 0.033,
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
    # Now armed — run many frames so EMA converges to the target positions.
    for i in range(1, 40):
        c.update(
            _frame(
                _hand("open", 0.3, 0.3, pinch=0.95),
                _hand("open", 0.7, 0.7, pinch=0.95),
            ),
            now=i * 0.033,
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



# ---------------------------------------------------------------------------
# Window / Mission / Desktops commits
# ---------------------------------------------------------------------------


def _find_root(c: DesktopController, name: str) -> int:
    for i, (n, _) in enumerate(c.root_items()):
        if n == name:
            return i
    raise AssertionError(f"no root item {name}")


def test_mission_commits_mission_control():
    c = DesktopController(mirrored=False)
    with patch("handspring.desktop_controller.os_control.mission_control") as m:
        c._commit_radial(_find_root(c, "Mission"), None)  # type: ignore[attr-defined]
        m.assert_called_once()


def test_flick_commit_fires_on_release_with_direction():
    """Pinch → tiny move in a direction → release fires that slice's command."""
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.mission_control") as mc,
    ):
        # Find Mission's slice index from the tree.
        items = c.root_items()
        mission_idx = next(i for i, (n, _) in enumerate(items) if n == "Mission")
        import math
        slice_size = 2 * math.pi / len(items)
        bisector = -math.pi / 2 + mission_idx * slice_size
        ux, uy = math.cos(bisector), math.sin(bisector)

        ox, oy = 0.3, 0.5
        # Frame 1: pinch at origin.
        c.update(_frame(_hand("open", ox, oy, pinch=0.95), _absent()), now=0.0)
        # Frame 2: move past flick_threshold (0.03) along Mission's bisector.
        c.update(
            _frame(_hand("open", ox + ux * 0.05, oy + uy * 0.05, pinch=0.95), _absent()),
            now=0.05,
        )
        # Frame 3: release pinch → fire.
        c.update(
            _frame(_hand("open", ox + ux * 0.05, oy + uy * 0.05, pinch=0.1), _absent()),
            now=0.10,
        )
        mc.assert_called_once()


def test_flick_cancels_when_released_at_origin():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.mission_control") as mc,
    ):
        c.update(_frame(_hand("open", 0.3, 0.5, pinch=0.95), _absent()), now=0.0)
        # Tiny jitter, well inside flick_threshold.
        c.update(_frame(_hand("open", 0.305, 0.502, pinch=0.95), _absent()), now=0.05)
        c.update(_frame(_hand("open", 0.305, 0.502, pinch=0.1), _absent()), now=0.10)
        mc.assert_not_called()


def test_flick_no_hold_required_menu_is_instant():
    """radial_state returns a payload on the very first pinching frame —
    no 0.4s dwell."""
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        c.update(_frame(_hand("open", 0.3, 0.5, pinch=0.95), _absent()), now=0.0)
        state = c.radial_state()
        assert state is not None


def test_flick_selects_app_via_create_leaf():
    """Flicking to the Create leaf spawns the configured app."""
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.new_app_window") as naw,
    ):
        items = c.root_items()
        create_idx = next(i for i, (n, _) in enumerate(items) if n == "Create")
        import math
        slice_size = 2 * math.pi / len(items)
        bisector = -math.pi / 2 + create_idx * slice_size
        ux, uy = math.cos(bisector), math.sin(bisector)
        ox, oy = 0.3, 0.5
        c.update(_frame(_hand("open", ox, oy, pinch=0.95), _absent()), now=0.0)
        c.update(
            _frame(_hand("open", ox + ux * 0.06, oy + uy * 0.06, pinch=0.95), _absent()),
            now=0.05,
        )
        c.update(
            _frame(_hand("open", ox + ux * 0.06, oy + uy * 0.06, pinch=0.1), _absent()),
            now=0.10,
        )
        naw.assert_called_once()
        assert naw.call_args[0][0] == "Finder"


def test_create_ghost_rect_ema_smooths_jitter():
    """A jittery fingertip pair produces a smoothed ghost rect that
    lags the raw position (not exactly equal)."""
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Arm: both hands pinching near each other.
        c.update(
            _frame(
                _hand("open", 0.45, 0.5, pinch=0.95),
                _hand("open", 0.50, 0.5, pinch=0.95),
            ),
            now=0.0,
        )
        # Huge jump on the left fingertip one frame.
        c.update(
            _frame(
                _hand("open", 0.10, 0.5, pinch=0.95),
                _hand("open", 0.50, 0.5, pinch=0.95),
            ),
            now=0.05,
        )
        rect = c.pending_create_bounds()
        assert rect is not None
        x_min, _y, w_, _h = rect
        # With smoothing < 1.0, the ghost should *not* have snapped fully
        # to the new raw left position (0.10). It should be between the
        # old (~0.45) and the raw target.
        # Concretely: the reported x_min should be greater than 0.10 + epsilon.
        assert x_min > 0.15


def test_peace_sign_held_fires_show_desktop():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.show_desktop") as sd,
    ):
        # Frame 0: peace on right hand — not yet held.
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=0.0)
        sd.assert_not_called()
        # Frame 1: still peace but total hold only 0.1s — no fire.
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=0.1)
        sd.assert_not_called()
        # Frame 2: held 0.35s ≥ 0.3s threshold — fire.
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=0.35)
        sd.assert_called_once()


def test_peace_fires_once_and_requires_drop_to_rearm():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.show_desktop") as sd,
    ):
        # Hold peace long enough to fire.
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=0.0)
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=0.5)
        assert sd.call_count == 1
        # Keep holding — must NOT fire again.
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=1.0)
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=1.5)
        assert sd.call_count == 1
        # Drop peace (open hand).
        c.update(_frame(_absent(), _hand("open", 0.5, 0.5)), now=2.0)
        # Re-show peace and hold — should fire again.
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=2.1)
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=2.5)
        assert sd.call_count == 2


def test_peace_suppressed_while_disabled():
    c = DesktopController(mirrored=False)
    # Disable via failsafe (both fists 5s).
    t = 0.0
    while t < 6.0:
        c.update(_frame(_hand("fist", 0.3, 0.5), _hand("fist", 0.7, 0.5)), now=t)
        t += 0.1
    assert not c.enabled()
    with patch("handspring.desktop_controller.os_control.show_desktop") as sd:
        # Hold peace — must NOT fire while disabled.
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=t)
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=t + 0.5)
        sd.assert_not_called()


def test_peace_command_override_runs_shell_instead():
    """If cfg.gestures.peace_command is set, run that shell cmd instead of show_desktop()."""
    from handspring.config import Config, ConfigStore, GesturesConfig
    cfg = Config(gestures=GesturesConfig(peace_command="echo hi"))
    store = ConfigStore(persist=False, initial=cfg)
    c = DesktopController(mirrored=False, store=store)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.show_desktop") as sd,
        patch("handspring.desktop_controller.subprocess.Popen") as pop,
    ):
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=0.0)
        c.update(_frame(_absent(), _hand("peace", 0.5, 0.5)), now=0.5)
        sd.assert_not_called()
        pop.assert_called_once()


def test_flick_hysteresis_keeps_selection_stable_near_boundary():
    """Small wobble across a slice boundary stays on the originally-picked slice."""
    import math

    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        items = c.root_items()
        n = len(items)
        slice_size = 2 * math.pi / n
        # Pick slice 2 (Mission) for clarity.
        target = 2
        bisector = -math.pi / 2 + target * slice_size
        # Aim at slice 2 solidly first.
        ox, oy = 0.3, 0.5
        c.update(_frame(_hand("open", ox, oy, pinch=0.95), _absent()), now=0.0)
        c.update(
            _frame(
                _hand("open", ox + math.cos(bisector) * 0.06, oy + math.sin(bisector) * 0.06, pinch=0.95),
                _absent(),
            ),
            now=0.05,
        )
        assert c._radial.hovered_root == target  # type: ignore[attr-defined]
        # Now nudge slightly toward the NEXT slice's bisector, but not
        # past the expanded angular half-width. With hysteresis = 0.15,
        # the current slice's half-range is slice_size/2 * 1.15 ≈ 36°
        # for 6 slices. Move hand by only half_slice_angle + 5° — inside
        # the hysteresis band.
        nudge_angle = bisector + slice_size * 0.55  # slightly past the boundary
        c.update(
            _frame(
                _hand(
                    "open",
                    ox + math.cos(nudge_angle) * 0.06,
                    oy + math.sin(nudge_angle) * 0.06,
                    pinch=0.95,
                ),
                _absent(),
            ),
            now=0.10,
        )
        # Thanks to hysteresis (0.15), it should still be on `target`,
        # not the neighbor.
        assert c._radial.hovered_root == target  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Face gate + mouth-open
# ---------------------------------------------------------------------------


def _face_absent() -> FaceState:
    return FaceState(
        present=False, features=None, expression="neutral",
        eye_left_open=0.0, eye_right_open=0.0,
    )


def _face_looking_away(yaw: float) -> FaceState:
    return FaceState(
        present=True,
        features=FaceFeatures(yaw=yaw, pitch=0.0, mouth_open=0.0),
        expression="neutral", eye_left_open=1.0, eye_right_open=1.0,
    )


def test_face_gate_suppresses_cursor_after_grace_period():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor") as mv,
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Present → cursor moves.
        c.update(_frame(_absent(), _hand("open", 0.5, 0.5)), now=0.0)
        assert mv.call_count == 1
        # Simulate the face disappearing past the grace period.
        for i in range(25):
            c.update(
                _frame(_absent(), _hand("open", 0.5, 0.5), face=_face_absent()),
                now=(i + 1) * 0.033,
            )
        # Gate should now be engaged.
        assert c.face_gated()
        # Further frames with the face absent must not trigger move_cursor.
        count_before = mv.call_count
        for i in range(10):
            c.update(
                _frame(_absent(), _hand("open", 0.5, 0.5), face=_face_absent()),
                now=1.0 + i * 0.033,
            )
        assert mv.call_count == count_before


def test_face_gate_reengages_quickly_when_face_returns():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor") as mv,
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Disappear for the grace period.
        for i in range(25):
            c.update(
                _frame(_absent(), _hand("open", 0.5, 0.5), face=_face_absent()),
                now=i * 0.033,
            )
        assert c.face_gated()
        # Single present-face frame re-enables.
        c.update(_frame(_absent(), _hand("open", 0.5, 0.5)), now=1.0)
        assert not c.face_gated()
        assert mv.called  # cursor resumed


def test_face_gate_looking_away_is_same_as_absent():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # yaw 0.9 is beyond the default 0.6 tolerance.
        for i in range(25):
            c.update(
                _frame(_absent(), _hand("open", 0.5, 0.5), face=_face_looking_away(0.9)),
                now=i * 0.033,
            )
        assert c.face_gated()


def test_face_gate_can_be_disabled_via_config():
    from handspring.config import Config, ConfigStore, FaceConfig
    store = ConfigStore(persist=False)
    store.set(Config(face=FaceConfig(gate_gestures=False)))
    c = DesktopController(mirrored=False, store=store)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor") as mv,
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
    ):
        # Long absence with gate disabled — cursor should keep moving.
        for i in range(30):
            c.update(
                _frame(_absent(), _hand("open", 0.5, 0.5), face=_face_absent()),
                now=i * 0.033,
            )
        assert not c.face_gated()
        assert mv.call_count == 30


def test_mouth_open_hold_fires_spotlight():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.spotlight") as sp,
    ):
        face_open = _face_facing(mouth_open=0.8)
        # Frame 0: mouth opens, start timer.
        c.update(_frame(_absent(), _absent(), face=face_open), now=0.0)
        assert sp.call_count == 0
        # Frame 1: still open but below hold time.
        c.update(_frame(_absent(), _absent(), face=face_open), now=1.0)
        assert sp.call_count == 0
        # Frame 2: past 3s hold → fires.
        c.update(_frame(_absent(), _absent(), face=face_open), now=3.1)
        assert sp.call_count == 1


def test_mouth_open_brief_open_does_not_fire():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.spotlight") as sp,
    ):
        # Talking: mouth opens and closes well within the 3s hold.
        c.update(_frame(_absent(), _absent(), face=_face_facing(mouth_open=0.7)), now=0.0)
        c.update(_frame(_absent(), _absent(), face=_face_facing(mouth_open=0.0)), now=0.5)
        c.update(_frame(_absent(), _absent(), face=_face_facing(mouth_open=0.8)), now=1.0)
        c.update(_frame(_absent(), _absent(), face=_face_facing(mouth_open=0.0)), now=1.5)
        assert sp.call_count == 0


def test_mouth_open_requires_close_before_refire():
    c = DesktopController(mirrored=False)
    with (
        patch("handspring.desktop_controller.os_control.move_cursor"),
        patch("handspring.desktop_controller.os_control.mouse_down"),
        patch("handspring.desktop_controller.os_control.mouse_up"),
        patch("handspring.desktop_controller.os_control.spotlight") as sp,
    ):
        face_open = _face_facing(mouth_open=0.8)
        c.update(_frame(_absent(), _absent(), face=face_open), now=0.0)
        c.update(_frame(_absent(), _absent(), face=face_open), now=3.1)
        assert sp.call_count == 1
        # Keep mouth open indefinitely — must not refire.
        c.update(_frame(_absent(), _absent(), face=face_open), now=10.0)
        assert sp.call_count == 1
        # Close + reopen + hold → fires again.
        c.update(_frame(_absent(), _absent(), face=_face_facing(mouth_open=0.0)), now=11.0)
        c.update(_frame(_absent(), _absent(), face=face_open), now=12.0)
        c.update(_frame(_absent(), _absent(), face=face_open), now=15.2)
        assert sp.call_count == 2
