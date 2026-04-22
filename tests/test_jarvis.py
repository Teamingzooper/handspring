"""Window data + WindowManager basic operations."""

from __future__ import annotations

from handspring.jarvis import JarvisController, Window, WindowManager
from handspring.types import (
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
    MotionState,
    PoseState,
)


def test_window_contains_palm():
    w = Window(id=1, x=0.2, y=0.3, width=0.4, height=0.3, z=0, color_idx=0)
    assert w.contains(0.4, 0.45)  # inside
    assert not w.contains(0.1, 0.1)  # outside


def test_manager_create_assigns_ids():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.2)
    b = m.create(x=0.5, y=0.5, width=0.3, height=0.2)
    assert a.id != b.id


def test_manager_cap_evicts_oldest():
    m = WindowManager(max_windows=3)
    created = [m.create(x=0.1 * i, y=0.1, width=0.1, height=0.1) for i in range(5)]
    windows = m.windows()
    assert len(windows) == 3
    # The three most-recently-created survive.
    remaining_ids = {w.id for w in windows}
    assert created[0].id not in remaining_ids
    assert created[1].id not in remaining_ids
    assert created[-1].id in remaining_ids


def test_manager_z_order_create_goes_top():
    m = WindowManager()
    _ = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    b = m.create(x=0.2, y=0.2, width=0.3, height=0.3)
    # b is on top.
    assert m.topmost_at(0.25, 0.25) is not None
    assert m.topmost_at(0.25, 0.25).id == b.id


def test_promote_to_top():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    _ = m.create(x=0.2, y=0.2, width=0.3, height=0.3)
    m.promote(a.id)
    # Now a should be topmost under the overlap.
    top = m.topmost_at(0.25, 0.25)
    assert top is not None and top.id == a.id


def test_move_window():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    m.move(a.id, dx=0.2, dy=-0.05)
    moved = m.get(a.id)
    assert moved is not None
    assert abs(moved.x - 0.3) < 1e-6
    assert abs(moved.y - 0.05) < 1e-6


def test_move_clamps_to_frame():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    m.move(a.id, dx=10.0, dy=10.0)
    moved = m.get(a.id)
    assert moved is not None
    assert moved.x + moved.width <= 1.0 + 1e-6
    assert moved.y + moved.height <= 1.0 + 1e-6


def test_cycle_color():
    m = WindowManager()
    a = m.create(x=0.1, y=0.1, width=0.3, height=0.3)
    initial = a.color_idx
    m.cycle_color(a.id)
    assert m.get(a.id).color_idx == (initial + 1) % 3  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# JarvisController — gesture state-machine tests
# ---------------------------------------------------------------------------


def _hf(x: float, y: float, pinch: float = 0.0) -> HandFeatures:
    # When pinch >= 0.85 we place thumb very close to index (distance ~0.01 < 0.05).
    # When pinch < 0.5 we place thumb 0.1 away so is_pinching() returns False.
    if pinch >= 0.85:
        thumb_x, thumb_y = x + 0.005, y + 0.005
    else:
        thumb_x, thumb_y = x + 0.1, y
    return HandFeatures(
        x=x,
        y=y,
        z=0.0,
        openness=1.0,
        pinch=pinch,
        index_x=x,
        index_y=y,
        thumb_x=thumb_x,
        thumb_y=thumb_y,
    )


def _hand(gesture: str, x: float, y: float, pinch: float = 0.0) -> HandState:
    return HandState(
        present=True,
        features=_hf(x, y, pinch),
        gesture=gesture,  # type: ignore[arg-type]
        motion=MotionState(
            pinching=pinch >= 0.85, dragging=False, drag_dx=0.0, drag_dy=0.0, event=None
        ),
    )


def _absent() -> HandState:
    return HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )


def _face() -> FaceState:
    return FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )


def _frame(left: HandState, right: HandState) -> FrameResult:
    return FrameResult(
        left=left,
        right=right,
        face=_face(),
        pose=PoseState(False, None),
        fps=30.0,
        clap_event=False,
    )


def test_pinch_fingertips_together_then_pull_creates_window():
    c = JarvisController()
    # Hands start pinching with index tips almost touching (distance < 0.08).
    # x positions 0.48 and 0.52 → distance 0.04 < 0.08.
    c.update(
        _frame(
            _hand("open", 0.48, 0.5, pinch=0.95),
            _hand("open", 0.52, 0.5, pinch=0.95),
        )
    )
    assert c.manager.windows() == []  # still in creation gesture
    # Pull apart — hands move to 0.25 and 0.75.
    c.update(
        _frame(
            _hand("open", 0.25, 0.35, pinch=0.95),
            _hand("open", 0.75, 0.65, pinch=0.95),
        )
    )
    # Release left pinch → commit.
    c.update(
        _frame(
            _hand("open", 0.25, 0.35, pinch=0.2),
            _hand("open", 0.75, 0.65, pinch=0.95),
        )
    )
    assert len(c.manager.windows()) == 1
    w = c.manager.windows()[0]
    # Window should span from 0.25..0.75 in x, 0.35..0.65 in y.
    assert abs(w.x - 0.25) < 1e-6
    assert abs(w.y - 0.35) < 1e-6
    # Aspect ratio 0.5/0.3 ≈ 1.67 < 2.0, no clamp.
    assert abs(w.width - 0.5) < 1e-6
    assert abs(w.height - 0.3) < 1e-6


def test_no_create_if_fingertips_far_apart_at_pinch_start():
    c = JarvisController()
    # Both pinching but hands far apart from the start.
    c.update(
        _frame(
            _hand("open", 0.2, 0.5, pinch=0.95),
            _hand("open", 0.8, 0.5, pinch=0.95),
        )
    )
    # Release — should NOT commit (was never in create state).
    c.update(
        _frame(
            _hand("open", 0.2, 0.5, pinch=0.2),
            _hand("open", 0.8, 0.5, pinch=0.2),
        )
    )
    assert c.manager.windows() == []


def test_tiny_pull_still_discarded():
    # Keep this test's spirit — tiny final diagonal → no window.
    c = JarvisController()
    c.update(
        _frame(
            _hand("open", 0.495, 0.5, pinch=0.95),
            _hand("open", 0.505, 0.5, pinch=0.95),
        )
    )
    # Release without pulling apart.
    c.update(
        _frame(
            _hand("open", 0.495, 0.5, pinch=0.2),
            _hand("open", 0.505, 0.5, pinch=0.95),
        )
    )
    assert c.manager.windows() == []


def test_grab_drag_release():
    c = JarvisController()
    # Seed a window at center.
    w = c.manager.create(x=0.3, y=0.3, width=0.3, height=0.3)
    # Open hand over the window, right side.
    c.update(_frame(_absent(), _hand("open", 0.45, 0.45)))
    # Close to fist over the window — grab.
    c.update(_frame(_absent(), _hand("fist", 0.45, 0.45)))
    # Move hand right.
    c.update(_frame(_absent(), _hand("fist", 0.60, 0.45)))
    # Open hand — release.
    c.update(_frame(_absent(), _hand("open", 0.60, 0.45)))
    moved = c.manager.get(w.id)
    assert moved is not None
    assert moved.x > 0.3  # dragged rightward


def test_point_tap_after_hover():
    c = JarvisController()
    w = c.manager.create(x=0.3, y=0.3, width=0.3, height=0.3)
    initial_color = w.color_idx
    # Point with index tip inside window for 6 frames (>= _TAP_HOVER_FRAMES=5).
    for _i in range(6):
        c.update(_frame(_absent(), _hand("point", 0.45, 0.45)))
    post = c.manager.get(w.id)
    assert post is not None
    assert post.color_idx != initial_color
    # Tap event reported for this frame cycle
    assert c.last_tap_id() == w.id


def test_point_no_tap_if_moves_away_early():
    c = JarvisController()
    w = c.manager.create(x=0.3, y=0.3, width=0.3, height=0.3)
    # 2 frames in, then leave — not long enough.
    c.update(_frame(_absent(), _hand("point", 0.45, 0.45)))
    c.update(_frame(_absent(), _hand("point", 0.45, 0.45)))
    c.update(_frame(_absent(), _hand("point", 0.1, 0.1)))
    unchanged = c.manager.get(w.id)
    assert unchanged is not None
    assert unchanged.color_idx == w.color_idx


def test_pending_rect_none_when_not_creating():
    c = JarvisController()
    c.update(_frame(_absent(), _absent()))
    assert c.pending_rect() is None


def test_pending_rect_tracks_while_creating():
    c = JarvisController()
    c.update(
        _frame(
            _hand("open", 0.48, 0.5, pinch=0.95),
            _hand("open", 0.52, 0.5, pinch=0.95),
        )
    )
    rect1 = c.pending_rect()
    assert rect1 is not None
    x, y, w, h = rect1
    # Initial bounds: x=0.48..0.52, y=0.5..0.5 → narrow
    assert abs(x - 0.48) < 1e-6
    assert abs(w - 0.04) < 1e-6
    # Move apart.
    c.update(
        _frame(
            _hand("open", 0.20, 0.30, pinch=0.95),
            _hand("open", 0.80, 0.70, pinch=0.95),
        )
    )
    rect2 = c.pending_rect()
    assert rect2 is not None
    x2, y2, w2, h2 = rect2
    assert abs(x2 - 0.20) < 1e-6
    assert abs(y2 - 0.30) < 1e-6
    assert abs(w2 - 0.60) < 1e-6
    assert abs(h2 - 0.40) < 1e-6


def test_pending_rect_clears_after_release():
    c = JarvisController()
    c.update(
        _frame(
            _hand("open", 0.48, 0.5, pinch=0.95),
            _hand("open", 0.52, 0.5, pinch=0.95),
        )
    )
    c.update(
        _frame(
            _hand("open", 0.20, 0.30, pinch=0.95),
            _hand("open", 0.80, 0.70, pinch=0.95),
        )
    )
    # Release left pinch — commit.
    c.update(
        _frame(
            _hand("open", 0.20, 0.30, pinch=0.2),
            _hand("open", 0.80, 0.70, pinch=0.95),
        )
    )
    assert c.pending_rect() is None
    # And a window was committed.
    assert len(c.manager.windows()) == 1


def test_resize_enters_when_pinch_near_top_right_corner():
    c = JarvisController()
    w = c.manager.create(x=0.2, y=0.2, width=0.4, height=0.3)
    # top-right corner at (0.6, 0.2)
    # Pinch with right hand at (0.6, 0.2), thumb close to index.
    frame = _frame(
        _absent(),
        HandState(
            present=True,
            features=HandFeatures(
                x=0.6,
                y=0.2,
                z=0.0,
                openness=0.9,
                pinch=0.0,
                index_x=0.6,
                index_y=0.2,
                thumb_x=0.605,
                thumb_y=0.205,
            ),
            gesture="open",
            motion=MotionState(False, False, 0.0, 0.0, None),
        ),
    )
    c.update(frame)
    assert c.resizing_window_id() == w.id


def test_resize_updates_window_size():
    c = JarvisController()
    w = c.manager.create(x=0.2, y=0.2, width=0.4, height=0.3)
    # Enter resize at (0.6, 0.2)
    enter = _frame(
        _absent(),
        HandState(
            present=True,
            features=HandFeatures(
                x=0.6,
                y=0.2,
                z=0.0,
                openness=0.9,
                pinch=0.0,
                index_x=0.6,
                index_y=0.2,
                thumb_x=0.605,
                thumb_y=0.205,
            ),
            gesture="open",
            motion=MotionState(False, False, 0.0, 0.0, None),
        ),
    )
    c.update(enter)
    # Move pinch to (0.8, 0.1) — larger window.
    move = _frame(
        _absent(),
        HandState(
            present=True,
            features=HandFeatures(
                x=0.8,
                y=0.1,
                z=0.0,
                openness=0.9,
                pinch=0.0,
                index_x=0.8,
                index_y=0.1,
                thumb_x=0.805,
                thumb_y=0.105,
            ),
            gesture="open",
            motion=MotionState(False, False, 0.0, 0.0, None),
        ),
    )
    c.update(move)
    resized = c.manager.get(w.id)
    assert resized is not None
    # Bottom-left fixed at (0.2, 0.5): width = 0.8 - 0.2 = 0.6, height = 0.5 - 0.1 = 0.4
    assert abs(resized.x - 0.2) < 1e-6
    assert abs(resized.y - 0.1) < 1e-6
    assert abs(resized.width - 0.6) < 1e-6
    assert abs(resized.height - 0.4) < 1e-6


def test_resize_ends_on_release():
    c = JarvisController()
    w = c.manager.create(x=0.2, y=0.2, width=0.4, height=0.3)
    c.update(
        _frame(
            _absent(),
            HandState(
                present=True,
                features=HandFeatures(
                    x=0.6,
                    y=0.2,
                    z=0.0,
                    openness=0.9,
                    pinch=0.0,
                    index_x=0.6,
                    index_y=0.2,
                    thumb_x=0.605,
                    thumb_y=0.205,
                ),
                gesture="open",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
        )
    )
    assert c.resizing_window_id() == w.id
    # Release pinch (thumb far from index).
    c.update(
        _frame(
            _absent(),
            HandState(
                present=True,
                features=HandFeatures(
                    x=0.6,
                    y=0.2,
                    z=0.0,
                    openness=0.9,
                    pinch=0.0,
                    index_x=0.6,
                    index_y=0.2,
                    thumb_x=0.8,
                    thumb_y=0.2,
                ),
                gesture="open",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
        )
    )
    assert c.resizing_window_id() is None


def test_resize_not_triggered_far_from_corner():
    c = JarvisController()
    c.manager.create(x=0.2, y=0.2, width=0.4, height=0.3)
    # Pinch at (0.1, 0.9) — far from any corner.
    c.update(
        _frame(
            _absent(),
            HandState(
                present=True,
                features=HandFeatures(
                    x=0.1,
                    y=0.9,
                    z=0.0,
                    openness=0.9,
                    pinch=0.0,
                    index_x=0.1,
                    index_y=0.9,
                    thumb_x=0.105,
                    thumb_y=0.905,
                ),
                gesture="open",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
        )
    )
    assert c.resizing_window_id() is None


def test_resize_enforces_minimum_size():
    c = JarvisController()
    w = c.manager.create(x=0.2, y=0.2, width=0.4, height=0.3)
    c.update(
        _frame(
            _absent(),
            HandState(
                present=True,
                features=HandFeatures(
                    x=0.6,
                    y=0.2,
                    z=0.0,
                    openness=0.9,
                    pinch=0.0,
                    index_x=0.6,
                    index_y=0.2,
                    thumb_x=0.605,
                    thumb_y=0.205,
                ),
                gesture="open",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
        )
    )
    # Try to shrink below minimum: pinch at (0.21, 0.49), which would produce
    # width=0.01 and height=0.01 — below the 0.05 floor.
    c.update(
        _frame(
            _absent(),
            HandState(
                present=True,
                features=HandFeatures(
                    x=0.21,
                    y=0.49,
                    z=0.0,
                    openness=0.9,
                    pinch=0.0,
                    index_x=0.21,
                    index_y=0.49,
                    thumb_x=0.215,
                    thumb_y=0.495,
                ),
                gesture="open",
                motion=MotionState(False, False, 0.0, 0.0, None),
            ),
        )
    )
    resized = c.manager.get(w.id)
    assert resized is not None
    assert resized.width >= 0.05
    assert resized.height >= 0.05
