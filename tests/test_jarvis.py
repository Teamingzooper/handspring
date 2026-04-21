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
    return HandFeatures(x=x, y=y, z=0.0, openness=1.0, pinch=pinch, index_x=x, index_y=y)


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


def test_both_pinch_and_pull_creates_window():
    c = JarvisController()
    # Start: both hands close together, pinching.
    c.update(_frame(_hand("fist", 0.4, 0.5, pinch=0.95), _hand("fist", 0.5, 0.5, pinch=0.95)))
    assert c.manager.windows() == []  # not created yet (still pinching)
    # Pull apart.
    c.update(_frame(_hand("fist", 0.25, 0.35, pinch=0.95), _hand("fist", 0.65, 0.65, pinch=0.95)))
    # Release one hand's pinch → commit.
    c.update(_frame(_hand("fist", 0.25, 0.35, pinch=0.2), _hand("fist", 0.65, 0.65, pinch=0.95)))
    assert len(c.manager.windows()) == 1


def test_tiny_pull_discarded():
    c = JarvisController()
    c.update(_frame(_hand("fist", 0.45, 0.50, pinch=0.95), _hand("fist", 0.47, 0.50, pinch=0.95)))
    # Release at almost-identical positions — diagonal < 0.1.
    c.update(_frame(_hand("fist", 0.45, 0.50, pinch=0.2), _hand("fist", 0.47, 0.50, pinch=0.95)))
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
