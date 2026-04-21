"""Window data + WindowManager basic operations."""

from __future__ import annotations

from handspring.jarvis import Window, WindowManager


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
