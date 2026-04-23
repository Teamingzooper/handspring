# Flick-Commit Radial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current hold-to-activate + nested-sub-menu radial with a flat 6-leaf flick-commit picker; smooth the two-hand create-window ghost rectangle with an EMA.

**Architecture:** Rewrite `_handle_radial` in `desktop_controller.py` around a pinch-start / release model — no hold, no subs, direction-at-release commits. Add angular hysteresis to kill boundary flicker. Delete sub-layout helpers and the overlay's chip/mini-ring code. Add a `smooth_left` / `smooth_right` pair to `_CreateState` so the ghost rect tweens instead of jumping. Config gains `flick_threshold`, `angular_hysteresis`, `create.smoothing`; loses the four hold/sub fields.

**Tech Stack:** Python 3.10+, dataclasses, pytest, AppKit/Quartz (overlay draw), no new runtime deps.

---

### Task 1: Update `RadialConfig` + `CreateConfig` fields

**Files:**
- Modify: `src/handspring/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config-defaults test**

Append to `tests/test_config.py`:

```python
def test_radial_config_defaults_match_flick_model() -> None:
    cfg = Config()
    assert cfg.radial.flick_threshold == 0.03
    assert cfg.radial.angular_hysteresis == 0.15
    # Old fields must be gone:
    assert not hasattr(cfg.radial, "hold_seconds")
    assert not hasattr(cfg.radial, "sub_threshold")
    assert not hasattr(cfg.radial, "inner_radius")
    assert not hasattr(cfg.radial, "sub_mini_inner")


def test_create_config_has_smoothing() -> None:
    cfg = Config()
    assert cfg.create.smoothing == 0.35
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_config.py::test_radial_config_defaults_match_flick_model tests/test_config.py::test_create_config_has_smoothing -q`
Expected: FAIL — old fields still present, new ones missing.

- [ ] **Step 3: Replace `RadialConfig`**

In `src/handspring/config.py`, change the whole class body:

```python
@dataclass(frozen=True)
class RadialConfig:
    # Minimum camera-space displacement from pinch origin for a release to
    # count as a commit. Below = cancel (deliberate "no-op" pinch).
    flick_threshold: float = 0.03
    # Fractional expansion of the current slice's angular range. Higher =
    # more sticky, less flicker near slice boundaries.
    angular_hysteresis: float = 0.15
```

- [ ] **Step 4: Extend `CreateConfig`**

Change `CreateConfig` in the same file:

```python
@dataclass(frozen=True)
class CreateConfig:
    entry_distance: float = 0.08
    min_diagonal: float = 0.15
    # EMA factor for the ghost rect corners (same idea as cursor smoothing).
    smoothing: float = 0.35
```

- [ ] **Step 5: Run tests, verify the two new tests pass**

Run: `PYTHONPATH=src pytest tests/test_config.py -q`
Expected: the two new tests PASS. Other tests in the file may FAIL (they reference removed fields) — that's OK; Task 4 fixes them.

- [ ] **Step 6: Commit**

```bash
git add src/handspring/config.py tests/test_config.py
git commit -m "feat(config): flick-commit radial fields + create smoothing"
```

---

### Task 2: Replace `_handle_radial` with flick-commit logic

**Files:**
- Modify: `src/handspring/desktop_controller.py`
- Test: `tests/test_desktop_controller.py`

- [ ] **Step 1: Write the failing flick-commit test**

Append to `tests/test_desktop_controller.py`:

```python
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
```

- [ ] **Step 2: Run to confirm they fail**

Run: `PYTHONPATH=src pytest tests/test_desktop_controller.py::test_flick_commit_fires_on_release_with_direction tests/test_desktop_controller.py::test_flick_cancels_when_released_at_origin tests/test_desktop_controller.py::test_flick_no_hold_required_menu_is_instant -q`
Expected: all FAIL (current code requires hold before `radial_state` fires and uses sub_threshold, not flick_threshold).

- [ ] **Step 3: Rewrite `_handle_radial`**

Replace the entire `_handle_radial` method in `src/handspring/desktop_controller.py` with:

```python
def _handle_radial(self, frame: FrameResult, now: float) -> None:
    r = self._radial
    left = frame.left
    right = frame.right
    cfg = self._cfg()
    tree = cfg.radial_tree

    if left.present and left.features is not None:
        lf = left.features
        lmx = (lf.index_x + lf.thumb_x) * 0.5
        lmy = (lf.index_y + lf.thumb_y) * 0.5
        if self._mirrored:
            lmx = 1.0 - lmx
        inset = cfg.cursor.inset
        span = max(1e-6, 1.0 - 2 * inset)
        sx = int(((lmx - inset) / span) * self._screen_w)
        sy = int(((lmy - inset) / span) * self._screen_h)
        sx = max(0, min(self._screen_w - 1, sx))
        sy = max(0, min(self._screen_h - 1, sy))
        self._left_cursor_screen = (sx, sy)
    else:
        self._left_cursor_screen = None

    right_pinching = is_pinching(right)
    pinching = (
        is_pinching(left)
        and left.features is not None
        and not right_pinching
        and not self._create.armed
    )

    if not pinching:
        # Release — if something was highlighted and we flicked far enough,
        # commit it. Otherwise cancel cleanly.
        if r.pinching and r.hovered_root is not None:
            dx = r.cur[0] - r.origin[0]
            dy = r.cur[1] - r.origin[1]
            if (dx * dx + dy * dy) ** 0.5 >= cfg.radial.flick_threshold:
                self._commit_radial(r.hovered_root, None)
        r.pinching = False
        r.active = False
        r.hovered_root = None
        r.hovered_sub = None
        return

    assert left.features is not None
    f = left.features
    cx = (f.index_x + f.thumb_x) * 0.5
    cy = (f.index_y + f.thumb_y) * 0.5

    if not r.pinching:
        # Instant: menu is active from frame zero.
        r.pinching = True
        r.pinch_start = now
        r.active = True
        r.origin = (cx, cy)
        r.cur = (cx, cy)
        r.hovered_root = None
        r.hovered_sub = None
        return

    r.cur = (cx, cy)
    r.active = True

    n_roots = len(tree)
    if n_roots == 0:
        return

    dx = cx - r.origin[0]
    dy = cy - r.origin[1]
    dist = (dx * dx + dy * dy) ** 0.5

    # Dead zone: hand still at origin, nothing highlighted.
    if dist < cfg.radial.flick_threshold * 0.5:
        r.hovered_root = None
        return

    # Angular hysteresis: once a slice is highlighted, neighbors have to
    # exceed the slice's expanded angular range to take over.
    import math
    new_idx = self._slice_index(dx, dy, n_roots)
    if r.hovered_root is None:
        r.hovered_root = new_idx
        return
    slice_size = 2 * math.pi / n_roots
    cur_center_cw = r.hovered_root * slice_size  # clockwise from up
    # angle clockwise-from-up for the current hand.
    angle = math.atan2(dy, dx)
    if self._mirrored:
        angle = math.atan2(dy, -dx)
    cw = (angle + math.pi / 2) % (2 * math.pi)
    delta = ((cw - cur_center_cw + math.pi) % (2 * math.pi)) - math.pi
    half = slice_size / 2 * (1 + cfg.radial.angular_hysteresis)
    if abs(delta) > half:
        r.hovered_root = new_idx
```

- [ ] **Step 4: Update `radial_state` to drop the countdown gate**

Replace the `radial_state` method with:

```python
def radial_state(
    self,
) -> tuple[tuple[float, float], tuple[float, float], int | None, int | None, float] | None:
    """Return (origin, cur, hovered_root, hovered_sub, progress) for the overlay.

    progress is always 1.0 in the flick model — kept in the tuple for
    payload compatibility. hovered_sub is always None.
    """
    r = self._radial
    if not r.pinching:
        return None
    return r.origin, r.cur, r.hovered_root, None, 1.0
```

- [ ] **Step 5: Run the three new tests**

Run: `PYTHONPATH=src pytest tests/test_desktop_controller.py::test_flick_commit_fires_on_release_with_direction tests/test_desktop_controller.py::test_flick_cancels_when_released_at_origin tests/test_desktop_controller.py::test_flick_no_hold_required_menu_is_instant -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/handspring/desktop_controller.py tests/test_desktop_controller.py
git commit -m "feat(radial): flick-commit model — no hold, no subs, hysteresis"
```

---

### Task 3: Simplify `_commit_radial` (flat leaves only)

**Files:**
- Modify: `src/handspring/desktop_controller.py`

- [ ] **Step 1: Replace `_commit_radial`**

Replace the method body with:

```python
def _commit_radial(self, root_idx: int | None, sub_idx: int | None) -> None:
    del sub_idx  # no longer used — all commands are flat leaves
    if root_idx is None:
        return
    tree = self._cfg().radial_tree
    if root_idx >= len(tree):
        return
    item = tree[root_idx]
    name = item.name

    if name == "None":
        self._mode = "none"
        self._events_out.append("mode:none")
        return
    if name == "Scroll":
        self._mode = "scroll"
        self._events_out.append("mode:scroll")
        return
    if name == "Mission":
        os_control.mission_control()
        self._events_out.append("mission")
        return
    if name == "Screenshot":
        os_control.screenshot("screen")
        self._events_out.append("screenshot:screen")
        return
    if name == "Settings":
        self._events_out.append("settings:open")
        if self._on_open_settings is not None:
            try:
                self._on_open_settings()
            except Exception as e:  # noqa: BLE001
                print(f"handspring: settings open failed ({e})", file=sys.stderr)
        return
    if name == "Create":
        # Flat "Create" leaf = spawn a default-sized Finder (or configured app)
        # at a reasonable position without needing the two-hand pull.
        app = item.subs[0] if item.subs else "Finder"
        self._selected_app = app
        self._mode = "create"
        cx = self._screen_w // 2
        cy = self._screen_h // 2
        bounds = (cx - 350, cy - 250, cx + 350, cy + 250)
        os_control.new_app_window(app, bounds=bounds)
        self._events_out.append(f"new_window:{app}")
        self._post_spawn = (bounds, self._last_now + self._post_spawn_hold_seconds)
        return
    # User-defined leaf with a custom shell command.
    if item.command:
        self._run_command(item.command)
        self._events_out.append(f"run:{name}")
```

- [ ] **Step 2: Delete `_handle_more`**

Remove the entire `_handle_more` method (no longer referenced).

- [ ] **Step 3: Run full suite**

Run: `PYTHONPATH=src pytest tests/ -q`
Expected: may have some failures in old tests that reference sub-menus or "More" — those get fixed in Task 4.

- [ ] **Step 4: Commit**

```bash
git add src/handspring/desktop_controller.py
git commit -m "refactor(radial): flat leaves in _commit_radial"
```

---

### Task 4: Delete `compute_sub_layout` + chip constants; update default `radial_tree`

**Files:**
- Modify: `src/handspring/desktop_controller.py`
- Modify: `src/handspring/config.py`
- Test: `tests/test_desktop_controller.py`, `tests/test_config.py`

- [ ] **Step 1: Remove chip constants and `compute_sub_layout` from desktop_controller**

In `src/handspring/desktop_controller.py`, delete these top-level declarations:

- The block `SUB_CHIP_W = 140` … `ROOT_RING_PX = 220` (all 6 constants).
- The entire `compute_sub_layout` function.

- [ ] **Step 2: Replace default `radial_tree` with the 6 flat commands**

In `src/handspring/config.py`, replace `_default_radial_tree`:

```python
def _default_radial_tree() -> tuple[RadialItem, ...]:
    return (
        RadialItem("None"),
        RadialItem("Settings"),
        RadialItem("Mission"),
        RadialItem("Create", ("Finder",)),  # subs[0] = which app to spawn
        RadialItem("Scroll"),
        RadialItem("Screenshot"),
    )
```

Also remove the now-unused constants:

```python
_DEFAULT_RADIAL_APPS = (...)           # delete
_DEFAULT_SCREENSHOT_SUBS = (...)       # delete
_DEFAULT_WINDOW_SUBS = (...)           # delete
_DEFAULT_DESKTOP_SUBS = (...)          # delete
_DEFAULT_MORE_SUBS = (...)             # delete
```

- [ ] **Step 3: Delete the old sub-layout test**

In `tests/test_desktop_controller.py`, delete these tests (no longer applicable):

- `test_radial_sub_row_selected_by_nearest_chip`
- `test_compute_sub_layout_clamps_offscreen`
- `test_radial_root_locks_when_hand_enters_sub_ring`

Also delete these window/mission tests that assumed a sub-selection path —
fix them to use the flat model by removing their sub-index step so the
one-frame pinch+flick+release commits the leaf:

Search for tests calling `_commit_radial(` directly with `sub_idx=<int>` and
replace with `sub_idx=None` so they exercise the new flat path. If the test
name references a sub behavior that no longer exists (e.g., `Tile Left`,
`Desktop Right`, `Screenshot Selection`), delete the test.

- [ ] **Step 4: Update the existing test that used hold semantics**

Find `test_radial_sets_selected_app_after_hold_and_pull` and
`test_radial_short_pinch_does_not_activate` in
`tests/test_desktop_controller.py`. Replace them with:

```python
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
```

Delete `test_radial_release_at_center_keeps_previous_selection` — it's
subsumed by the new "cancels when released at origin" test from Task 2.

Delete `test_radial_short_pinch_does_not_activate` — no hold = nothing to
"not activate."

- [ ] **Step 5: Update the `test_default_config_has_more_slot` test**

In `tests/test_config.py`, replace the `test_default_config_has_more_slot`
test (which asserts "More" exists) with:

```python
def test_default_config_has_expected_six_commands() -> None:
    cfg = Config()
    names = [it.name for it in cfg.radial_tree]
    assert names == ["None", "Settings", "Mission", "Create", "Scroll", "Screenshot"]
```

- [ ] **Step 6: Run full suite**

Run: `PYTHONPATH=src pytest tests/ -q`
Expected: PASS (or a short list of other window/desktop sub tests we delete — see Step 7).

- [ ] **Step 7: Delete dead sub-behavior tests**

In `tests/test_desktop_controller.py`, delete any test whose name contains
`window_left`, `window_right`, `window_close`, `window_minimize`,
`window_fullscreen`, `desktop_left`, `desktop_right`,
`screenshot_window`, `screenshot_selection`, `more_settings`,
`more_reload`, `more_quit`. These commands were sub-menu leaves that no
longer exist as default radial entries. Users can reintroduce them via
custom `command = "..."` `radial_tree` entries in `config.toml`.

- [ ] **Step 8: Run full suite, expect clean pass**

Run: `PYTHONPATH=src pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add src/handspring/desktop_controller.py src/handspring/config.py tests/test_desktop_controller.py tests/test_config.py
git commit -m "refactor(radial): drop sub-layout + chip code; flat 6-cmd default"
```

---

### Task 5: Add EMA smoothing to the two-hand create gesture

**Files:**
- Modify: `src/handspring/desktop_controller.py`
- Test: `tests/test_desktop_controller.py`

- [ ] **Step 1: Write the failing smoothness test**

Append to `tests/test_desktop_controller.py`:

```python
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
        # old (0.45 — under-mirror so actually 0.55 or similar) and the
        # raw target.
        # Concretely: the reported x_min should be greater than 0.10 + epsilon.
        assert x_min > 0.15
```

- [ ] **Step 2: Run, verify failure**

Run: `PYTHONPATH=src pytest tests/test_desktop_controller.py::test_create_ghost_rect_ema_smooths_jitter -q`
Expected: FAIL — current code uses raw positions, so `x_min` snaps to the jumped fingertip.

- [ ] **Step 3: Add `smooth_left` / `smooth_right` to `_CreateState`**

In `src/handspring/desktop_controller.py`:

```python
@dataclass
class _CreateState:
    armed: bool = False
    cur_left: tuple[float, float] = (0.0, 0.0)
    cur_right: tuple[float, float] = (0.0, 0.0)
    smooth_left: tuple[float, float] | None = None
    smooth_right: tuple[float, float] | None = None
```

- [ ] **Step 4: EMA the fingertips in `_handle_create`**

Replace the "Armed — live-track corners" section at the end of
`_handle_create` with:

```python
# Armed — live-track corners with EMA smoothing.
alpha = self._cfg().create.smoothing
if self._create.smooth_left is None:
    self._create.smooth_left = (left_sx, left_sy)
    self._create.smooth_right = (right_sx, right_sy)
else:
    sl = self._create.smooth_left
    sr = self._create.smooth_right
    assert sr is not None
    self._create.smooth_left = (
        alpha * left_sx + (1 - alpha) * sl[0],
        alpha * left_sy + (1 - alpha) * sl[1],
    )
    self._create.smooth_right = (
        alpha * right_sx + (1 - alpha) * sr[0],
        alpha * right_sy + (1 - alpha) * sr[1],
    )
self._create.cur_left = self._create.smooth_left
self._create.cur_right = self._create.smooth_right
```

Also, at the top of `_handle_create` where the armed state is reset on
disarm, clear the smoothed positions:

```python
if not both_pinching:
    if self._create.armed:
        # ... existing commit-or-discard logic unchanged ...
        self._create.armed = False
        self._create.smooth_left = None
        self._create.smooth_right = None
    return
```

And in the "just armed" branch (`if not self._create.armed:`), initialize
smoothed state:

```python
if hand_dist < self._cfg().create.entry_distance:
    self._create.armed = True
    self._create.cur_left = (left_sx, left_sy)
    self._create.cur_right = (right_sx, right_sy)
    self._create.smooth_left = (left_sx, left_sy)
    self._create.smooth_right = (right_sx, right_sy)
return
```

- [ ] **Step 5: Run the smoothness test**

Run: `PYTHONPATH=src pytest tests/test_desktop_controller.py::test_create_ghost_rect_ema_smooths_jitter -q`
Expected: PASS.

- [ ] **Step 6: Run full suite**

Run: `PYTHONPATH=src pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/handspring/desktop_controller.py tests/test_desktop_controller.py
git commit -m "feat(create): EMA smooth the ghost rect corners"
```

---

### Task 6: Simplify the overlay

**Files:**
- Modify: `src/handspring/overlay.py`

- [ ] **Step 1: Rewrite `_draw_radial_tree` to draw pinch dot + threshold circle + 6 wedges**

Replace the body of `_draw_radial_tree` in `src/handspring/overlay.py` with:

```python
def _draw_radial_tree(
    ctx: Any,
    ox: float,
    oy: float,
    hovered_root: int | None,
    hovered_sub: int | None,
    progress: float,
    root_items: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    del hovered_sub, progress  # flick model has no sub or countdown
    r_inner = 40.0
    r_root = 220.0

    # Pinch-origin dot (where "no commit" lives).
    Quartz.CGContextSetRGBFillColor(ctx, 0.85, 0.85, 0.85, 0.85)
    Quartz.CGContextFillEllipseInRect(ctx, Quartz.CGRectMake(ox - 6, oy - 6, 12, 12))

    # Flick-threshold circle (commit boundary).
    Quartz.CGContextSetRGBStrokeColor(ctx, 0.85, 0.85, 0.85, 0.35)
    Quartz.CGContextSetLineWidth(ctx, 1.0)
    Quartz.CGContextStrokeEllipseInRect(
        ctx, Quartz.CGRectMake(ox - r_inner, oy - r_inner, 2 * r_inner, 2 * r_inner)
    )

    n = len(root_items)
    if n == 0:
        return
    slice_size = 2 * math.pi / n
    for i, (name, _) in enumerate(root_items):
        start = -math.pi / 2 + (i - 0.5) * slice_size
        end = start + slice_size
        highlighted = i == hovered_root
        _pie_wedge_path(ctx, ox, oy, r_inner, r_root, start, end)
        if highlighted:
            Quartz.CGContextSetRGBFillColor(ctx, 0.28, 0.50, 0.10, 0.88)
        else:
            Quartz.CGContextSetRGBFillColor(ctx, 0.10, 0.10, 0.10, 0.55)
        Quartz.CGContextFillPath(ctx)
        _pie_wedge_path(ctx, ox, oy, r_inner, r_root, start, end)
        if highlighted:
            Quartz.CGContextSetRGBStrokeColor(ctx, 0.53, 1.0, 0.0, 1.0)
            Quartz.CGContextSetLineWidth(ctx, 2.0)
        else:
            Quartz.CGContextSetRGBStrokeColor(ctx, 0.40, 0.40, 0.40, 0.75)
            Quartz.CGContextSetLineWidth(ctx, 1.0)
        Quartz.CGContextStrokePath(ctx)
        # Label
        center_angle = -math.pi / 2 + i * slice_size
        lr = (r_inner + r_root) * 0.5
        lx = ox + lr * math.cos(center_angle)
        ly = oy + lr * math.sin(center_angle)
        color = (
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.60, 1.0, 0.20, 1.0)
            if highlighted
            else AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.92, 0.92, 0.92, 0.95)
        )
        _draw_label(name, lx, ly, color, highlighted)
```

- [ ] **Step 2: Delete the chip row / rounded-rect helpers**

Delete from `src/handspring/overlay.py`:

- The entire `_draw_sub_chips_row` function.
- The entire `_rounded_rect_path` function.
- Any `from handspring.desktop_controller import SUB_CHIP_H, SUB_CHIP_W, compute_sub_layout` line.

- [ ] **Step 3: Run full suite**

Run: `PYTHONPATH=src pytest tests/ -q`
Expected: all PASS (overlay code isn't imported by tests on non-darwin, but the import-time changes should still not break collection).

- [ ] **Step 4: Import-smoke check**

Run: `PYTHONPATH=src python -c "import handspring.overlay; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/handspring/overlay.py
git commit -m "feat(overlay): simplify to pinch dot + threshold circle + 6 wedges"
```

---

### Task 7: README + CLI help sync

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the "Desktop mode: the radial menu" section**

Open `README.md`, find the section header `## Desktop mode: the radial menu`, and replace its body (through the next `## ` header) with:

```markdown
## Desktop mode: the radial menu

Pinch your **left** hand to open the radial. There's no hold — labels appear instantly around the pinch point. Move your hand a tiny bit toward one of the six commands, then release pinch to fire it. Release at the origin (or very close to it) cancels.

Default commands, clockwise from 12 o'clock:

| Position | Command |
|---|---|
| Top | None (disable gestures) |
| Upper-right | Settings (opens the settings web UI) |
| Lower-right | Mission Control |
| Bottom | New Finder window |
| Lower-left | Scroll mode (left hand y → scroll wheel) |
| Upper-left | Screenshot (whole screen) |

**Why this is fast:**
- No hold-to-open delay.
- Direction is only *committed* on release — intermediate jitter doesn't mis-fire.
- Angular hysteresis prevents flicker between neighboring slices.
- Only one level; no sub-menus.

**Customizing.** Edit `~/.config/handspring/config.toml`'s `radial_tree` section (or use the Settings UI) to reorder, add, or swap commands. Built-in command names are `None`, `Settings`, `Mission`, `Create`, `Scroll`, `Screenshot`. Any other name with a `command = "..."` field runs that shell command when fired, so you can add e.g.:

\`\`\`toml
[[radial_tree]]
name    = "Slack"
command = "open -a Slack"
\`\`\`

Tuning (camera-space units):

\`\`\`toml
[radial]
flick_threshold    = 0.03   # minimum displacement to commit (smaller = twitchier)
angular_hysteresis = 0.15   # larger = stickier slice boundaries
\`\`\`

The **two-hand create gesture** (pinch both hands near each other, pull apart, release) is unchanged. Its ghost rectangle now smooths with an EMA (`[create] smoothing = 0.35`) so the preview doesn't jitter while you're sizing.
```

- [ ] **Step 2: Remove stale references to sub-menus**

Search for and remove any remaining mentions of: "sub-ring", "sub ring", "More →", "Window tiles", "Desktops L/R", "Screenshot variants", "hold 0.4s", "pinch-and-hold" in non-historical text. Keep references in the Versions changelog section as-is.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: radial redesign — flick-commit, flat 6 commands"
```

---

### Task 8: Full verification

- [ ] **Step 1: Lint + full test run**

Run: `ruff check src/ tests/ && PYTHONPATH=src python -m pytest tests/ -q`
Expected: `All checks passed!` and all tests PASS. Fix any issues inline.

- [ ] **Step 2: Smoke-test the imports end-to-end**

Run:
```
PYTHONPATH=src python -c "
from handspring.desktop_controller import DesktopController
from handspring.config import Config, ConfigStore
from handspring.settings_server import SettingsServer
c = DesktopController(mirrored=False)
print('tree:', [n for n, _ in c.root_items()])
print('flick_threshold:', Config().radial.flick_threshold)
print('create.smoothing:', Config().create.smoothing)
print('ok')
"
```
Expected: prints the 6 command names, 0.03, 0.35, `ok`.

- [ ] **Step 3: Push to main**

Fast-forward push (the user has already approved this pattern for this worktree):

```bash
git fetch origin main
git merge-base --is-ancestor origin/main HEAD && echo FF-OK
git push origin HEAD:main
```

Expected: `FF-OK` and a successful push.

---

## Self-Review Checklist

1. **Spec coverage:**
   - Flick interaction model → Task 2. ✅
   - 6 commands + layout → Task 4 (default tree) + Task 7 (README). ✅
   - Config adds/removes → Task 1 + Task 4. ✅
   - Angular hysteresis → Task 2. ✅
   - Create smoothing → Task 5. ✅
   - Overlay simplification → Task 6. ✅
   - Test updates → Tasks 2, 4, 5. ✅
   - README update → Task 7. ✅

2. **Placeholder scan:** No "TBD"/"TODO"/"similar to" strings. Every code step contains the actual code.

3. **Type consistency:** `RadialConfig` and `CreateConfig` definitions in Task 1 match the fields referenced in Tasks 2, 3, 5. `_CreateState` extended fields in Task 5 match the assignments in `_handle_create`. The `radial_tree` shape from Task 4 matches what `_commit_radial` reads in Task 3.
