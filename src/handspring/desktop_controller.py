"""Gesture → macOS-action state machine.

Replaces JarvisController when --os-control is active. Reads FrameResult,
drives the OS via handspring.os_control.

Gestures:
- Cursor follows right index fingertip (continuous, no gesture needed).
- Right-hand pinch = left-mouse button held down. Move hand = drag. Unpinch = release.
- Both-hand pinch with hands close together, then pull apart = new Finder window.
- Both-hand FIST held for 5 seconds = toggle "disabled" mode (failsafe).
  While disabled, no events fire. Repeat to re-enable.

All tuning thresholds and the radial tree are read from a ConfigStore each
update. Hand-edits to the config file or writes from the settings web UI
take effect on the next frame — no restart needed.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

from handspring import os_control
from handspring.config import Config, ConfigStore, RadialItem
from handspring.features import is_pinching
from handspring.types import FrameResult, HandFeatures

# Kept as a module-level default so _cam_to_screen in __main__ can still read
# an inset for overlay positioning even when no store is active. The live
# value always comes from the config.
_CURSOR_INSET = 0.08


@dataclass
class _CursorState:
    pressed: bool = False
    last_x: int = 0
    last_y: int = 0
    smooth_nx: float | None = None
    smooth_ny: float | None = None


@dataclass
class _CreateState:
    armed: bool = False
    cur_left: tuple[float, float] = (0.0, 0.0)
    cur_right: tuple[float, float] = (0.0, 0.0)


@dataclass
class _RadialState:
    pinching: bool = False
    pinch_start: float = 0.0
    active: bool = False
    origin: tuple[float, float] = (0.0, 0.0)
    cur: tuple[float, float] = (0.0, 0.0)
    hovered_root: int | None = None
    hovered_sub: int | None = None


class DesktopController:
    def __init__(
        self,
        *,
        mirrored: bool = True,
        store: ConfigStore | None = None,
        on_open_settings: Callable[[], None] | None = None,
        on_reload_config: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ) -> None:
        self._mirrored = mirrored
        self._store: ConfigStore = (
            store if store is not None else ConfigStore(persist=False)
        )
        self._on_open_settings = on_open_settings
        self._on_reload_config = on_reload_config
        self._on_quit = on_quit
        self._cursor = _CursorState()
        self._create = _CreateState()
        self._radial = _RadialState()
        self._disabled = False
        self._failsafe_start: float | None = None
        self._screen_w, self._screen_h = os_control.screen_size()
        self._events_out: list[str] = []
        self._mode: str = "create"
        # Default selected app: first submenu entry of "Create" if present.
        tree = self._store.get().radial_tree
        self._selected_app: str = _first_create_app(tree)
        self._left_cursor_screen: tuple[int, int] | None = None
        self._post_spawn: tuple[tuple[int, int, int, int], float] | None = None
        self._post_spawn_hold_seconds = 2.0

    # ---- config access --------------------------------------------------

    @property
    def store(self) -> ConfigStore:
        return self._store

    def _cfg(self) -> Config:
        return self._store.get()

    def selected_app(self) -> str:
        return self._selected_app

    def mode(self) -> str:
        return self._mode

    def root_items(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple((it.name, it.subs) for it in self._cfg().radial_tree)

    def left_cursor_screen(self) -> tuple[int, int] | None:
        return self._left_cursor_screen

    def pending_create_screen_bounds(self) -> tuple[int, int, int, int] | None:
        rect = self.pending_create_bounds()
        if rect is None:
            return None
        x, y, w, h = rect
        return (
            int(x * self._screen_w),
            int(y * self._screen_h),
            int((x + w) * self._screen_w),
            int((y + h) * self._screen_h),
        )

    def post_spawn_screen_bounds(self) -> tuple[int, int, int, int] | None:
        if self._post_spawn is None:
            return None
        bounds, expire = self._post_spawn
        if self._last_now >= expire:
            self._post_spawn = None
            return None
        return bounds

    def enabled(self) -> bool:
        return not self._disabled

    def failsafe_progress(self) -> float:
        if self._failsafe_start is None:
            return 0.0
        elapsed = self._last_now - self._failsafe_start
        return max(0.0, min(1.0, elapsed / self._cfg().failsafe.hold_seconds))

    def pop_events(self) -> list[str]:
        out = self._events_out
        self._events_out = []
        return out

    # ---- main loop ------------------------------------------------------

    def update(self, frame: FrameResult, now: float) -> None:
        self._last_now = now
        if self._handle_failsafe(frame, now):
            return
        if self._disabled:
            if self._cursor.pressed:
                os_control.mouse_up(self._cursor.last_x, self._cursor.last_y)
                self._cursor.pressed = False
            return

        self._handle_cursor(frame)
        if self._mode == "create":
            self._handle_create(frame)
        self._handle_radial(frame, now)
        if self._mode == "scroll":
            self._handle_scroll(frame)

    # ---- failsafe -------------------------------------------------------

    def _handle_failsafe(self, frame: FrameResult, now: float) -> bool:
        both_fist = (
            frame.left.present
            and frame.right.present
            and frame.left.gesture == "fist"
            and frame.right.gesture == "fist"
        )
        if not both_fist:
            self._failsafe_start = None
            return False
        if self._failsafe_start is None:
            self._failsafe_start = now
            return False
        if now - self._failsafe_start >= self._cfg().failsafe.hold_seconds:
            self._disabled = not self._disabled
            self._failsafe_start = None
            self._events_out.append("disabled" if self._disabled else "enabled")
            if self._cursor.pressed:
                os_control.mouse_up(self._cursor.last_x, self._cursor.last_y)
                self._cursor.pressed = False
            return True
        return False

    # ---- cursor ---------------------------------------------------------

    def _handle_cursor(self, frame: FrameResult) -> None:
        right = frame.right
        if not right.present or right.features is None:
            if self._cursor.pressed:
                os_control.mouse_up(self._cursor.last_x, self._cursor.last_y)
                self._cursor.pressed = False
            self._cursor.smooth_nx = None
            self._cursor.smooth_ny = None
            return

        cfg = self._cfg()
        inset = cfg.cursor.inset
        smoothing = cfg.cursor.smoothing

        f = right.features
        raw_nx = (f.index_x + f.thumb_x) * 0.5
        raw_ny = (f.index_y + f.thumb_y) * 0.5
        if self._mirrored:
            raw_nx = 1.0 - raw_nx

        if self._cursor.smooth_nx is None:
            self._cursor.smooth_nx = raw_nx
            self._cursor.smooth_ny = raw_ny
        else:
            a = smoothing
            self._cursor.smooth_nx = a * raw_nx + (1 - a) * self._cursor.smooth_nx
            assert self._cursor.smooth_ny is not None
            self._cursor.smooth_ny = a * raw_ny + (1 - a) * self._cursor.smooth_ny

        nx = self._cursor.smooth_nx
        ny = self._cursor.smooth_ny
        assert ny is not None
        span = max(1e-6, 1.0 - 2 * inset)
        mapped_x = (nx - inset) / span
        mapped_y = (ny - inset) / span
        sx = int(mapped_x * self._screen_w)
        sy = int(mapped_y * self._screen_h)
        sx = max(0, min(self._screen_w - 1, sx))
        sy = max(0, min(self._screen_h - 1, sy))

        pinching = is_pinching(right)

        if pinching and not self._cursor.pressed:
            os_control.mouse_down(sx, sy)
            self._cursor.pressed = True
            self._events_out.append("click_down")
        elif pinching and self._cursor.pressed:
            os_control.mouse_drag(sx, sy)
        elif not pinching and self._cursor.pressed:
            os_control.mouse_up(sx, sy)
            self._cursor.pressed = False
            self._events_out.append("click_up")
        else:
            os_control.move_cursor(sx, sy)

        self._cursor.last_x = sx
        self._cursor.last_y = sy

    # ---- create gesture -------------------------------------------------

    def _index_screen(self, features: HandFeatures) -> tuple[float, float]:
        ix = features.index_x
        iy = features.index_y
        if self._mirrored:
            ix = 1.0 - ix
        return float(ix), float(iy)

    def pending_create_bounds(self) -> tuple[float, float, float, float] | None:
        if not self._create.armed:
            return None
        lx, ly = self._create.cur_left
        rx, ry = self._create.cur_right
        x_min, x_max = min(lx, rx), max(lx, rx)
        y_min, y_max = min(ly, ry), max(ly, ry)
        return x_min, y_min, x_max - x_min, y_max - y_min

    # ---- radial ---------------------------------------------------------

    def radial_state(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], int | None, int | None, float] | None:
        r = self._radial
        if not r.pinching:
            return None
        hold = self._cfg().radial.hold_seconds
        progress = min(1.0, (self._last_now - r.pinch_start) / max(1e-6, hold))
        if not r.active and progress < 0.05:
            return None
        return r.origin, r.cur, r.hovered_root, r.hovered_sub, progress

    def radial_apps(self) -> tuple[str, ...]:
        for it in self._cfg().radial_tree:
            if it.name == "Create":
                return it.subs
        return ()

    def _slice_index(self, dx: float, dy: float, n: int) -> int:
        import math

        if self._mirrored:
            dx = -dx
        angle = math.atan2(dy, dx)
        cw = (angle + math.pi / 2) % (2 * math.pi)
        slice_size = 2 * math.pi / n
        return int((cw + slice_size / 2) // slice_size) % n

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
            if r.pinching and r.active:
                self._commit_radial(r.hovered_root, r.hovered_sub)
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
            r.pinching = True
            r.pinch_start = now
            r.active = False
            r.origin = (cx, cy)
            r.cur = (cx, cy)
            r.hovered_root = None
            r.hovered_sub = None
            return

        r.cur = (cx, cy)
        if not r.active and (now - r.pinch_start) >= cfg.radial.hold_seconds:
            r.active = True

        if not r.active:
            return

        dx = cx - r.origin[0]
        dy = cy - r.origin[1]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < cfg.radial.inner_radius:
            r.hovered_root = None
            r.hovered_sub = None
            return

        n_roots = len(tree)
        if n_roots == 0:
            return

        if dist < cfg.radial.sub_threshold:
            r.hovered_root = self._slice_index(dx, dy, n_roots)
            r.hovered_sub = None
            return

        if r.hovered_root is None or r.hovered_root >= n_roots:
            r.hovered_root = self._slice_index(dx, dy, n_roots)
        subs = tree[r.hovered_root].subs
        if subs:
            r.hovered_sub = self._slice_index(dx, dy, len(subs))
        else:
            r.hovered_sub = None

    def _commit_radial(self, root_idx: int | None, sub_idx: int | None) -> None:
        if root_idx is None:
            return
        tree = self._cfg().radial_tree
        if root_idx >= len(tree):
            return
        item = tree[root_idx]
        name = item.name
        subs = item.subs
        sub_name = subs[sub_idx] if (sub_idx is not None and sub_idx < len(subs)) else None

        # Built-in roots.
        if name == "None":
            self._mode = "none"
            self._events_out.append("mode:none")
            return
        if name == "Create":
            self._mode = "create"
            if sub_name is not None:
                self._selected_app = sub_name
                self._events_out.append(f"select_app:{self._selected_app}")
            self._events_out.append("mode:create")
            return
        if name == "Scroll":
            self._mode = "scroll"
            self._events_out.append("mode:scroll")
            return
        if name == "Screenshot":
            variant = (sub_name or "Screen").lower()
            os_control.screenshot(variant)
            self._events_out.append(f"screenshot:{variant}")
            return
        if name == "Window":
            sub = sub_name or "Center"
            if sub == "Close":
                os_control.close_frontmost_window()
            elif sub == "Minimize":
                os_control.minimize_front_window()
            elif sub == "Fullscreen":
                os_control.fullscreen_front_window()
            elif sub in ("Left", "Right", "Center"):
                os_control.tile_front_window(sub.lower())
            self._events_out.append(f"window:{sub.lower()}")
            return
        if name == "Mission":
            os_control.mission_control()
            self._events_out.append("mission")
            return
        if name == "Desktops":
            direction = (sub_name or "Right").lower()
            os_control.switch_desktop(direction)
            self._events_out.append(f"desktop:{direction}")
            return
        if name == "More":
            self._handle_more(sub_name)
            return

        # User-defined root with a custom command on the root item itself
        # (leaf with no subs) OR a custom sub selection. If the selected
        # RadialItem has a command string, run it.
        if item.command and sub_name is None:
            self._run_command(item.command)
            self._events_out.append(f"run:{name}")

    def _handle_more(self, sub_name: str | None) -> None:
        if sub_name == "Settings":
            self._events_out.append("settings:open")
            if self._on_open_settings is not None:
                try:
                    self._on_open_settings()
                except Exception as e:  # noqa: BLE001
                    print(f"handspring: settings open failed ({e})", file=sys.stderr)
        elif sub_name == "Reload":
            self._events_out.append("config:reload")
            if self._on_reload_config is not None:
                try:
                    self._on_reload_config()
                except Exception as e:  # noqa: BLE001
                    print(f"handspring: reload failed ({e})", file=sys.stderr)
            else:
                self._store.reload()
        elif sub_name == "Quit":
            self._events_out.append("app:quit")
            if self._on_quit is not None:
                try:
                    self._on_quit()
                except Exception as e:  # noqa: BLE001
                    print(f"handspring: quit handler failed ({e})", file=sys.stderr)

    @staticmethod
    def _run_command(command: str) -> None:
        """Run a user-defined shell command in the background, non-blocking."""
        try:
            subprocess.Popen(  # noqa: S603
                shlex.split(command),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, ValueError) as e:
            print(f"handspring: command failed ({command!r}): {e}", file=sys.stderr)

    # ---- scroll ---------------------------------------------------------

    def _handle_scroll(self, frame: FrameResult) -> None:
        left = frame.left
        if not left.present or left.features is None:
            return
        cfg = self._cfg().scroll
        f = left.features
        ny = (f.index_y + f.thumb_y) * 0.5
        delta = ny - 0.5
        if abs(delta) < cfg.deadzone:
            return
        direction = -1 if delta < 0 else 1
        mag = (abs(delta) - cfg.deadzone) / max(1e-6, (0.5 - cfg.deadzone))
        mag = max(0.0, min(1.0, mag))
        dy_pixels = -direction * int(cfg.max_pixels * mag)
        if dy_pixels != 0:
            os_control.scroll(dy_pixels)

    def _handle_create(self, frame: FrameResult) -> None:
        left = frame.left
        right = frame.right
        cfg = self._cfg().create
        both_pinching = (
            is_pinching(left)
            and is_pinching(right)
            and left.features is not None
            and right.features is not None
        )
        if not both_pinching:
            if self._create.armed:
                lx, ly = self._create.cur_left
                rx, ry = self._create.cur_right
                dx = rx - lx
                dy = ry - ly
                diag = (dx * dx + dy * dy) ** 0.5
                if diag >= cfg.min_diagonal:
                    x_min, x_max = min(lx, rx), max(lx, rx)
                    y_min, y_max = min(ly, ry), max(ly, ry)
                    bx1 = int(x_min * self._screen_w)
                    by1 = int(y_min * self._screen_h)
                    bx2 = int(x_max * self._screen_w)
                    by2 = int(y_max * self._screen_h)
                    if bx2 - bx1 < 200:
                        bx2 = bx1 + 200
                    if by2 - by1 < 150:
                        by2 = by1 + 150
                    os_control.new_app_window(self._selected_app, bounds=(bx1, by1, bx2, by2))
                    self._events_out.append(f"new_window:{self._selected_app}")
                    self._post_spawn = (
                        (bx1, by1, bx2, by2),
                        self._last_now + self._post_spawn_hold_seconds,
                    )
                self._create.armed = False
            return
        assert left.features is not None and right.features is not None
        left_sx, left_sy = self._index_screen(left.features)
        right_sx, right_sy = self._index_screen(right.features)
        hdx = left_sx - right_sx
        hdy = left_sy - right_sy
        hand_dist = (hdx * hdx + hdy * hdy) ** 0.5
        if not self._create.armed:
            if hand_dist < self._cfg().create.entry_distance:
                self._create.armed = True
                self._create.cur_left = (left_sx, left_sy)
                self._create.cur_right = (right_sx, right_sy)
            return
        self._create.cur_left = (left_sx, left_sy)
        self._create.cur_right = (right_sx, right_sy)


def _first_create_app(tree: tuple[RadialItem, ...]) -> str:
    for it in tree:
        if it.name == "Create" and it.subs:
            return it.subs[0]
    return "Finder"
