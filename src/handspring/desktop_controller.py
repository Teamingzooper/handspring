"""Gesture → macOS-action state machine.

Replaces JarvisController when --os-control is active. Reads FrameResult,
drives the OS via handspring.os_control.

Gestures:
- Cursor follows right index fingertip (continuous, no gesture needed).
- Right-hand pinch = left-mouse button held down. Move hand = drag. Unpinch = release.
- Both-hand pinch with hands close together, then pull apart = new Finder window.
- Both-hand FIST held for 5 seconds = toggle "disabled" mode (failsafe).
  While disabled, no events fire. Repeat to re-enable.
"""

from __future__ import annotations

from dataclasses import dataclass

from handspring import os_control
from handspring.features import is_pinching
from handspring.types import FrameResult, HandFeatures

_CREATE_ENTRY_DISTANCE = 0.08  # index-tip distance threshold to arm "new window"
_CREATE_MIN_DIAGONAL = 0.15  # hands must pull to this diagonal distance to commit
_FAILSAFE_HOLD_SECONDS = 5.0  # both-fist hold duration to toggle disabled
# EMA smoothing factor for the cursor. Higher = more responsive but more jittery.
_CURSOR_SMOOTHING = 0.35
# Camera-to-screen input inset. Camera coords in [INSET, 1 - INSET] map to
# screen coords [0, 1]; anything beyond clamps to the screen edge. This means
# the user doesn't need to stretch to the extreme camera frame edges to reach
# the dock or menubar — the "comfortable middle" covers the whole screen.
_CURSOR_INSET = 0.08

# Radial app launcher (left hand).
_RADIAL_HOLD_SECONDS = 0.4  # pinch must be held this long to open the wheel
_RADIAL_SELECT_RADIUS = 0.05  # hand must move this far from origin to commit a slice
_RADIAL_APPS: tuple[str, ...] = (
    "Finder",
    "Safari",
    "Messages",
    "Notes",
    "Terminal",
    "Music",
)


@dataclass
class _CursorState:
    """Bookkeeping for the right-hand click/drag state."""

    pressed: bool = False
    last_x: int = 0
    last_y: int = 0
    # Smoothed normalized screen coords (0..1). None until first frame.
    smooth_nx: float | None = None
    smooth_ny: float | None = None


@dataclass
class _CreateState:
    """Both-hand-pinch arming state for new-window gesture.

    While armed we track the live bounding box of the two index fingertips so
    we can commit a correctly-sized window on release.
    """

    armed: bool = False  # hands were close together while both pinching
    # Live fingertip positions (screen-space, mirrored already applied).
    cur_left: tuple[float, float] = (0.0, 0.0)
    cur_right: tuple[float, float] = (0.0, 0.0)


@dataclass
class _RadialState:
    """Left-hand pinch-and-hold radial-menu state.

    origin stores the raw (un-mirrored) camera-space pinch position, so the
    preview can mirror when drawing.
    """

    pinching: bool = False
    pinch_start: float = 0.0  # monotonic time of pinch start
    active: bool = False  # True once the hold duration has elapsed
    origin: tuple[float, float] = (0.0, 0.0)  # raw camera coords
    cur: tuple[float, float] = (0.0, 0.0)  # raw camera coords
    selected: int | None = None  # index into _RADIAL_APPS, or None (center)


class DesktopController:
    def __init__(self, *, mirrored: bool = True) -> None:
        self._mirrored = mirrored
        self._cursor = _CursorState()
        self._create = _CreateState()
        self._radial = _RadialState()
        self._disabled = False
        self._failsafe_start: float | None = None
        self._screen_w, self._screen_h = os_control.screen_size()
        self._events_out: list[str] = []
        # App to spawn on the next both-hand-pinch-pull-apart. Defaults to Finder.
        self._selected_app: str = _RADIAL_APPS[0]
        # Exposed for the native overlay: left-hand midpoint in screen pixels.
        self._left_cursor_screen: tuple[int, int] | None = None

    def selected_app(self) -> str:
        return self._selected_app

    def left_cursor_screen(self) -> tuple[int, int] | None:
        """Screen pixel coords for the left hand (for the native overlay)."""
        return self._left_cursor_screen

    def enabled(self) -> bool:
        return not self._disabled

    def failsafe_progress(self) -> float:
        """0..1 indicating how far into the failsafe-hold countdown we are."""
        if self._failsafe_start is None:
            return 0.0
        # caller must pass `now` in update(); we stash it for use here via _last_now.
        elapsed = self._last_now - self._failsafe_start
        return max(0.0, min(1.0, elapsed / _FAILSAFE_HOLD_SECONDS))

    def pop_events(self) -> list[str]:
        out = self._events_out
        self._events_out = []
        return out

    # ---- main loop ----

    def update(self, frame: FrameResult, now: float) -> None:
        self._last_now = now

        # Always check failsafe first so the user can toggle regardless of state.
        if self._handle_failsafe(frame, now):
            # A toggle fired — don't do anything else this frame.
            return

        if self._disabled:
            # Release any held click so we don't get stuck.
            if self._cursor.pressed:
                os_control.mouse_up(self._cursor.last_x, self._cursor.last_y)
                self._cursor.pressed = False
            return

        self._handle_cursor(frame)
        self._handle_create(frame)
        self._handle_radial(frame, now)

    # ---- failsafe: both-fist 5s hold ----

    def _handle_failsafe(self, frame: FrameResult, now: float) -> bool:
        """Returns True when a toggle fires (to skip other handlers this frame)."""
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
        if now - self._failsafe_start >= _FAILSAFE_HOLD_SECONDS:
            self._disabled = not self._disabled
            self._failsafe_start = None
            self._events_out.append("disabled" if self._disabled else "enabled")
            # Release any stuck click.
            if self._cursor.pressed:
                os_control.mouse_up(self._cursor.last_x, self._cursor.last_y)
                self._cursor.pressed = False
            return True
        return False

    # ---- right-hand cursor + click ----

    def _handle_cursor(self, frame: FrameResult) -> None:
        right = frame.right
        if not right.present or right.features is None:
            # Hand left frame: release any held click and reset smoothing so
            # we don't interpolate from a stale position when the hand returns.
            if self._cursor.pressed:
                os_control.mouse_up(self._cursor.last_x, self._cursor.last_y)
                self._cursor.pressed = False
            self._cursor.smooth_nx = None
            self._cursor.smooth_ny = None
            return

        f = right.features
        # Use the MIDPOINT of thumb tip + index tip as the cursor anchor.
        # Pinching collapses both tips together, so the midpoint stays stable
        # right where the pinch happens — the cursor doesn't drift downward
        # when the index finger curls to meet the thumb.
        raw_nx = (f.index_x + f.thumb_x) * 0.5
        raw_ny = (f.index_y + f.thumb_y) * 0.5
        if self._mirrored:
            raw_nx = 1.0 - raw_nx

        # Low-pass / EMA smoothing: tweens the cursor toward the raw target
        # each frame instead of jumping there. Kills MediaPipe's per-frame
        # landmark jitter without adding noticeable lag.
        if self._cursor.smooth_nx is None:
            self._cursor.smooth_nx = raw_nx
            self._cursor.smooth_ny = raw_ny
        else:
            a = _CURSOR_SMOOTHING
            self._cursor.smooth_nx = a * raw_nx + (1 - a) * self._cursor.smooth_nx
            assert self._cursor.smooth_ny is not None
            self._cursor.smooth_ny = a * raw_ny + (1 - a) * self._cursor.smooth_ny

        nx = self._cursor.smooth_nx
        ny = self._cursor.smooth_ny
        assert ny is not None
        # Apply input inset: camera [INSET, 1-INSET] → screen [0, 1].
        span = 1.0 - 2 * _CURSOR_INSET
        mapped_x = (nx - _CURSOR_INSET) / span
        mapped_y = (ny - _CURSOR_INSET) / span
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
            # Continue drag at new position.
            os_control.mouse_drag(sx, sy)
        elif not pinching and self._cursor.pressed:
            os_control.mouse_up(sx, sy)
            self._cursor.pressed = False
            self._events_out.append("click_up")
        else:
            # Plain cursor move.
            os_control.move_cursor(sx, sy)

        self._cursor.last_x = sx
        self._cursor.last_y = sy

    # ---- both-hand pinch pull-apart → new Finder window ----

    def _index_screen(self, features: HandFeatures) -> tuple[float, float]:
        """Return index-tip in screen-normalized coords (mirror applied)."""
        ix = features.index_x
        iy = features.index_y
        if self._mirrored:
            ix = 1.0 - ix
        return float(ix), float(iy)

    def pending_create_bounds(self) -> tuple[float, float, float, float] | None:
        """While arming, return the live bounding rect (in screen 0..1 coords).

        Used by the preview to draw a "NEW" ghost rectangle.
        """
        if not self._create.armed:
            return None
        lx, ly = self._create.cur_left
        rx, ry = self._create.cur_right
        x_min, x_max = min(lx, rx), max(lx, rx)
        y_min, y_max = min(ly, ry), max(ly, ry)
        return x_min, y_min, x_max - x_min, y_max - y_min

    # ---- left-hand radial app launcher ----

    def radial_state(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], int | None, float] | None:
        """Return (origin_raw, cur_raw, selected_idx, progress) while active,
        or None when not in radial mode. Progress is 0..1 during the hold
        countdown; 1.0 once the wheel is active.
        """
        r = self._radial
        if not r.pinching:
            return None
        progress = min(1.0, (self._last_now - r.pinch_start) / _RADIAL_HOLD_SECONDS)
        # Only show the countdown if something noticeable has elapsed.
        if not r.active and progress < 0.15:
            return None
        return r.origin, r.cur, r.selected, progress

    @staticmethod
    def radial_apps() -> tuple[str, ...]:
        return _RADIAL_APPS

    def _handle_radial(self, frame: FrameResult, now: float) -> None:
        r = self._radial
        left = frame.left
        right = frame.right

        # Track the left-hand screen-space position for the native overlay,
        # regardless of radial state.
        if left.present and left.features is not None:
            lf = left.features
            lmx = (lf.index_x + lf.thumb_x) * 0.5
            lmy = (lf.index_y + lf.thumb_y) * 0.5
            if self._mirrored:
                lmx = 1.0 - lmx
            span = 1.0 - 2 * _CURSOR_INSET
            sx = int(((lmx - _CURSOR_INSET) / span) * self._screen_w)
            sy = int(((lmy - _CURSOR_INSET) / span) * self._screen_h)
            sx = max(0, min(self._screen_w - 1, sx))
            sy = max(0, min(self._screen_h - 1, sy))
            self._left_cursor_screen = (sx, sy)
        else:
            self._left_cursor_screen = None

        # If right hand is pinching, we're either clicking or in the middle of
        # a both-hand-create gesture — suppress the radial entirely so left
        # pinches don't also pop the wheel.
        right_pinching = is_pinching(right)
        pinching = (
            is_pinching(left)
            and left.features is not None
            and not right_pinching
            and not self._create.armed
        )

        if not pinching:
            if r.pinching and r.active and r.selected is not None:
                # Commit: SET the selected-app, don't launch anything yet.
                self._selected_app = _RADIAL_APPS[r.selected]
                self._events_out.append(f"select_app:{self._selected_app}")
            # Reset everything.
            r.pinching = False
            r.active = False
            r.selected = None
            return

        assert left.features is not None
        f = left.features
        # Use index/thumb midpoint for stability (same trick as cursor).
        cx = (f.index_x + f.thumb_x) * 0.5
        cy = (f.index_y + f.thumb_y) * 0.5

        if not r.pinching:
            r.pinching = True
            r.pinch_start = now
            r.active = False
            r.origin = (cx, cy)
            r.cur = (cx, cy)
            r.selected = None
            return

        r.cur = (cx, cy)
        if not r.active and (now - r.pinch_start) >= _RADIAL_HOLD_SECONDS:
            r.active = True

        if r.active:
            # Decide which slice the current hand position falls into.
            dx = cx - r.origin[0]
            # Flip dy sign for angle so up = -y is "north" at angle 0.
            dy = cy - r.origin[1]
            # When preview is mirrored, the user's "right" in meatspace is
            # raw camera LEFT (smaller x). Flip dx for angle so that moving
            # the hand visually right in the preview selects slices on the
            # right of the wheel.
            if self._mirrored:
                dx = -dx
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < _RADIAL_SELECT_RADIUS:
                r.selected = None
            else:
                import math

                # atan2 gives angle from +x axis, CCW positive; image y is
                # DOWN, so the returned angle treats "north" as -y (we want
                # slice 0 at top of wheel). Rotate so 0 = up.
                angle = math.atan2(dy, dx)  # -π..π, 0 = right, π/2 = down
                # Convert to "clockwise from top": 0 = up, π/2 = right, π = down.
                cw = (angle + math.pi / 2) % (2 * math.pi)
                n = len(_RADIAL_APPS)
                slice_size = 2 * math.pi / n
                # Add half-slice so each app owns a slice centered on its direction.
                r.selected = int((cw + slice_size / 2) // slice_size) % n

    def _handle_create(self, frame: FrameResult) -> None:
        left = frame.left
        right = frame.right
        both_pinching = (
            is_pinching(left)
            and is_pinching(right)
            and left.features is not None
            and right.features is not None
        )
        if not both_pinching:
            if self._create.armed:
                # Released — commit a new Finder window if we pulled apart far enough.
                lx, ly = self._create.cur_left
                rx, ry = self._create.cur_right
                dx = rx - lx
                dy = ry - ly
                diag = (dx * dx + dy * dy) ** 0.5
                if diag >= _CREATE_MIN_DIAGONAL:
                    x_min, x_max = min(lx, rx), max(lx, rx)
                    y_min, y_max = min(ly, ry), max(ly, ry)
                    # Convert 0..1 screen coords to pixel bounds.
                    bx1 = int(x_min * self._screen_w)
                    by1 = int(y_min * self._screen_h)
                    bx2 = int(x_max * self._screen_w)
                    by2 = int(y_max * self._screen_h)
                    # Enforce a minimum pixel size to avoid degenerate windows.
                    if bx2 - bx1 < 200:
                        bx2 = bx1 + 200
                    if by2 - by1 < 150:
                        by2 = by1 + 150
                    os_control.new_app_window(self._selected_app, bounds=(bx1, by1, bx2, by2))
                    self._events_out.append(f"new_window:{self._selected_app}")
                self._create.armed = False
            return
        assert left.features is not None and right.features is not None
        left_sx, left_sy = self._index_screen(left.features)
        right_sx, right_sy = self._index_screen(right.features)
        # Hand-separation in normalized screen coords (not raw, since we flipped x).
        hdx = left_sx - right_sx
        hdy = left_sy - right_sy
        hand_dist = (hdx * hdx + hdy * hdy) ** 0.5
        if not self._create.armed:
            if hand_dist < _CREATE_ENTRY_DISTANCE:
                self._create.armed = True
                self._create.cur_left = (left_sx, left_sy)
                self._create.cur_right = (right_sx, right_sy)
            return
        # Armed — live-track corners.
        self._create.cur_left = (left_sx, left_sy)
        self._create.cur_right = (right_sx, right_sy)
