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
from handspring.types import FrameResult

_CREATE_ENTRY_DISTANCE = 0.08  # index-tip distance threshold to arm "new window"
_CREATE_PULL_RELEASE = 0.25  # hands must pull to this distance before release to fire
_FAILSAFE_HOLD_SECONDS = 5.0  # both-fist hold duration to toggle disabled


@dataclass
class _CursorState:
    """Bookkeeping for the right-hand click/drag state."""

    pressed: bool = False
    last_x: int = 0
    last_y: int = 0


@dataclass
class _CreateState:
    """Both-hand-pinch arming state for new-window gesture."""

    armed: bool = False  # hands were close together while both pinching
    start_distance: float = 0.0


class DesktopController:
    def __init__(self, *, mirrored: bool = True) -> None:
        self._mirrored = mirrored
        self._cursor = _CursorState()
        self._create = _CreateState()
        self._disabled = False
        self._failsafe_start: float | None = None
        self._screen_w, self._screen_h = os_control.screen_size()
        self._events_out: list[str] = []

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
            # Hand left frame: release any held click.
            if self._cursor.pressed:
                os_control.mouse_up(self._cursor.last_x, self._cursor.last_y)
                self._cursor.pressed = False
            return

        # Map index-tip normalized coords → screen pixels. When mirrored, the
        # user sees a flipped image; flip x to map "my hand to the right of my
        # body" → "cursor on the right of my screen".
        nx = 1.0 - right.features.index_x if self._mirrored else right.features.index_x
        ny = right.features.index_y
        sx = int(nx * self._screen_w)
        sy = int(ny * self._screen_h)
        # Clamp to valid screen range.
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
                # Released — only commit if we'd also pulled apart enough.
                self._create.armed = False
            return
        assert left.features is not None and right.features is not None
        dx = left.features.index_x - right.features.index_x
        dy = left.features.index_y - right.features.index_y
        distance = (dx * dx + dy * dy) ** 0.5
        if not self._create.armed:
            if distance < _CREATE_ENTRY_DISTANCE:
                self._create.armed = True
                self._create.start_distance = distance
            return
        # Armed — wait for pull-apart + release to fire.
        if distance >= _CREATE_PULL_RELEASE:
            os_control.new_finder_window()
            self._events_out.append("new_finder_window")
            self._create.armed = False
