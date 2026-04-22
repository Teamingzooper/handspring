"""macOS OS-level control primitives: cursor, clicks, Finder windows.

All functions are no-ops on non-macOS platforms (they import guarded). Requires
Accessibility permission for the running Python interpreter in System Settings
→ Privacy & Security → Accessibility.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from typing import Any

_MAC = sys.platform == "darwin"

if _MAC:
    try:
        import Quartz
        from AppKit import NSScreen

        _AVAILABLE = True
    except ImportError:
        _AVAILABLE = False
else:
    _AVAILABLE = False


def available() -> bool:
    """True if we can actually drive the OS on this platform."""
    return _AVAILABLE


def screen_size() -> tuple[int, int]:
    """Return (width, height) of the main screen in pixels. Fallback (1440, 900)."""
    if not _AVAILABLE:
        return (1440, 900)
    screen = NSScreen.mainScreen()
    f = screen.frame()
    return int(f.size.width), int(f.size.height)


def _cg_point(x: int, y: int) -> Any:
    return Quartz.CGPoint(x, y)


def move_cursor(x: int, y: int) -> None:
    if not _AVAILABLE:
        return
    Quartz.CGWarpMouseCursorPosition(_cg_point(x, y))
    # Reassociate so the cursor responds to subsequent movements without delay.
    Quartz.CGAssociateMouseAndMouseCursorPosition(True)


def mouse_down(x: int, y: int) -> None:
    if not _AVAILABLE:
        return
    evt = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventLeftMouseDown,
        _cg_point(x, y),
        Quartz.kCGMouseButtonLeft,
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def mouse_up(x: int, y: int) -> None:
    if not _AVAILABLE:
        return
    evt = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventLeftMouseUp,
        _cg_point(x, y),
        Quartz.kCGMouseButtonLeft,
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def mouse_drag(x: int, y: int) -> None:
    """Post a drag event. Call while the mouse button is held down to move with drag semantics."""
    if not _AVAILABLE:
        return
    evt = Quartz.CGEventCreateMouseEvent(
        None,
        Quartz.kCGEventLeftMouseDragged,
        _cg_point(x, y),
        Quartz.kCGMouseButtonLeft,
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def new_finder_window(bounds: tuple[int, int, int, int] | None = None) -> None:
    """Open a new Finder window.

    If ``bounds`` is provided as ``(x, y, x2, y2)`` in screen pixels (top-left
    origin), the new window is sized and positioned to match.
    """
    if not _MAC:
        return
    if bounds is None:
        script = 'tell application "Finder" to make new Finder window'
    else:
        x, y, x2, y2 = bounds
        script = (
            'tell application "Finder"\n'
            "    set w to make new Finder window\n"
            f"    set bounds of w to {{{x}, {y}, {x2}, {y2}}}\n"
            "end tell"
        )
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError):
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            timeout=2.0,
        )


def launch_app(name: str) -> None:
    """Launch/focus a macOS app by name (e.g. 'Safari', 'Finder')."""
    if not _MAC:
        return
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError):
        subprocess.run(
            ["open", "-a", name],
            check=False,
            capture_output=True,
            timeout=2.0,
        )


def close_frontmost_window() -> None:
    """Close the frontmost window of the active application (Cmd+W)."""
    if not _MAC:
        return
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError):
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "w" using {command down}',
            ],
            check=False,
            capture_output=True,
            timeout=2.0,
        )
