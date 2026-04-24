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


def new_app_window(name: str, bounds: tuple[int, int, int, int] | None = None) -> None:
    """Open a new window of ``name`` and optionally position it.

    Finder gets AppleScript's native ``make new Finder window``. Other apps
    get activated then fired Cmd+N via System Events — works for Safari,
    Chrome, Notes, Terminal, TextEdit, and most apps with a standard "New
    Window" menu item. Bounds are applied afterward via the AX API so it
    works regardless of the app.
    """
    if not _MAC:
        return
    if name == "Finder":
        new_finder_window(bounds=bounds)
        return
    # Activate the app and fire Cmd+N.
    script = (
        f'tell application "{name}" to activate\n'
        "delay 0.15\n"
        'tell application "System Events" to keystroke "n" using {command down}\n'
    )
    if bounds is not None:
        x, y, x2, y2 = bounds
        script += (
            "delay 0.3\n"
            'tell application "System Events"\n'
            f'    tell process "{name}"\n'
            f"        try\n"
            f"            set position of front window to {{{x}, {y}}}\n"
            f"            set size of front window to {{{x2 - x}, {y2 - y}}}\n"
            f"        end try\n"
            "    end tell\n"
            "end tell\n"
        )
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError):
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            timeout=3.0,
        )


def scroll(dy_pixels: float) -> None:
    """Post a vertical scroll-wheel event. Positive dy = scroll up."""
    if not _AVAILABLE:
        return
    # kCGScrollEventUnitPixel = 0, unit line = 1. Use pixels for smoother feel.
    evt = Quartz.CGEventCreateScrollWheelEvent(None, 0, 1, int(dy_pixels))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)


def screenshot(mode: str = "screen") -> str | None:
    """Run ``screencapture`` in the requested mode.

    mode:
      - "screen": silent full-display capture → saved immediately.
      - "window": user clicks a window to capture.
      - "selection": user drags a rectangle to capture.

    Returns the saved file path on "screen", None on interactive modes
    (screencapture handles the filename prompt itself if no path given).
    """
    if not _MAC:
        return None
    import datetime
    import os

    home = os.path.expanduser("~")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = f"{home}/Desktop/handspring_{ts}.png"
    flags = {"screen": ["-x"], "window": ["-w", "-x"], "selection": ["-s", "-x"]}.get(mode, ["-x"])
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError):
        subprocess.Popen(
            ["screencapture", *flags, path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return path


def _run_script(script: str, timeout: float = 2.0) -> None:
    if not _MAC:
        return
    with contextlib.suppress(subprocess.TimeoutExpired, FileNotFoundError):
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            timeout=timeout,
        )


def _keystroke(key: str, modifiers: list[str]) -> None:
    mods = ", ".join(modifiers)
    _run_script(f'tell application "System Events" to keystroke "{key}" using {{{mods}}}')


def _key_code(code: int, modifiers: list[str]) -> None:
    mods = ", ".join(modifiers)
    suffix = f" using {{{mods}}}" if modifiers else ""
    _run_script(f'tell application "System Events" to key code {code}{suffix}')


def close_frontmost_window() -> None:
    """Close the frontmost window of the active application (Cmd+W)."""
    _keystroke("w", ["command down"])


def minimize_front_window() -> None:
    """Minimize the frontmost window (Cmd+M)."""
    _keystroke("m", ["command down"])


def fullscreen_front_window() -> None:
    """Toggle native macOS fullscreen on the frontmost window (Ctrl+Cmd+F)."""
    _keystroke("f", ["control down", "command down"])


def mission_control() -> None:
    """Open Mission Control (Ctrl+Up). Shows all windows across the current Space."""
    _key_code(126, ["control down"])  # 126 = up arrow


def show_desktop() -> None:
    """Toggle Show Desktop (F11 keycode 103 by default on macOS)."""
    if not _MAC:
        return
    _run_script('tell application "System Events" to key code 103')


def switch_desktop(direction: str) -> None:
    """Page left/right between macOS Spaces (Ctrl+←/→)."""
    code = 123 if direction == "left" else 124  # 123=left, 124=right
    _key_code(code, ["control down"])


def visible_frame() -> tuple[int, int, int, int]:
    """Usable area of the main screen in top-origin pixels (menubar + dock excluded).

    Returns ``(x, y, width, height)``. AppleScript / AX windows use this origin.
    """
    if not _AVAILABLE:
        return (0, 25, 1440, 820)
    screen = NSScreen.mainScreen()
    full = screen.frame()
    vis = screen.visibleFrame()
    x = int(vis.origin.x)
    # Cocoa origin is bottom-left; AppleScript uses top-left. Flip the y.
    top = int(full.size.height - (vis.origin.y + vis.size.height))
    return (x, top, int(vis.size.width), int(vis.size.height))


def tile_front_window(position: str) -> None:
    """Resize + move the frontmost window within the visible screen.

    position: "left" | "right" | "full" | "center"
    """
    if not _MAC:
        return
    x, y, w, h = visible_frame()
    if position == "left":
        px, py, sw, sh = x, y, w // 2, h
    elif position == "right":
        px, py, sw, sh = x + w // 2, y, w - w // 2, h
    elif position == "full":
        px, py, sw, sh = x, y, w, h
    elif position == "center":
        # 70% × 75% of usable, centered.
        sw = int(w * 0.7)
        sh = int(h * 0.75)
        px = x + (w - sw) // 2
        py = y + (h - sh) // 2
    else:
        return
    _run_script(
        'tell application "System Events"\n'
        "    set frontApp to first application process whose frontmost is true\n"
        "    tell frontApp\n"
        "        try\n"
        f"            set position of front window to {{{px}, {py}}}\n"
        f"            set size of front window to {{{sw}, {sh}}}\n"
        "        end try\n"
        "    end tell\n"
        "end tell"
    )
