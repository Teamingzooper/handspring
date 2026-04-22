"""Fullscreen transparent click-through overlay (macOS only).

Shows a green dot at the left hand's screen position + a radial app picker
when the radial is active. Drawn above all other windows so the user can
see where their left hand is regardless of what's focused.
"""

from __future__ import annotations

import math
import sys
from typing import Any

_MAC = sys.platform == "darwin"
_AVAILABLE = False

if _MAC:
    try:
        import AppKit
        import Foundation
        import Quartz

        _AVAILABLE = True
    except ImportError:
        _AVAILABLE = False


def available() -> bool:
    return _AVAILABLE


# State set from outside: a dict carrying everything the view needs to draw.
# Using a dict avoids Objective-C typing complications with pyobjc properties.
_state: dict[str, Any] = {
    "cursor": None,  # (screen_x, screen_y) or None
    "radial": None,  # (origin_screen_xy, selected_idx, progress, apps) or None
    "selected_app": "Finder",  # small label under the dot
    "pending_rect": None,  # (x1, y1, x2, y2) while pinching to create
    "committed_rect": None,  # (x1, y1, x2, y2) after release, until window appears
}


def set_state(
    *,
    cursor: tuple[int, int] | None,
    radial: tuple[tuple[int, int], int | None, float, tuple[str, ...]] | None,
    selected_app: str,
    pending_rect: tuple[int, int, int, int] | None = None,
    committed_rect: tuple[int, int, int, int] | None = None,
) -> None:
    _state["cursor"] = cursor
    _state["radial"] = radial
    _state["selected_app"] = selected_app
    _state["pending_rect"] = pending_rect
    _state["committed_rect"] = committed_rect


if _AVAILABLE:

    class _OverlayView(AppKit.NSView):  # type: ignore[misc]
        def isFlipped(self) -> bool:  # noqa: N802
            # Use top-left origin so screen coords map naturally.
            return True

        def drawRect_(self, _rect: Any) -> None:  # noqa: N802
            ctx = AppKit.NSGraphicsContext.currentContext().CGContext()
            cursor = _state["cursor"]
            radial = _state["radial"]
            selected_app = _state["selected_app"]
            pending_rect = _state["pending_rect"]
            committed_rect = _state["committed_rect"]

            # Post-commit ghost rect (lighter grey, underneath) — shown until
            # the real macOS window appears.
            if committed_rect is not None:
                x1, y1, x2, y2 = committed_rect
                rect = Quartz.CGRectMake(x1, y1, x2 - x1, y2 - y1)
                Quartz.CGContextSetRGBFillColor(ctx, 0.85, 0.85, 0.85, 0.14)
                Quartz.CGContextFillRect(ctx, rect)
                Quartz.CGContextSetRGBStrokeColor(ctx, 0.85, 0.85, 0.85, 0.55)
                Quartz.CGContextSetLineWidth(ctx, 1.5)
                Quartz.CGContextStrokeRect(ctx, rect)

            # Pending create rect (darker grey, while pinching both hands).
            if pending_rect is not None:
                x1, y1, x2, y2 = pending_rect
                rect = Quartz.CGRectMake(x1, y1, x2 - x1, y2 - y1)
                Quartz.CGContextSetRGBFillColor(ctx, 0.70, 0.70, 0.70, 0.22)
                Quartz.CGContextFillRect(ctx, rect)
                Quartz.CGContextSetRGBStrokeColor(ctx, 0.70, 0.70, 0.70, 0.85)
                Quartz.CGContextSetLineWidth(ctx, 2.0)
                Quartz.CGContextStrokeRect(ctx, rect)

            # Left-hand cursor: plain grey circle.
            if cursor is not None:
                cx, cy = cursor
                Quartz.CGContextSetRGBFillColor(ctx, 0.70, 0.70, 0.70, 0.75)
                Quartz.CGContextFillEllipseInRect(ctx, Quartz.CGRectMake(cx - 10, cy - 10, 20, 20))
                # Selected-app label right under the dot.
                _draw_label(
                    selected_app,
                    cx,
                    cy + 24,
                    AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.80, 0.80, 0.80, 0.95),
                    False,
                )

            # Radial menu (in screen coords).
            if radial is not None:
                (ox, oy), selected, progress, apps = radial
                r_outer = 140.0
                r_inner = 40.0
                # Background disk.
                Quartz.CGContextSetRGBFillColor(ctx, 0.08, 0.08, 0.08, 0.78)
                Quartz.CGContextFillEllipseInRect(
                    ctx,
                    Quartz.CGRectMake(ox - r_outer, oy - r_outer, r_outer * 2, r_outer * 2),
                )
                # Outer ring.
                Quartz.CGContextSetRGBStrokeColor(ctx, 0.39, 0.78, 1.0, 1.0)
                Quartz.CGContextSetLineWidth(ctx, 2.0)
                Quartz.CGContextStrokeEllipseInRect(
                    ctx,
                    Quartz.CGRectMake(ox - r_outer, oy - r_outer, r_outer * 2, r_outer * 2),
                )
                # Inner ring.
                Quartz.CGContextStrokeEllipseInRect(
                    ctx,
                    Quartz.CGRectMake(ox - r_inner, oy - r_inner, r_inner * 2, r_inner * 2),
                )
                # Slice dividers + labels.
                n = len(apps)
                slice_size = 2 * math.pi / n
                for i, name in enumerate(apps):
                    center_cw = i * slice_size  # 0 = up, clockwise
                    screen_angle = -math.pi / 2 + center_cw  # atan2-style
                    label_r = r_outer * 0.65
                    lx = ox + label_r * math.cos(screen_angle)
                    ly = oy + label_r * math.sin(screen_angle)
                    # Divider line.
                    div_angle = -math.pi / 2 + (i - 0.5) * slice_size
                    dx = ox + r_outer * math.cos(div_angle)
                    dy = oy + r_outer * math.sin(div_angle)
                    Quartz.CGContextSetRGBStrokeColor(ctx, 0.25, 0.25, 0.25, 0.9)
                    Quartz.CGContextSetLineWidth(ctx, 1.0)
                    Quartz.CGContextBeginPath(ctx)
                    Quartz.CGContextMoveToPoint(ctx, ox, oy)
                    Quartz.CGContextAddLineToPoint(ctx, dx, dy)
                    Quartz.CGContextStrokePath(ctx)
                    # Label.
                    highlighted = (selected == i) and progress >= 1.0
                    color = (
                        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.53, 1.0, 0.0, 1.0)
                        if highlighted
                        else AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                            0.88, 0.88, 0.88, 1.0
                        )
                    )
                    _draw_label(name, lx, ly, color, highlighted)
                # Hold countdown arc on inner ring.
                if progress < 1.0:
                    end_radians = -math.pi / 2 + 2 * math.pi * progress
                    Quartz.CGContextSetRGBStrokeColor(ctx, 1.0, 0.86, 0.23, 1.0)
                    Quartz.CGContextSetLineWidth(ctx, 4.0)
                    Quartz.CGContextBeginPath(ctx)
                    Quartz.CGContextAddArc(
                        ctx, ox, oy, r_inner + 8, -math.pi / 2, end_radians, False
                    )
                    Quartz.CGContextStrokePath(ctx)

    def _draw_label(text: str, cx: float, cy: float, color: Any, bold: bool) -> None:
        attrs = {
            AppKit.NSFontAttributeName: AppKit.NSFont.boldSystemFontOfSize_(13)
            if bold
            else AppKit.NSFont.systemFontOfSize_(13),
            AppKit.NSForegroundColorAttributeName: color,
        }
        ns = Foundation.NSString.stringWithString_(text)
        size = ns.sizeWithAttributes_(attrs)
        # Shadow behind label for readability against arbitrary backgrounds.
        shadow_attrs = {
            AppKit.NSFontAttributeName: attrs[AppKit.NSFontAttributeName],
            AppKit.NSForegroundColorAttributeName: AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0, 0, 0, 0.9
            ),
        }
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ns.drawAtPoint_withAttributes_(
                    Foundation.NSMakePoint(cx - size.width / 2 + dx, cy - size.height / 2 + dy),
                    shadow_attrs,
                )
        ns.drawAtPoint_withAttributes_(
            Foundation.NSMakePoint(cx - size.width / 2, cy - size.height / 2), attrs
        )


class Overlay:
    """Manages the overlay window lifetime + per-frame pump."""

    def __init__(self) -> None:
        self._window: Any = None
        self._view: Any = None
        self._app: Any = None

    def start(self) -> bool:
        if not _AVAILABLE:
            return False
        self._app = AppKit.NSApplication.sharedApplication()
        # Accessory = app has no Dock icon, doesn't steal focus.
        self._app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

        screen = AppKit.NSScreen.mainScreen()
        frame = screen.frame()

        self._window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self._window.setOpaque_(False)
        self._window.setBackgroundColor_(AppKit.NSColor.clearColor())
        # Float above normal windows. NSPopUpMenuWindowLevel = 101.
        self._window.setLevel_(AppKit.NSPopUpMenuWindowLevel)
        self._window.setIgnoresMouseEvents_(True)
        self._window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorIgnoresCycle
        )
        self._window.setHasShadow_(False)

        self._view = _OverlayView.alloc().initWithFrame_(frame)
        self._window.setContentView_(self._view)
        self._window.orderFrontRegardless()
        return True

    def redraw(self) -> None:
        if self._view is not None:
            self._view.setNeedsDisplay_(True)

    def pump(self) -> None:
        """Drain any pending AppKit events so the window renders."""
        if self._app is None:
            return
        while True:
            event = self._app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                AppKit.NSEventMaskAny,
                Foundation.NSDate.dateWithTimeIntervalSinceNow_(0),
                AppKit.NSDefaultRunLoopMode,
                True,
            )
            if event is None:
                break
            self._app.sendEvent_(event)

    def stop(self) -> None:
        if self._window is not None:
            self._window.orderOut_(None)
            self._window = None
            self._view = None
