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
    # Radial tree: (origin, hovered_root, hovered_sub, progress, root_items)
    # root_items is a sequence of (name, subs). None = not visible.
    "radial": None,
    "selected_app": "Finder",
    "mode": "create",  # "create" | "scroll" | "none"
    "pending_rect": None,
    "committed_rect": None,
}


def set_state(
    *,
    cursor: tuple[int, int] | None,
    radial: tuple[
        tuple[int, int],
        int | None,
        int | None,
        float,
        tuple[tuple[str, tuple[str, ...]], ...],
    ]
    | None,
    selected_app: str,
    mode: str,
    pending_rect: tuple[int, int, int, int] | None = None,
    committed_rect: tuple[int, int, int, int] | None = None,
) -> None:
    _state["cursor"] = cursor
    _state["radial"] = radial
    _state["selected_app"] = selected_app
    _state["mode"] = mode
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

            # Left-hand cursor: color + label depend on mode.
            mode = _state["mode"]
            if mode == "scroll":
                dot_rgb = (0.30, 0.60, 1.0)
                dot_label = "SCROLL"
            elif mode == "none":
                dot_rgb = (0.40, 0.40, 0.40)
                dot_label = "NONE"
            else:  # create
                dot_rgb = (0.70, 0.70, 0.70)
                dot_label = selected_app

            if cursor is not None:
                cx, cy = cursor
                Quartz.CGContextSetRGBFillColor(ctx, dot_rgb[0], dot_rgb[1], dot_rgb[2], 0.75)
                Quartz.CGContextFillEllipseInRect(ctx, Quartz.CGRectMake(cx - 10, cy - 10, 20, 20))
                _draw_label(
                    dot_label,
                    cx,
                    cy + 24,
                    AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                        min(1.0, dot_rgb[0] + 0.1),
                        min(1.0, dot_rgb[1] + 0.1),
                        min(1.0, dot_rgb[2] + 0.1),
                        0.95,
                    ),
                    False,
                )

            # Radial tree.
            if radial is not None:
                (ox, oy), hovered_root, hovered_sub, progress, root_items = radial
                _draw_radial_tree(ctx, ox, oy, hovered_root, hovered_sub, progress, root_items)

    def _pie_wedge_path(
        ctx: Any,
        ox: float,
        oy: float,
        r_inner: float,
        r_outer: float,
        start_angle: float,
        end_angle: float,
    ) -> None:
        """Build an annular wedge path (ready to fill or stroke).

        Angles are in radians, screen-convention (0 = +x, π/2 = +y = down).
        For a solid pie slice (no inner cutout), pass r_inner=0.
        """
        Quartz.CGContextBeginPath(ctx)
        if r_inner <= 0.5:
            Quartz.CGContextMoveToPoint(ctx, ox, oy)
            Quartz.CGContextAddArc(ctx, ox, oy, r_outer, start_angle, end_angle, False)
        else:
            # Annular wedge: inner arc → outer arc (reverse).
            inner_start_x = ox + r_inner * math.cos(start_angle)
            inner_start_y = oy + r_inner * math.sin(start_angle)
            Quartz.CGContextMoveToPoint(ctx, inner_start_x, inner_start_y)
            Quartz.CGContextAddArc(ctx, ox, oy, r_inner, start_angle, end_angle, False)
            Quartz.CGContextAddArc(ctx, ox, oy, r_outer, end_angle, start_angle, True)
        Quartz.CGContextClosePath(ctx)

    def _draw_radial_tree(
        ctx: Any,
        ox: float,
        oy: float,
        hovered_root: int | None,
        hovered_sub: int | None,
        progress: float,
        root_items: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> None:
        r_inner = 40.0  # center dead-zone visual
        r_root = 220.0  # outer radius of root ring

        # During the hold countdown, show ONLY a thin filling arc — no slices.
        if progress < 1.0:
            Quartz.CGContextSetRGBStrokeColor(ctx, 0.75, 0.75, 0.75, 0.9)
            Quartz.CGContextSetLineWidth(ctx, 8.0)
            end_radians = -math.pi / 2 + 2 * math.pi * progress
            Quartz.CGContextBeginPath(ctx)
            Quartz.CGContextAddArc(ctx, ox, oy, 60.0, -math.pi / 2, end_radians, False)
            Quartz.CGContextStrokePath(ctx)
            return

        # --- Root ring: filled annular wedges ---
        n = len(root_items)
        slice_size = 2 * math.pi / n
        for i, (name, _) in enumerate(root_items):
            # Slice i occupies the angular range centered on its cardinal direction.
            start = -math.pi / 2 + (i - 0.5) * slice_size
            end = start + slice_size
            highlighted = i == hovered_root
            # Wedge fill.
            _pie_wedge_path(ctx, ox, oy, r_inner, r_root, start, end)
            if highlighted:
                Quartz.CGContextSetRGBFillColor(ctx, 0.30, 0.30, 0.30, 0.85)
            else:
                Quartz.CGContextSetRGBFillColor(ctx, 0.10, 0.10, 0.10, 0.70)
            Quartz.CGContextFillPath(ctx)
            # Wedge outline.
            _pie_wedge_path(ctx, ox, oy, r_inner, r_root, start, end)
            if highlighted:
                Quartz.CGContextSetRGBStrokeColor(ctx, 0.90, 0.90, 0.90, 0.95)
                Quartz.CGContextSetLineWidth(ctx, 1.5)
            else:
                Quartz.CGContextSetRGBStrokeColor(ctx, 0.45, 0.45, 0.45, 0.80)
                Quartz.CGContextSetLineWidth(ctx, 1.0)
            Quartz.CGContextStrokePath(ctx)
            # Label at the slice center.
            center_angle = -math.pi / 2 + i * slice_size
            lr = (r_inner + r_root) * 0.5
            lx = ox + lr * math.cos(center_angle)
            ly = oy + lr * math.sin(center_angle)
            color = (
                AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 1.0)
                if highlighted
                else AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.80, 0.80, 0.80, 1.0)
            )
            _draw_label(name, lx, ly, color, highlighted)

        # --- Horizontal sub-chip row at the hovered slice's tip ---
        # Chips are always laid out in a horizontal line with spacing. The
        # row auto-picks a side (left/right) and slides back if it would
        # clip the screen edge. Selection math lives in desktop_controller —
        # the overlay just draws the positions it reports.
        if hovered_root is None:
            return
        _name, subs = root_items[hovered_root]
        if not subs:
            return
        _draw_sub_chips_row(ctx, ox, oy, hovered_root, hovered_sub, subs, len(root_items), r_root)

    def _draw_sub_chips_row(
        ctx: Any,
        ox: float,
        oy: float,
        hovered_root: int,
        hovered_sub: int | None,
        subs: tuple[str, ...],
        n_roots: int,
        r_root: float,
    ) -> None:
        from handspring.desktop_controller import (
            SUB_CHIP_H,
            SUB_CHIP_W,
            compute_sub_layout,
        )

        screen = AppKit.NSScreen.mainScreen().frame()
        screen_w = int(screen.size.width)
        screen_h = int(screen.size.height)

        # The overlay receives origin_screen already mirrored by __main__'s
        # _cam_to_screen. Pass mirrored=False to compute_sub_layout so it
        # doesn't double-flip.
        centers, tip, _dir = compute_sub_layout(
            (int(ox), int(oy)),
            hovered_root,
            n_roots,
            len(subs),
            screen_w,
            screen_h,
            mirrored=False,
        )

        # Thin connector line from slice tip to first chip.
        if centers:
            Quartz.CGContextSetRGBStrokeColor(ctx, 0.60, 0.60, 0.60, 0.55)
            Quartz.CGContextSetLineWidth(ctx, 1.0)
            Quartz.CGContextBeginPath(ctx)
            Quartz.CGContextMoveToPoint(ctx, tip[0], tip[1])
            Quartz.CGContextAddLineToPoint(ctx, centers[0][0], centers[0][1])
            Quartz.CGContextStrokePath(ctx)

        sub_armed = hovered_sub is not None
        corner = 14.0
        for j, sub_name in enumerate(subs):
            cx, cy = centers[j]
            highlighted = j == hovered_sub
            rect = Quartz.CGRectMake(
                cx - SUB_CHIP_W / 2, cy - SUB_CHIP_H / 2, SUB_CHIP_W, SUB_CHIP_H
            )
            _rounded_rect_path(ctx, rect, corner)
            if highlighted:
                Quartz.CGContextSetRGBFillColor(ctx, 0.28, 0.50, 0.10, 0.92)
            elif sub_armed:
                Quartz.CGContextSetRGBFillColor(ctx, 0.12, 0.12, 0.12, 0.85)
            else:
                Quartz.CGContextSetRGBFillColor(ctx, 0.10, 0.10, 0.10, 0.60)
            Quartz.CGContextFillPath(ctx)

            _rounded_rect_path(ctx, rect, corner)
            if highlighted:
                Quartz.CGContextSetRGBStrokeColor(ctx, 0.53, 1.0, 0.0, 1.0)
                Quartz.CGContextSetLineWidth(ctx, 2.0)
            else:
                Quartz.CGContextSetRGBStrokeColor(
                    ctx, 0.45, 0.45, 0.45, 0.90 if sub_armed else 0.55
                )
                Quartz.CGContextSetLineWidth(ctx, 1.0)
            Quartz.CGContextStrokePath(ctx)

            color = (
                AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.60, 1.0, 0.20, 1.0)
                if highlighted
                else AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.92, 0.92, 0.92, 0.95 if sub_armed else 0.70
                )
            )
            _draw_label(sub_name, cx, cy, color, highlighted)

    def _rounded_rect_path(ctx: Any, rect: Any, radius: float) -> None:
        x = rect.origin.x
        y = rect.origin.y
        w = rect.size.width
        h = rect.size.height
        r = min(radius, w / 2, h / 2)
        Quartz.CGContextBeginPath(ctx)
        Quartz.CGContextMoveToPoint(ctx, x + r, y)
        Quartz.CGContextAddLineToPoint(ctx, x + w - r, y)
        Quartz.CGContextAddArcToPoint(ctx, x + w, y, x + w, y + r, r)
        Quartz.CGContextAddLineToPoint(ctx, x + w, y + h - r)
        Quartz.CGContextAddArcToPoint(ctx, x + w, y + h, x + w - r, y + h, r)
        Quartz.CGContextAddLineToPoint(ctx, x + r, y + h)
        Quartz.CGContextAddArcToPoint(ctx, x, y + h, x, y + h - r, r)
        Quartz.CGContextAddLineToPoint(ctx, x, y + r)
        Quartz.CGContextAddArcToPoint(ctx, x, y, x + r, y, r)
        Quartz.CGContextClosePath(ctx)

    def _draw_label(text: str, cx: float, cy: float, color: Any, bold: bool) -> None:
        attrs = {
            AppKit.NSFontAttributeName: AppKit.NSFont.boldSystemFontOfSize_(18)
            if bold
            else AppKit.NSFont.systemFontOfSize_(16),
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
