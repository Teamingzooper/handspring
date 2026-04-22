"""`python -m handspring` entry point: camera → tracker → macOS desktop control + web stream.

Gestures (when enabled):
- Right index fingertip → system cursor position
- Right-hand pinch → left-mouse click (down on pinch start, up on release)
  (so pinching the title bar of a window and moving drags it naturally)
- Both-hand pinch close together, then pull apart → new Finder window
- Both-hand FIST held for 5 seconds → toggle failsafe (disables/enables gesture control)

Also starts a local MJPEG web server at http://localhost:8765/ that serves the
annotated camera feed — point Plash at this URL (or at the `web/` folder on
disk) to use it as your desktop background.

The synth half lives in the `handspring-synth` command (separate process).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from types import FrameType

import cv2
import numpy as np
from numpy.typing import NDArray

from handspring import __version__, os_control
from handspring.desktop_controller import DesktopController
from handspring.osc_out import OscEmitter
from handspring.preview import Preview
from handspring.tracker import Tracker, TrackerConfig
from handspring.types import FrameResult
from handspring.web_server import LatestFrame, WebServer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="handspring",
        description="Gesture → macOS desktop control + web stream.",
    )
    p.add_argument("--host", default="127.0.0.1", help="OSC receiver host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=9000, help="OSC receiver port (default: 9000)")
    p.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    p.add_argument("--no-preview", action="store_true", help="disable the OpenCV preview window")
    p.add_argument("--no-face", action="store_true", help="disable face tracking")
    p.add_argument("--no-pose", action="store_true", help="disable body/arm pose tracking")
    p.add_argument("--hands", type=int, choices=[0, 1, 2], default=2, help="max hands to track")
    p.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="disable preview mirror",
    )
    p.add_argument(
        "--no-os-control",
        dest="os_control",
        action="store_false",
        help="disable macOS cursor/click/Finder gestures (still renders web stream)",
    )
    p.add_argument("--web-port", type=int, default=8765, help="web stream port (default: 8765)")
    p.add_argument("--no-web", action="store_true", help="disable the MJPEG web server")
    p.add_argument(
        "--fps-log-interval",
        type=float,
        default=0.5,
        help="print FPS + state to terminal every N seconds (default: 0.5)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.set_defaults(mirror=True, os_control=True)
    return p.parse_args(argv)


class _Shutdown:
    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, _signum: int, _frame: FrameType | None) -> None:
        self.requested = True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"error: could not open camera {args.camera}", file=sys.stderr)
        return 2

    tracker = Tracker(
        TrackerConfig(
            max_hands=args.hands,
            track_face=not args.no_face,
            track_pose=not args.no_pose,
        )
    )
    emitter = OscEmitter(host=args.host, port=args.port)
    desktop = DesktopController(mirrored=args.mirror) if args.os_control else None

    preview = Preview(mirror=args.mirror, show_window=not args.no_preview)

    latest: LatestFrame | None = None
    web: WebServer | None = None
    if not args.no_web:
        latest = LatestFrame()
        web = WebServer(port=args.web_port, latest=latest)
        web.start()

    shutdown = _Shutdown()

    print(f"handspring {__version__}", flush=True)
    print(f"camera: {args.camera}", flush=True)
    print(f"OSC:    {args.host}:{args.port}", flush=True)
    if web is not None:
        print(f"web:    http://127.0.0.1:{args.web_port}/  (Plash-ready)", flush=True)
    if args.os_control:
        if os_control.available():
            sw, sh = os_control.screen_size()
            print(f"OS control: ON  (screen {sw}x{sh})", flush=True)
            print(
                "Note: requires Accessibility permission for the Python interpreter.",
                flush=True,
            )
        else:
            print("OS control: requested but unavailable on this platform", flush=True)
    else:
        print("OS control: OFF (--no-os-control)", flush=True)
    print("Failsafe: hold BOTH fists for 5s to toggle gesture control.", flush=True)
    print("Ctrl+C to quit.", flush=True)

    last_log = 0.0
    try:
        while not shutdown.requested:
            ok, bgr_raw = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            bgr: NDArray[np.uint8] = bgr_raw  # type: ignore[assignment]
            tracker_output = tracker.process(bgr)
            result = tracker_output.frame
            emitter.emit(result)
            now = time.monotonic()

            if desktop is not None:
                desktop.update(result, now=now)

            if not preview.render(
                bgr,
                tracker_output.hand_landmark_lists,
                tracker_output.face_landmark_lists,
                tracker_output.pose_landmarks,
                result,
                f"{args.host}:{args.port}",
                None,  # no synth snapshot
                None,  # no synth hint
                "jarvis",
                None,  # no internal jarvis windows in desktop mode
            ):
                break

            # Publish the annotated frame to the web server.
            if latest is not None:
                display = preview.last_display()
                if display is not None:
                    # Overlay failsafe progress + disabled badge onto the web frame.
                    if desktop is not None:
                        _overlay_status(display, desktop, mirrored=args.mirror)
                    ok, buf = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        latest.set(buf.tobytes())

            if now - last_log >= args.fps_log_interval:
                _print_status(result, desktop)
                last_log = now
    finally:
        cap.release()
        tracker.close()
        preview.close()
        if web is not None:
            web.stop()
        cv2.destroyAllWindows()
    print()
    return 0


def _overlay_status(
    display: NDArray[np.uint8], desktop: DesktopController, *, mirrored: bool
) -> None:
    """Draw DISABLED banner + failsafe progress + pending window + radial menu."""
    h, w = display.shape[:2]
    # Live "NEW WINDOW" ghost while the create gesture is armed.
    pending = desktop.pending_create_bounds()
    if pending is not None:
        px, py, pw, ph = pending
        x0 = int(px * w)
        y0 = int(py * h)
        x1 = int((px + pw) * w)
        y1 = int((py + ph) * h)
        ghost = display.copy()
        cv2.rectangle(ghost, (x0, y0), (x1, y1), (136, 255, 0), -1)
        cv2.addWeighted(ghost, 0.18, display, 0.82, 0, dst=display)
        cv2.rectangle(display, (x0, y0), (x1, y1), (136, 255, 0), 2)
        for color, thick in [((20, 20, 20), 3), ((136, 255, 0), 1)]:
            cv2.putText(
                display,
                "NEW WINDOW",
                (x0 + 10, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thick,
                cv2.LINE_AA,
            )

    # Radial app launcher (left hand pinch-and-hold).
    rs = desktop.radial_state()
    if rs is not None:
        _draw_radial(display, rs, desktop.radial_apps(), mirrored=mirrored)
    if not desktop.enabled():
        # Red "DISABLED" pill at top-center.
        text = "GESTURES DISABLED (hold both fists 5s to re-enable)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
        cx = (w - tw) // 2
        cy = 80
        cv2.rectangle(display, (cx - 14, cy - 28), (cx + tw + 14, cy + 10), (20, 20, 20), -1)
        cv2.rectangle(display, (cx - 14, cy - 28), (cx + tw + 14, cy + 10), (40, 40, 230), 2)
        cv2.putText(display, text, (cx, cy), font, scale, (40, 40, 230), thick, cv2.LINE_AA)
    progress = desktop.failsafe_progress()
    if progress > 0.0:
        # Growing arc in the top-right as a countdown indicator.
        cx_r = w - 60
        cy_r = 60
        radius = 40
        cv2.circle(display, (cx_r, cy_r), radius, (40, 40, 40), 6, cv2.LINE_AA)
        end_angle = int(360 * progress)
        cv2.ellipse(
            display,
            (cx_r, cy_r),
            (radius, radius),
            -90,
            0,
            end_angle,
            (255, 220, 60),
            6,
            cv2.LINE_AA,
        )
        label = "TOGGLING..." if progress > 0.1 else ""
        if label:
            cv2.putText(
                display,
                label,
                (cx_r - 60, cy_r + radius + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                label,
                (cx_r - 60, cy_r + radius + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 220, 60),
                1,
                cv2.LINE_AA,
            )


def _draw_radial(
    display: NDArray[np.uint8],
    rs: tuple[tuple[float, float], tuple[float, float], int | None, float],
    apps: tuple[str, ...],
    *,
    mirrored: bool,
) -> None:
    """Draw a radial slice menu centered at the pinch origin."""
    import math

    h, w = display.shape[:2]
    origin_raw, _cur_raw, selected, progress = rs
    ox_n = 1.0 - origin_raw[0] if mirrored else origin_raw[0]
    oy_n = origin_raw[1]
    cx = int(ox_n * w)
    cy = int(oy_n * h)
    # Keep the wheel on-screen.
    r_outer = 110
    cx = max(r_outer + 10, min(w - r_outer - 10, cx))
    cy = max(r_outer + 10, min(h - r_outer - 10, cy))
    r_inner = 30

    # Background disk (translucent).
    overlay = display.copy()
    cv2.circle(overlay, (cx, cy), r_outer, (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, display, 0.35, 0, dst=display)
    cv2.circle(display, (cx, cy), r_outer, (100, 200, 255), 2)
    cv2.circle(display, (cx, cy), r_inner, (100, 200, 255), 1)

    n = len(apps)
    slice_size = 2 * math.pi / n
    for i, name in enumerate(apps):
        # Slice center angle: 0 = up, clockwise.
        center_cw = i * slice_size
        # Convert to screen angle (+x = right, +y = down): angle from +x CCW.
        # cw_from_up → screen_angle: screen_angle = -π/2 + cw_from_up.
        screen_angle = -math.pi / 2 + center_cw
        lx = int(cx + (r_outer * 0.65) * math.cos(screen_angle))
        ly = int(cy + (r_outer * 0.65) * math.sin(screen_angle))
        highlighted = (selected == i) and progress >= 1.0
        color = (136, 255, 0) if highlighted else (220, 220, 220)
        thick = 2 if highlighted else 1
        # Divider line.
        divider_angle = -math.pi / 2 + (i - 0.5) * slice_size
        dx = int(cx + r_outer * math.cos(divider_angle))
        dy = int(cy + r_outer * math.sin(divider_angle))
        cv2.line(display, (cx, cy), (dx, dy), (60, 60, 60), 1, cv2.LINE_AA)
        # App label.
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thick)
        cv2.putText(
            display,
            name,
            (lx - tw // 2, ly + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            name,
            (lx - tw // 2, ly + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thick,
            cv2.LINE_AA,
        )

    # Hold-progress arc on the inner ring (until active).
    if progress < 1.0:
        end_angle = int(360 * progress)
        cv2.ellipse(
            display,
            (cx, cy),
            (r_inner + 6, r_inner + 6),
            -90,
            0,
            end_angle,
            (255, 220, 60),
            3,
            cv2.LINE_AA,
        )

    # Center hint.
    hint = "RELEASE=cancel" if selected is None and progress >= 1.0 else ""
    if hint:
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(
            display,
            hint,
            (cx - tw // 2, cy + r_outer + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            hint,
            (cx - tw // 2, cy + r_outer + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def _print_status(result: FrameResult, desktop: DesktopController | None) -> None:
    left = result.left.gesture if result.left.present else "-"
    right = result.right.gesture if result.right.present else "-"
    state = ""
    if desktop is not None:
        state = "DISABLED" if not desktop.enabled() else "active"
    print(
        f"\rFPS {result.fps:5.1f}  L:{left:<10} R:{right:<10}  ctrl:{state:<9}",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
