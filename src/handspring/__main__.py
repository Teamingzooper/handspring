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
import webbrowser
from types import FrameType

import cv2
import numpy as np
from numpy.typing import NDArray

from handspring import __version__, os_control, overlay
from handspring.config import ConfigStore, start_watcher
from handspring.desktop_controller import DesktopController
from handspring.osc_out import OscEmitter
from handspring.preview import Preview
from handspring.settings_server import SettingsServer
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
        "--settings-port",
        type=int,
        default=8766,
        help="settings UI port (default: 8766)",
    )
    p.add_argument(
        "--no-settings",
        action="store_true",
        help="disable the settings web UI (radial → More → Settings)",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="override path to config.toml (default: ~/.config/handspring/config.toml)",
    )
    p.add_argument(
        "--no-overlay",
        dest="overlay",
        action="store_false",
        help="disable the always-on-top left-hand cursor + radial overlay",
    )
    p.add_argument(
        "--fps-log-interval",
        type=float,
        default=0.5,
        help="print FPS + state to terminal every N seconds (default: 0.5)",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.set_defaults(mirror=True, os_control=True, overlay=True)
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

    from pathlib import Path

    config_path = Path(args.config).expanduser() if args.config else None
    store = ConfigStore(path=config_path)
    watcher = start_watcher(store)

    settings_server: SettingsServer | None = None
    if not args.no_settings:
        settings_server = SettingsServer(store, port=args.settings_port)
        settings_server.start()

    shutdown = _Shutdown()

    def _open_settings() -> None:
        if settings_server is not None:
            webbrowser.open(settings_server.url)

    def _reload_config() -> None:
        store.reload()

    def _quit() -> None:
        shutdown.requested = True

    tracker = Tracker(
        TrackerConfig(
            max_hands=args.hands,
            track_face=not args.no_face,
            track_pose=not args.no_pose,
        )
    )
    emitter = OscEmitter(host=args.host, port=args.port)
    desktop = (
        DesktopController(
            mirrored=args.mirror,
            store=store,
            on_open_settings=_open_settings,
            on_reload_config=_reload_config,
            on_quit=_quit,
        )
        if args.os_control
        else None
    )

    preview = Preview(mirror=args.mirror, show_window=not args.no_preview)

    latest: LatestFrame | None = None
    web: WebServer | None = None
    if not args.no_web:
        latest = LatestFrame()
        web = WebServer(port=args.web_port, latest=latest)
        web.start()

    overlay_inst: overlay.Overlay | None = None
    if args.overlay and overlay.available():
        overlay_inst = overlay.Overlay()
        if not overlay_inst.start():
            overlay_inst = None

    print(f"handspring {__version__}", flush=True)
    print(f"camera: {args.camera}", flush=True)
    print(f"OSC:    {args.host}:{args.port}", flush=True)
    if web is not None:
        print(f"web:    http://127.0.0.1:{args.web_port}/  (Plash-ready)", flush=True)
    if settings_server is not None:
        print(f"settings: {settings_server.url}  (radial → More → Settings)", flush=True)
    print(f"config: {store.path}", flush=True)
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
    if overlay_inst is not None:
        print("Overlay:    ON  (left-hand dot + radial float above all apps)", flush=True)
    elif args.overlay:
        print("Overlay:    requested but unavailable", flush=True)
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

            # Push state into the native overlay (if enabled) and pump
            # AppKit events so the window redraws.
            if overlay_inst is not None and desktop is not None:
                rs = desktop.radial_state()
                radial_payload: (
                    tuple[
                        tuple[int, int],
                        int | None,
                        int | None,
                        float,
                        tuple[tuple[str, tuple[str, ...]], ...],
                    ]
                    | None
                ) = None
                if rs is not None:
                    origin_raw, _cur_raw, hovered_root, hovered_sub, progress = rs
                    screen_origin = _cam_to_screen(
                        origin_raw[0], origin_raw[1], desktop, mirrored=args.mirror
                    )
                    radial_payload = (
                        screen_origin,
                        hovered_root,
                        hovered_sub,
                        progress,
                        desktop.root_items(),
                    )
                overlay.set_state(
                    cursor=desktop.left_cursor_screen(),
                    radial=radial_payload,
                    selected_app=desktop.selected_app(),
                    mode=desktop.mode(),
                    pending_rect=desktop.pending_create_screen_bounds(),
                    committed_rect=desktop.post_spawn_screen_bounds(),
                )
                overlay_inst.redraw()
                overlay_inst.pump()

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
        if settings_server is not None:
            settings_server.stop()
        watcher.stop()
        if overlay_inst is not None:
            overlay_inst.stop()
        cv2.destroyAllWindows()
    print()
    return 0


def _cam_to_screen(
    cam_x: float,
    cam_y: float,
    desktop: DesktopController,
    *,
    mirrored: bool,
) -> tuple[int, int]:
    """Apply the same mirror + inset mapping the cursor uses."""
    inset = desktop.store.get().cursor.inset
    nx = 1.0 - cam_x if mirrored else cam_x
    ny = cam_y
    span = max(1e-6, 1.0 - 2 * inset)
    mx = (nx - inset) / span
    my = (ny - inset) / span
    sw = desktop._screen_w  # noqa: SLF001
    sh = desktop._screen_h  # noqa: SLF001
    sx = max(0, min(sw - 1, int(mx * sw)))
    sy = max(0, min(sh - 1, int(my * sh)))
    return sx, sy


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

    # Radial menu is rendered by the native overlay — not duplicated here.
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
