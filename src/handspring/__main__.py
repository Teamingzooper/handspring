"""`python -m handspring` entry point: camera → tracker → OSC + preview."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from types import FrameType
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from handspring import __version__
from handspring.app_mode import AppMode, AppModeController
from handspring.jarvis import JarvisController
from handspring.osc_out import OscEmitter
from handspring.preview import Preview
from handspring.synth import Synth
from handspring.synth_params import SynthParams
from handspring.synth_ui import SynthController
from handspring.tracker import Tracker, TrackerConfig
from handspring.types import FrameResult


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="handspring",
        description="Live gesture tracker → OSC stream.",
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
        "--fps-log-interval",
        type=float,
        default=0.5,
        help="print FPS + state to terminal every N seconds (default: 0.5)",
    )
    p.add_argument("--no-synth", action="store_true", help="disable in-process synth audio output")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.set_defaults(mirror=True)
    return p.parse_args(argv)


class _Shutdown:
    """Tiny helper so Ctrl+C sets a flag instead of raising inside the main loop."""

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
    synth_params = SynthParams()
    synth_controller = SynthController(synth_params)

    last_user_volume = {"v": synth_params.snapshot().volume}

    def _on_app_mode_change(new_mode: AppMode) -> None:
        if new_mode == "jarvis":
            last_user_volume["v"] = synth_params.snapshot().volume
            synth_params.set_volume(0.0)
        else:
            synth_params.set_volume(last_user_volume["v"])

    app_mode_controller = AppModeController(on_change=_on_app_mode_change)
    jarvis = JarvisController()

    synth: Synth | None = None
    if not args.no_synth:
        try:
            synth = Synth(synth_params)
            synth.start()
        except Exception as e:  # noqa: BLE001
            print(
                f"warning: could not start synth ({e}); continuing without audio", file=sys.stderr
            )
            synth = None
    preview = Preview(mirror=args.mirror) if not args.no_preview else None
    shutdown = _Shutdown()

    print(f"handspring {__version__}", flush=True)
    print(f"camera: {args.camera}", flush=True)
    print(f"OSC:    {args.host}:{args.port}", flush=True)
    synth_status = "off" if args.no_synth or synth is None else "on"
    print(
        f"hands:  {args.hands}   face: {'off' if args.no_face else 'on'}   "
        f"pose: {'off' if args.no_pose else 'on'}   synth: {synth_status}",
        flush=True,
    )
    print("Ctrl+C to quit.", flush=True)

    last_log = 0.0
    try:
        while not shutdown.requested:
            ok, bgr_raw = cap.read()
            if not ok:
                # Momentary read failure — try again after a beat.
                time.sleep(0.01)
                continue
            bgr: NDArray[np.uint8] = bgr_raw  # type: ignore[assignment]
            result = tracker.process(bgr)
            emitter.emit(result)
            synth_controller.update(result)
            now = time.monotonic()
            mouth_open_val = (
                result.face.features.mouth_open
                if result.face.present and result.face.features is not None
                else 0.0
            )
            app_mode_controller.update(
                mouth_open=mouth_open_val,
                face_present=result.face.present,
                now=now,
            )
            mode = app_mode_controller.mode()

            if mode == "jarvis":
                jarvis.update(result, now=now)

            emitter.emit_app_mode(mode)
            if mode == "jarvis":
                emitter.emit_jarvis_events(
                    jarvis.pop_events(),
                    window_count=len(jarvis.manager.windows()),
                )

            if not args.no_synth:
                emitter.emit_synth(synth_params.snapshot())

            if preview is not None:
                hand_landmarks, face_landmarks, pose_landmarks = _extract_landmark_lists(
                    tracker, bgr
                )
                snap_for_preview = synth_params.snapshot() if not args.no_synth else None
                hint_for_preview = synth_controller.ui_hint() if not args.no_synth else None
                if not preview.render(
                    bgr,
                    hand_landmarks,
                    face_landmarks,
                    pose_landmarks,
                    result,
                    f"{args.host}:{args.port}",
                    snap_for_preview,
                    hint_for_preview,
                    mode,
                    jarvis,
                ):
                    break

            if now - last_log >= args.fps_log_interval:
                _print_status(result)
                last_log = now
    finally:
        cap.release()
        tracker.close()
        if synth is not None:
            synth.stop()
        if preview is not None:
            preview.close()
        cv2.destroyAllWindows()
    # Newline after the \r-based status line so the shell prompt lands cleanly.
    print()
    return 0


def _extract_landmark_lists(
    tracker: Tracker, bgr: NDArray[np.uint8]
) -> tuple[list[Any], list[Any], Any | None]:
    """Re-run the underlying MediaPipe solvers so the preview sees the same
    landmark lists the tracker used."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hand_results = tracker._hands.process(rgb)  # noqa: SLF001
    face_results = (
        tracker._face_mesh.process(rgb) if tracker._face_mesh is not None else None  # noqa: SLF001
    )
    pose_results = (
        tracker._pose.process(rgb) if tracker._pose is not None else None  # noqa: SLF001
    )
    hand_lists = list(hand_results.multi_hand_landmarks or [])
    face_lists = list(face_results.multi_face_landmarks or []) if face_results is not None else []
    pose_landmarks = pose_results.pose_landmarks if pose_results is not None else None
    return hand_lists, face_lists, pose_landmarks


def _print_status(result: FrameResult) -> None:
    left = result.left.gesture if result.left.present else "-"
    right = result.right.gesture if result.right.present else "-"
    face = result.face.expression if result.face.present else "-"
    clap = "CLAP" if result.clap_event else "    "
    print(
        f"\rFPS {result.fps:5.1f}  L:{left:<10} R:{right:<10} face:{face:<10} {clap}",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
