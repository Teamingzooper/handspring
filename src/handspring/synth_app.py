"""`handspring-synth` entry point: camera → tracker → synth + OSC + preview.

The synth half of handspring, now runnable as its own process so the jarvis
half (desktop integration) can run independently. Mouth-toggle app-mode
switching is gone — each entry point owns its own camera and preview window.
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

from handspring import __version__
from handspring.osc_out import OscEmitter
from handspring.preview import Preview
from handspring.synth import Synth
from handspring.synth_params import SynthParams
from handspring.synth_ui import SynthController
from handspring.tracker import Tracker, TrackerConfig
from handspring.types import FrameResult


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="handspring-synth",
        description="Gesture-driven synth (synth-only entry point).",
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
    p.add_argument(
        "--no-synth", action="store_true", help="disable audio output (OSC still streams)"
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.set_defaults(mirror=True)
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
    synth_params = SynthParams()
    synth_controller = SynthController(synth_params)

    synth: Synth | None = None
    if not args.no_synth:
        try:
            synth = Synth(synth_params)
            synth.start()
        except Exception as e:  # noqa: BLE001
            print(
                f"warning: could not start synth ({e}); continuing without audio",
                file=sys.stderr,
            )
            synth = None

    preview = Preview(mirror=args.mirror) if not args.no_preview else None
    shutdown = _Shutdown()

    synth_status = "off" if args.no_synth or synth is None else "on"
    print(f"handspring-synth {__version__}", flush=True)
    print(f"camera: {args.camera}   synth: {synth_status}", flush=True)
    print(f"OSC:    {args.host}:{args.port}", flush=True)
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
            synth_controller.update(result)
            now = time.monotonic()

            emitter.emit_app_mode("synth")
            if not args.no_synth:
                emitter.emit_synth(synth_params.snapshot())

            if preview is not None:
                snap_for_preview = synth_params.snapshot() if not args.no_synth else None
                hint_for_preview = synth_controller.ui_hint() if not args.no_synth else None
                if not preview.render(
                    bgr,
                    tracker_output.hand_landmark_lists,
                    tracker_output.face_landmark_lists,
                    tracker_output.pose_landmarks,
                    result,
                    f"{args.host}:{args.port}",
                    snap_for_preview,
                    hint_for_preview,
                    "synth",
                    None,  # no jarvis controller in synth-only mode
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
    print()
    return 0


def _print_status(result: FrameResult) -> None:
    left = result.left.gesture if result.left.present else "-"
    right = result.right.gesture if result.right.present else "-"
    face = result.face.expression if result.face.present else "-"
    print(
        f"\rFPS {result.fps:5.1f}  L:{left:<10} R:{right:<10} face:{face:<10}",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
