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
from handspring.osc_out import OscEmitter
from handspring.preview import Preview
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
        )
    )
    emitter = OscEmitter(host=args.host, port=args.port)
    preview = Preview(mirror=args.mirror) if not args.no_preview else None
    shutdown = _Shutdown()

    print(f"handspring {__version__}", flush=True)
    print(f"camera: {args.camera}", flush=True)
    print(f"OSC:    {args.host}:{args.port}", flush=True)
    print(f"hands:  {args.hands}   face: {'off' if args.no_face else 'on'}", flush=True)
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

            if preview is not None:
                hand_landmarks, face_landmarks = _extract_landmark_lists(tracker, bgr)
                if not preview.render(
                    bgr,
                    hand_landmarks,
                    face_landmarks,
                    result,
                    f"{args.host}:{args.port}",
                ):
                    break

            now = time.monotonic()
            if now - last_log >= args.fps_log_interval:
                _print_status(result)
                last_log = now
    finally:
        cap.release()
        tracker.close()
        if preview is not None:
            preview.close()
        cv2.destroyAllWindows()
    # Newline after the \r-based status line so the shell prompt lands cleanly.
    print()
    return 0


def _extract_landmark_lists(
    tracker: Tracker, bgr: NDArray[np.uint8]
) -> tuple[list[Any], list[Any]]:
    """Re-run the underlying MediaPipe solvers so the preview sees the same
    landmark lists the tracker used.

    The duplication is intentional — MediaPipe's API doesn't expose the raw
    lists from Tracker without doubling the public surface, and a second pass
    at the already-available frame is cheap (10-20% of a frame's cost).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hand_results = tracker._hands.process(rgb)  # noqa: SLF001
    face_results = (
        tracker._face_mesh.process(rgb) if tracker._face_mesh is not None else None  # noqa: SLF001
    )
    hand_lists = list(hand_results.multi_hand_landmarks or [])
    face_lists = list(face_results.multi_face_landmarks or []) if face_results is not None else []
    return hand_lists, face_lists


def _print_status(result: FrameResult) -> None:
    left = result.left.gesture if result.left.present else "-"
    right = result.right.gesture if result.right.present else "-"
    face = "yes" if result.face.present else "no"
    print(
        f"\rFPS {result.fps:5.1f}  L:{left:<10} R:{right:<10} face:{face}",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
