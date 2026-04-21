"""Minimal demo receiver for handspring.

- Left hand Y → pitch in 200–800 Hz (exponential; low Y = low pitch)
- Right hand Y → amplitude in 0..0.3
- Left hand gesture "fist" → mute; "open" → unmute

Run:
    python examples/tone_synth.py            # listens on 127.0.0.1:9000
    python examples/tone_synth.py --host X   # override
"""

from __future__ import annotations

import argparse
import math
import threading
from typing import Any

import numpy as np
import sounddevice as sd  # type: ignore[import-not-found]
from pythonosc import dispatcher, osc_server

SAMPLE_RATE = 48_000
SMOOTH = 0.01  # one-pole smoothing coef per frame (smaller number = slower)


class Synth:
    def __init__(self) -> None:
        self.target_freq = 440.0
        self.target_amp = 0.0
        self.muted = False
        self._freq = 440.0
        self._amp = 0.0
        self._phase = 0.0
        self._lock = threading.Lock()

    def set_left_y(self, y: float) -> None:
        y = max(0.0, min(1.0, float(y)))
        # y=0 (top) → high pitch; y=1 (bottom) → low pitch.
        log_low = math.log2(200.0)
        log_high = math.log2(800.0)
        self.target_freq = 2 ** (log_high - y * (log_high - log_low))

    def set_right_y(self, y: float) -> None:
        y = max(0.0, min(1.0, float(y)))
        self.target_amp = (1.0 - y) * 0.3

    def set_left_gesture(self, gesture: str) -> None:
        if gesture == "fist":
            self.muted = True
        elif gesture == "open":
            self.muted = False

    def callback(self, outdata: np.ndarray, frames: int, _time_info: Any, _status: Any) -> None:
        with self._lock:
            for i in range(frames):
                self._freq += (self.target_freq - self._freq) * SMOOTH
                target_amp = 0.0 if self.muted else self.target_amp
                self._amp += (target_amp - self._amp) * SMOOTH
                sample = math.sin(self._phase) * self._amp
                outdata[i, 0] = sample
                self._phase += 2 * math.pi * self._freq / SAMPLE_RATE
                if self._phase > 2 * math.pi:
                    self._phase -= 2 * math.pi


def _make_dispatcher(synth: Synth) -> dispatcher.Dispatcher:
    d = dispatcher.Dispatcher()
    d.map("/hand/left/y", lambda _addr, val: synth.set_left_y(val))
    d.map("/hand/right/y", lambda _addr, val: synth.set_right_y(val))
    d.map("/hand/left/gesture", lambda _addr, val: synth.set_left_gesture(str(val)))
    return d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    synth = Synth()
    with sd.OutputStream(
        channels=1, samplerate=SAMPLE_RATE, callback=synth.callback, blocksize=256
    ):
        server = osc_server.ThreadingOSCUDPServer((args.host, args.port), _make_dispatcher(synth))
        print(f"tone_synth listening on {args.host}:{args.port}; Ctrl+C to quit")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
