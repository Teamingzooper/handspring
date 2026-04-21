"""Thread-safe parameter store for the in-process synth.

Writes are done by the main thread (via the gesture → parameter mapper).
Reads happen in the audio callback thread once per block via `snapshot()`,
which acquires the lock briefly and returns a frozen copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal

SynthMode = Literal["play", "edit_left", "edit_right"]


# Hard ranges (spec §Autonomous design decisions).
VOL_MIN, VOL_MAX = 0.0, 1.0
NOTE_MIN, NOTE_MAX = 131.0, 1047.0  # C3..C6
STEP_MIN, STEP_MAX = 0.0, 16.0
CUTOFF_MIN, CUTOFF_MAX = 200.0, 8000.0
MOD_DEPTH_MIN, MOD_DEPTH_MAX = 0.0, 1.0
MOD_RATE_MIN, MOD_RATE_MAX = 0.1, 10.0


@dataclass(frozen=True)
class SynthSnapshot:
    volume: float
    note_hz: float
    stepping_hz: float
    cutoff_hz: float
    mod_depth: float
    mod_rate: float
    mode: SynthMode


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


class SynthParams:
    def __init__(self) -> None:
        self._lock = Lock()
        self._volume = 0.4
        self._note_hz = 440.0
        self._stepping_hz = 0.0
        self._cutoff_hz = 3000.0
        self._mod_depth = 0.0
        self._mod_rate = 1.0
        self._mode: SynthMode = "play"

    def snapshot(self) -> SynthSnapshot:
        with self._lock:
            return SynthSnapshot(
                volume=self._volume,
                note_hz=self._note_hz,
                stepping_hz=self._stepping_hz,
                cutoff_hz=self._cutoff_hz,
                mod_depth=self._mod_depth,
                mod_rate=self._mod_rate,
                mode=self._mode,
            )

    def set_volume(self, v: float) -> None:
        with self._lock:
            self._volume = _clamp(v, VOL_MIN, VOL_MAX)

    def set_note_hz(self, hz: float) -> None:
        with self._lock:
            self._note_hz = _clamp(hz, NOTE_MIN, NOTE_MAX)

    def set_stepping_hz(self, hz: float) -> None:
        with self._lock:
            self._stepping_hz = _clamp(hz, STEP_MIN, STEP_MAX)

    def set_cutoff_hz(self, hz: float) -> None:
        with self._lock:
            self._cutoff_hz = _clamp(hz, CUTOFF_MIN, CUTOFF_MAX)

    def set_mod_depth(self, v: float) -> None:
        with self._lock:
            self._mod_depth = _clamp(v, MOD_DEPTH_MIN, MOD_DEPTH_MAX)

    def set_mod_rate(self, hz: float) -> None:
        with self._lock:
            self._mod_rate = _clamp(hz, MOD_RATE_MIN, MOD_RATE_MAX)

    def set_mode(self, mode: SynthMode) -> None:
        with self._lock:
            self._mode = mode
