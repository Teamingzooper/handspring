"""In-process gesture-driven synth: saw → lowpass → amp env → tremolo."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import sounddevice as sd

from handspring.synth_params import SynthParams

_SAMPLE_RATE = 48_000
_BLOCK_SIZE = 256
_SMOOTH = 0.01  # per-sample one-pole smoothing coefficient
_ENV_ATTACK_SAMPLES = int(0.005 * _SAMPLE_RATE)  # 5 ms attack
_ENV_RELEASE_SAMPLES = int(0.060 * _SAMPLE_RATE)  # 60 ms release


class Synth:
    """sounddevice-driven audio engine reading from a SynthParams store.

    Use `start()` to open the audio stream, `stop()` to release it.
    Parameters are updated externally by calling SynthParams setters.
    """

    def __init__(self, params: SynthParams) -> None:
        self._params = params
        self._stream: sd.OutputStream | None = None

        # Smoothed parameter state (starts at defaults).
        snap = params.snapshot()
        self._s_volume = snap.volume
        self._s_note_hz = snap.note_hz
        self._s_stepping_hz = snap.stepping_hz
        self._s_cutoff_hz = snap.cutoff_hz
        self._s_mod_depth = snap.mod_depth
        self._s_mod_rate = snap.mod_rate

        # Oscillator / envelope / mod state.
        self._osc_phase = 0.0
        self._mod_phase = 0.0
        self._env = 0.0  # amplitude envelope value
        self._env_target = 1.0  # attacking or releasing
        self._step_phase = 0.0  # 0..1; resets at each stepping retrigger
        self._lp_prev = 0.0  # one-pole lowpass memory

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=_SAMPLE_RATE,
            blocksize=_BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(
        self,
        outdata: np.ndarray[Any, np.dtype[np.float32]],
        frames: int,
        _time_info: Any,
        _status: Any,
    ) -> None:
        target = self._params.snapshot()
        out = outdata[:, 0]

        two_pi = 2.0 * math.pi

        for i in range(frames):
            # Smooth parameters toward targets.
            self._s_volume += (target.volume - self._s_volume) * _SMOOTH
            self._s_note_hz += (target.note_hz - self._s_note_hz) * _SMOOTH
            self._s_stepping_hz += (target.stepping_hz - self._s_stepping_hz) * _SMOOTH
            self._s_cutoff_hz += (target.cutoff_hz - self._s_cutoff_hz) * _SMOOTH
            self._s_mod_depth += (target.mod_depth - self._s_mod_depth) * _SMOOTH
            self._s_mod_rate += (target.mod_rate - self._s_mod_rate) * _SMOOTH

            # Oscillator: saw.
            self._osc_phase += self._s_note_hz / _SAMPLE_RATE
            if self._osc_phase >= 1.0:
                self._osc_phase -= 1.0
            osc = 2.0 * self._osc_phase - 1.0

            # One-pole lowpass.
            wc = self._s_cutoff_hz
            # alpha = 1 - e^(-2πwc/sr) approximates proper LP; use simpler form.
            alpha = (two_pi * wc) / (two_pi * wc + _SAMPLE_RATE)
            self._lp_prev = self._lp_prev + alpha * (osc - self._lp_prev)
            filtered = self._lp_prev

            # Stepping retrigger: if stepping_hz > 0, retrigger env periodically.
            if self._s_stepping_hz > 0.01:
                self._step_phase += self._s_stepping_hz / _SAMPLE_RATE
                if self._step_phase >= 1.0:
                    self._step_phase -= 1.0
                    self._env = 0.0  # retrigger
                    self._env_target = 1.0
            else:
                self._step_phase = 0.0
                self._env_target = 1.0

            # Envelope (attack then release).
            if self._env < self._env_target:
                self._env += 1.0 / _ENV_ATTACK_SAMPLES
                if self._env > 1.0:
                    self._env = 1.0
            elif self._s_stepping_hz > 0.01:
                # Releasing inside a step cycle.
                self._env -= 1.0 / _ENV_RELEASE_SAMPLES
                if self._env < 0.0:
                    self._env = 0.0

            # Tremolo.
            self._mod_phase += self._s_mod_rate / _SAMPLE_RATE
            if self._mod_phase >= 1.0:
                self._mod_phase -= 1.0
            tremolo = 1.0 - self._s_mod_depth * (0.5 - 0.5 * math.cos(two_pi * self._mod_phase))

            out[i] = self._s_volume * self._env * tremolo * filtered * 0.35  # headroom
