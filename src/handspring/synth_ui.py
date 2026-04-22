"""Gesture → synth parameter state machine.

Reads FrameResult each frame, handles fist debouncing, determines the active
mode, and writes to the SynthParams instance. Also exposes `ui_hints` so the
preview can draw the correct slider / crosshair at the current finger position.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

from handspring.synth_params import (
    CUTOFF_MAX,
    CUTOFF_MIN,
    MOD_DEPTH_MAX,
    MOD_DEPTH_MIN,
    MOD_RATE_MAX,
    MOD_RATE_MIN,
    NOTE_MAX,
    NOTE_MIN,
    STEP_MAX,
    STEP_MIN,
    SynthParams,
)
from handspring.types import FrameResult, HandState

_DEBOUNCE_N = 3

# UI hint kinds.
HintKind = Literal["none", "slider", "xy"]


@dataclass(frozen=True)
class UiHint:
    kind: HintKind
    # slider anchor (for slider) or palm center (for xy), normalized 0..1
    x: float
    y: float
    # Primary label + value for the slider (or x-axis label for xy).
    label_a: str
    value_a: float  # normalized 0..1
    display_a: str  # pretty-printed value text
    # Secondary (y-axis) for xy only.
    label_b: str
    value_b: float
    display_b: str


_NO_HINT = UiHint("none", 0.0, 0.0, "", 0.0, "", "", 0.0, "")


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


def _log_lerp(lo: float, hi: float, t: float) -> float:
    return math.exp(_lerp(math.log(lo), math.log(hi), t))


def _y_to_norm(y: float) -> float:
    """Hand Y 0 (top) → 1.0, Y 1 (bottom) → 0.0. Clamped."""
    return max(0.0, min(1.0, 1.0 - y))


def _x_to_norm(x: float) -> float:
    return max(0.0, min(1.0, x))


class SynthController:
    def __init__(self, params: SynthParams) -> None:
        self._params = params
        self._left_fists: deque[bool] = deque(maxlen=_DEBOUNCE_N)
        self._right_fists: deque[bool] = deque(maxlen=_DEBOUNCE_N)
        self._active_mode: Literal["play", "edit_left", "edit_right"] = "play"
        self._last_hint: UiHint = _NO_HINT
        self._slider_anchor: tuple[float, float] | None = None
        self._anchor_owner_mode: str | None = None
        self._prev_mode: str = "play"

    def update(self, frame: FrameResult) -> None:
        # Record fist state for debounce.
        self._left_fists.append(frame.left.present and frame.left.gesture == "fist")
        self._right_fists.append(frame.right.present and frame.right.gesture == "fist")

        left_fist_stable = len(self._left_fists) == _DEBOUNCE_N and all(self._left_fists)
        right_fist_stable = len(self._right_fists) == _DEBOUNCE_N and all(self._right_fists)
        left_released = len(self._left_fists) == _DEBOUNCE_N and not any(self._left_fists)
        right_released = len(self._right_fists) == _DEBOUNCE_N and not any(self._right_fists)

        # Mode transitions.
        if self._active_mode == "play":
            if left_fist_stable:
                self._active_mode = "edit_left"
            elif right_fist_stable:
                self._active_mode = "edit_right"
        elif self._active_mode == "edit_left":
            if left_released:
                if right_fist_stable:
                    self._active_mode = "edit_right"
                else:
                    self._active_mode = "play"
        else:  # edit_right
            if right_released:
                if left_fist_stable:
                    self._active_mode = "edit_left"
                else:
                    self._active_mode = "play"

        self._params.set_mode(self._active_mode)

        # Clear slider anchor if mode changed since last update.
        if self._active_mode != self._prev_mode:
            self._clear_slider_anchor()
        self._prev_mode = self._active_mode

        # Apply edits based on the non-fist hand.
        self._last_hint = _NO_HINT
        if self._active_mode == "edit_left":
            self._apply_edit_left(frame.right)
        elif self._active_mode == "edit_right":
            self._apply_edit_right(frame.left)

    def ui_hint(self) -> UiHint:
        return self._last_hint

    # ---- Mode 1: left fist, right hand edits ----

    def _clear_slider_anchor(self) -> None:
        self._slider_anchor = None
        self._anchor_owner_mode = None

    def _apply_edit_left(self, right: HandState) -> None:
        if not right.present or right.features is None:
            self._clear_slider_anchor()
            return
        f = right.features
        if right.gesture == "point":
            vol = _y_to_norm(f.y)
            self._params.set_volume(vol)
            # Anchor the slider at the first point-frame; stay pinned thereafter.
            if self._slider_anchor is None or self._anchor_owner_mode != "edit_left":
                self._slider_anchor = (f.index_x, f.index_y)
                self._anchor_owner_mode = "edit_left"
            anchor_x, anchor_y = self._slider_anchor
            self._last_hint = UiHint(
                "slider",
                x=anchor_x,
                y=anchor_y,
                label_a="volume",
                value_a=vol,
                display_a=f"{vol:.2f}",
                label_b="",
                value_b=0.0,
                display_b="",
            )
        elif right.gesture == "open":
            pitch_t = _y_to_norm(f.y)
            note_hz = _log_lerp(NOTE_MIN, NOTE_MAX, pitch_t)
            step_t = _x_to_norm(f.x)
            stepping_hz = _lerp(STEP_MIN, STEP_MAX, step_t)
            self._params.set_note_hz(note_hz)
            self._params.set_stepping_hz(stepping_hz)
            self._clear_slider_anchor()
            self._last_hint = UiHint(
                "xy",
                x=f.x,
                y=f.y,
                label_a="step",
                value_a=step_t,
                display_a=f"{stepping_hz:.1f} Hz",
                label_b="pitch",
                value_b=pitch_t,
                display_b=f"{_hz_to_note(note_hz)} ({note_hz:.0f} Hz)",
            )
        else:
            self._clear_slider_anchor()

    # ---- Mode 2: right fist, left hand edits ----

    def _apply_edit_right(self, left: HandState) -> None:
        if not left.present or left.features is None:
            self._clear_slider_anchor()
            return
        f = left.features
        if left.gesture == "point":
            cutoff_t = _y_to_norm(f.y)
            cutoff_hz = _log_lerp(CUTOFF_MIN, CUTOFF_MAX, cutoff_t)
            self._params.set_cutoff_hz(cutoff_hz)
            if self._slider_anchor is None or self._anchor_owner_mode != "edit_right":
                self._slider_anchor = (f.index_x, f.index_y)
                self._anchor_owner_mode = "edit_right"
            anchor_x, anchor_y = self._slider_anchor
            self._last_hint = UiHint(
                "slider",
                x=anchor_x,
                y=anchor_y,
                label_a="cutoff",
                value_a=cutoff_t,
                display_a=f"{cutoff_hz:.0f} Hz",
                label_b="",
                value_b=0.0,
                display_b="",
            )
        elif left.gesture == "open":
            depth_t = _y_to_norm(f.y)
            rate_t = _x_to_norm(f.x)
            mod_depth = _lerp(MOD_DEPTH_MIN, MOD_DEPTH_MAX, depth_t)
            mod_rate = _log_lerp(MOD_RATE_MIN, MOD_RATE_MAX, rate_t)
            self._params.set_mod_depth(mod_depth)
            self._params.set_mod_rate(mod_rate)
            self._clear_slider_anchor()
            self._last_hint = UiHint(
                "xy",
                x=f.x,
                y=f.y,
                label_a="mod_rate",
                value_a=rate_t,
                display_a=f"{mod_rate:.2f} Hz",
                label_b="mod_depth",
                value_b=depth_t,
                display_b=f"{mod_depth:.2f}",
            )
        else:
            self._clear_slider_anchor()


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_note(hz: float) -> str:
    if hz <= 0:
        return "-"
    midi = 69.0 + 12.0 * math.log2(hz / 440.0)
    midi_int = round(midi)
    octave = midi_int // 12 - 1
    name = _NOTE_NAMES[midi_int % 12]
    return f"{name}{octave}"
