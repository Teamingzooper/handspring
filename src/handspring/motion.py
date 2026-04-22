"""Pure-function motion detectors reading from HandHistory buffers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from handspring.types import MotionEvent

if TYPE_CHECKING:
    from handspring.history import HandHistory


# ---- Tuning thresholds (documented in spec) ----

_PINCH_ON = 0.85
_PINCH_OFF = 0.4
_PINCH_COOLDOWN = 0.3  # seconds
_DRAG_VELOCITY_ON = 0.1  # units/second (normalized frame units)
_DRAG_VELOCITY_OFF = 0.03
_DRAG_ARM_DURATION = 0.2  # seconds of sustained motion while pinching to enter drag
_DRAG_IDLE_DURATION = 0.3  # seconds of low velocity → exit drag
_WAVE_AMPLITUDE = 0.05
_WAVE_MIN_FREQ = 1.5
_WAVE_MAX_FREQ = 4.0
_WAVE_MIN_DURATION = 0.8
_WAVE_Y_MAX = 0.5  # wave only if hand.y < this (above mid-frame)
_WAVE_COOLDOWN = 1.0
_CLAP_NEAR = 0.08
_CLAP_FAR = 0.25
_CLAP_WINDOW = 0.3
_CLAP_COOLDOWN = 0.4


@dataclass(frozen=True)
class MotionUpdate:
    pinching: bool
    dragging: bool
    drag_dx: float
    drag_dy: float
    event: MotionEvent | None


class MotionDetector:
    """Stateful per-hand motion detector.

    Reads features from the caller's HandHistory and emits one-shot
    MotionEvents plus continuous pinching/dragging state. State is
    minimal (just recent transitions and cooldowns); the HandHistory
    carries the bulk of the time-series.
    """

    def __init__(self) -> None:
        self._was_pinching = False
        self._pinch_cooldown_until = -1.0
        self._wave_cooldown_until = -1.0

        self._dragging = False
        self._drag_start_xy: tuple[float, float] | None = None
        self._arming_drag_since: float | None = None
        self._idle_since: float | None = None

    def update(self, history: HandHistory, now: float) -> MotionUpdate:
        latest = history.latest()
        if latest is None:
            self._was_pinching = False
            self._reset_drag()
            return MotionUpdate(False, False, 0.0, 0.0, None)

        pinching = latest.features.pinch >= _PINCH_ON or (
            self._was_pinching and latest.features.pinch > _PINCH_OFF
        )
        event: MotionEvent | None = None

        # --- Pinch rising edge ---
        if pinching and not self._was_pinching and now >= self._pinch_cooldown_until:
            event = "pinch"
            self._pinch_cooldown_until = now + _PINCH_COOLDOWN

        # --- Expand falling edge ---
        if not pinching and self._was_pinching and now >= self._pinch_cooldown_until:
            event = "expand"
            self._pinch_cooldown_until = now + _PINCH_COOLDOWN
            # Ending pinch always ends drag too.
            if self._dragging:
                event = "drag_end"
                self._reset_drag()
            else:
                self._reset_drag()

        self._was_pinching = pinching

        # --- Drag logic (only while pinching) ---
        drag_dx = 0.0
        drag_dy = 0.0
        if pinching:
            velocity = _recent_velocity(history, window=0.1, now=now)
            if not self._dragging:
                if velocity >= _DRAG_VELOCITY_ON:
                    if self._arming_drag_since is None:
                        self._arming_drag_since = now
                    elif now - self._arming_drag_since >= _DRAG_ARM_DURATION:
                        # Commit to drag.
                        self._dragging = True
                        self._drag_start_xy = (latest.features.x, latest.features.y)
                        self._arming_drag_since = None
                        self._idle_since = None
                        if event is None:
                            event = "drag_start"
                else:
                    self._arming_drag_since = None
            else:
                # Currently dragging — update delta + watch for idle.
                if self._drag_start_xy is not None:
                    drag_dx = latest.features.x - self._drag_start_xy[0]
                    drag_dy = latest.features.y - self._drag_start_xy[1]
                if velocity < _DRAG_VELOCITY_OFF:
                    if self._idle_since is None:
                        self._idle_since = now
                    elif now - self._idle_since >= _DRAG_IDLE_DURATION:
                        if event is None:
                            event = "drag_end"
                        self._reset_drag()
                else:
                    self._idle_since = None
        else:
            self._arming_drag_since = None

        # --- Wave detection (only if no event claimed yet) ---
        if event is None and now >= self._wave_cooldown_until and _detect_wave(history, now=now):
            event = "wave"
            self._wave_cooldown_until = now + _WAVE_COOLDOWN

        return MotionUpdate(
            pinching=pinching,
            dragging=self._dragging,
            drag_dx=drag_dx,
            drag_dy=drag_dy,
            event=event,
        )

    def _reset_drag(self) -> None:
        self._dragging = False
        self._drag_start_xy = None
        self._arming_drag_since = None
        self._idle_since = None


def _recent_velocity(history: HandHistory, window: float, now: float) -> float:
    """Average speed (units/sec) over samples in the last `window` seconds."""
    recent = history.samples_since(now - window)
    if len(recent) < 2:
        return 0.0
    dx = recent[-1].features.x - recent[0].features.x
    dy = recent[-1].features.y - recent[0].features.y
    dt = recent[-1].timestamp - recent[0].timestamp
    if dt <= 0:
        return 0.0
    return math.hypot(dx, dy) / dt


def _detect_wave(history: HandHistory, now: float) -> bool:
    """Detect a periodic horizontal oscillation in the last ~1s window."""
    window_start = now - 1.0
    samples = history.samples_since(window_start)
    if len(samples) < 10:
        return False
    # Require hand above mid-frame (avoid resting-hand-shake false positives).
    if any(s.features.y > _WAVE_Y_MAX for s in samples):
        return False
    xs = [s.features.x for s in samples]
    min_x = min(xs)
    max_x = max(xs)
    amplitude = (max_x - min_x) / 2.0
    if amplitude < _WAVE_AMPLITUDE:
        return False
    # Count zero-crossings around the mean to estimate frequency.
    mean = sum(xs) / len(xs)
    crossings = 0
    prev_above = xs[0] > mean
    for x in xs[1:]:
        above = x > mean
        if above != prev_above:
            crossings += 1
            prev_above = above
    # Two crossings per cycle.
    duration = samples[-1].timestamp - samples[0].timestamp
    if duration < _WAVE_MIN_DURATION:
        return False
    freq = crossings / 2.0 / duration if duration > 0 else 0.0
    return _WAVE_MIN_FREQ <= freq <= _WAVE_MAX_FREQ


class _ClapDetector:
    """Detects hand-clap impacts from a running stream of hand-to-hand distances."""

    def __init__(self) -> None:
        self._history: list[tuple[float, float]] = []  # (timestamp, distance)
        self._cooldown_until = -1.0

    def update(self, distance: float, now: float) -> bool:
        # Keep only last `_CLAP_WINDOW + 0.1` seconds.
        cutoff = now - (_CLAP_WINDOW + 0.1)
        self._history = [(t, d) for (t, d) in self._history if t >= cutoff]
        self._history.append((now, distance))

        if now < self._cooldown_until:
            return False
        if distance >= _CLAP_NEAR:
            return False
        # distance < near; check if recently we were >= far.
        recent = [d for (t, d) in self._history if t >= now - _CLAP_WINDOW]
        if any(d >= _CLAP_FAR for d in recent):
            self._cooldown_until = now + _CLAP_COOLDOWN
            return True
        return False


def bi_hand_clap_detector() -> _ClapDetector:
    """Factory — keeps the class internal but returns a reusable instance."""
    return _ClapDetector()
