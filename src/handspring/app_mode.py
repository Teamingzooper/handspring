"""App-level mode state machine: toggle between synth and jarvis via mouth-open."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

AppMode = Literal["synth", "jarvis"]

_MOUTH_THRESHOLD = 0.7
_DEBOUNCE_FRAMES = 15  # ~500 ms at 30 FPS
_COOLDOWN_SECONDS = 1.5


class AppModeController:
    def __init__(
        self,
        *,
        initial_mode: AppMode = "synth",
        on_change: Callable[[AppMode], None] | None = None,
    ) -> None:
        self._mode: AppMode = initial_mode
        self._on_change = on_change
        self._frames_open = 0
        self._last_toggle_time: float = -_COOLDOWN_SECONDS

    def mode(self) -> AppMode:
        return self._mode

    def update(
        self,
        *,
        mouth_open: float,
        face_present: bool,
        now: float,
    ) -> None:
        """Call once per frame with the current face.mouth_open + face.present."""
        if not face_present:
            self._frames_open = 0
            return

        if mouth_open >= _MOUTH_THRESHOLD:
            self._frames_open += 1
        else:
            self._frames_open = 0

        if (
            self._frames_open >= _DEBOUNCE_FRAMES
            and now - self._last_toggle_time >= _COOLDOWN_SECONDS
        ):
            self._mode = "jarvis" if self._mode == "synth" else "synth"
            self._frames_open = 0
            self._last_toggle_time = now
            if self._on_change is not None:
                self._on_change(self._mode)
