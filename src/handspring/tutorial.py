"""First-run tutorial: state machine, renderer, and glue loop.

The tutorial walks the user through core gestures and captures calibration
data (hand size, pinch thresholds). It is structured as three classes:

- TutorialStateMachine  — pure logic, testable without cv2 or a camera.
- TutorialRenderer      — draws overlays onto a BGR frame using cv2.
- Tutorial              — glue: owns VideoCapture + Tracker, runs the loop.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from handspring.features import is_pinching
from handspring.types import FrameResult, HandState

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Public API types
# ---------------------------------------------------------------------------


class StepId(str, Enum):
    DETECT_RIGHT = "detect_right"
    POINT_RIGHT = "point_right"
    MOVE_RIGHT = "move_right"
    PINCH_RIGHT = "pinch_right"
    PINCH_LEFT = "pinch_left"
    RADIAL_FLICK = "radial_flick"
    PEACE = "peace"
    DONE = "done"


# Ordered list of actionable steps (excludes DONE).
_STEPS: list[StepId] = [
    StepId.DETECT_RIGHT,
    StepId.POINT_RIGHT,
    StepId.MOVE_RIGHT,
    StepId.PINCH_RIGHT,
    StepId.PINCH_LEFT,
    StepId.RADIAL_FLICK,
    StepId.PEACE,
]

_STEP_INSTRUCTIONS: dict[StepId, str] = {
    StepId.DETECT_RIGHT: "Show your right hand to the camera.",
    StepId.POINT_RIGHT: "Point your index finger and hold still.",
    StepId.MOVE_RIGHT: "Move your right hand across the frame.",
    StepId.PINCH_RIGHT: "Pinch your right thumb and index finger together.",
    StepId.PINCH_LEFT: "Pinch your left thumb and index finger together.",
    StepId.RADIAL_FLICK: "Pinch your left hand, flick in any direction, then release.",
    StepId.PEACE: "Show a peace sign with either hand.",
    StepId.DONE: "Tutorial complete!",
}

_STEP_TITLES: dict[StepId, str] = {
    StepId.DETECT_RIGHT: "Detect Right Hand",
    StepId.POINT_RIGHT: "Point Gesture",
    StepId.MOVE_RIGHT: "Move Hand",
    StepId.PINCH_RIGHT: "Right Pinch",
    StepId.PINCH_LEFT: "Left Pinch",
    StepId.RADIAL_FLICK: "Radial Flick",
    StepId.PEACE: "Peace Sign",
    StepId.DONE: "Done",
}


@dataclass
class CalibrationResult:
    """Captured values from the tutorial. None for steps the user skipped."""

    hand_size: float | None = None        # camera-space thumb-index spread during point
    pinch_min_right: float | None = None  # min pinch distance during right-pinch step
    pinch_min_left: float | None = None   # min pinch distance during left-pinch step
    skipped_steps: set[StepId] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _thumb_index_dist(hand: HandState) -> float:
    """Euclidean distance between thumb tip and index tip in camera space."""
    if hand.features is None:
        return float("inf")
    dx = hand.features.thumb_x - hand.features.index_x
    dy = hand.features.thumb_y - hand.features.index_y
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# TutorialStateMachine
# ---------------------------------------------------------------------------


class TutorialStateMachine:
    """Pure-logic state machine for the first-run tutorial.

    Feed FrameResult objects via ``update(frame, now)`` (wall-clock seconds).
    Check ``current_step``, ``done``, and ``result`` to observe progress.
    """

    def __init__(self, step_timeout_seconds: float = 30.0) -> None:
        self._timeout = step_timeout_seconds
        self._result = CalibrationResult()

        # Step ordering
        self._step_index: int = 0  # index into _STEPS; len(_STEPS) == DONE
        self._step_start_time: float | None = None

        # Per-step counters / accumulators (reset on each transition)
        self._counter: int = 0

        # POINT_RIGHT
        self._point_accum: list[float] = []

        # MOVE_RIGHT
        self._prev_xy: tuple[float, float] | None = None
        self._traveled: float = 0.0

        # PINCH_RIGHT / PINCH_LEFT
        self._pinch_counter: int = 0
        self._pinch_min: float = float("inf")

        # RADIAL_FLICK sub-state: "waiting_pinch" | "pinched" | "moved"
        self._flick_state: str = "waiting_pinch"
        self._flick_origin: tuple[float, float] | None = None

        # PEACE
        self._peace_counter: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> StepId:
        if self._step_index >= len(_STEPS):
            return StepId.DONE
        return _STEPS[self._step_index]

    @property
    def done(self) -> bool:
        return self._step_index >= len(_STEPS)

    @property
    def result(self) -> CalibrationResult:
        return self._result

    @property
    def instruction(self) -> str:
        return _STEP_INSTRUCTIONS.get(self.current_step, "")

    @property
    def step_title(self) -> str:
        return _STEP_TITLES.get(self.current_step, "")

    @property
    def step_number(self) -> int:
        """1-based step number (0 when done)."""
        if self.done:
            return 0
        return self._step_index + 1

    @property
    def progress_fraction(self) -> float:
        """Soft 0..1 progress within the current step."""
        step = self.current_step
        if step == StepId.DETECT_RIGHT:
            return min(self._counter / 20, 1.0)
        if step == StepId.POINT_RIGHT:
            return min(len(self._point_accum) / 15, 1.0)
        if step == StepId.MOVE_RIGHT:
            return min(self._traveled / 0.3, 1.0)
        if step == StepId.PINCH_RIGHT or step == StepId.PINCH_LEFT:
            return min(self._pinch_counter / 8, 1.0)
        if step == StepId.RADIAL_FLICK:
            if self._flick_state == "waiting_pinch":
                return 0.0
            if self._flick_state == "pinched":
                return 0.5
            return 0.9  # moved
        if step == StepId.PEACE:
            return min(self._peace_counter / 10, 1.0)
        return 1.0  # DONE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance(self) -> None:
        """Move to the next step and reset per-step state."""
        self._step_index += 1
        self._step_start_time = None
        self._counter = 0
        self._point_accum = []
        self._prev_xy = None
        self._traveled = 0.0
        self._pinch_counter = 0
        self._pinch_min = float("inf")
        self._flick_state = "waiting_pinch"
        self._flick_origin = None
        self._peace_counter = 0

    def _check_timeout(self, now: float) -> bool:
        """Return True (and auto-skip) if the current step has timed out."""
        if self._step_start_time is None:
            return False
        if now - self._step_start_time >= self._timeout:
            self.skip_current(now)
            return True
        return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, frame: FrameResult, now: float) -> None:
        """Feed a frame. Advances step if criteria met."""
        if self.done:
            return

        # Record start time on first frame of a new step.
        if self._step_start_time is None:
            self._step_start_time = now

        # Check timeout before processing.
        if self._check_timeout(now):
            return

        step = self.current_step

        if step == StepId.DETECT_RIGHT:
            self._update_detect_right(frame)
        elif step == StepId.POINT_RIGHT:
            self._update_point_right(frame)
        elif step == StepId.MOVE_RIGHT:
            self._update_move_right(frame)
        elif step == StepId.PINCH_RIGHT:
            self._update_pinch(frame.right, is_right=True)
        elif step == StepId.PINCH_LEFT:
            self._update_pinch(frame.left, is_right=False)
        elif step == StepId.RADIAL_FLICK:
            self._update_radial_flick(frame)
        elif step == StepId.PEACE:
            self._update_peace(frame)

    def skip_current(self, now: float) -> None:  # noqa: ARG002
        """User pressed Skip — mark this step skipped, advance."""
        if self.done:
            return
        self._result.skipped_steps.add(self.current_step)
        self._advance()

    def skip_all(self) -> None:
        """User pressed Esc/q — mark all remaining as skipped, move to DONE."""
        while not self.done:
            self._result.skipped_steps.add(self.current_step)
            self._advance()

    # ------------------------------------------------------------------
    # Per-step update logic
    # ------------------------------------------------------------------

    def _update_detect_right(self, frame: FrameResult) -> None:
        if frame.right.present:
            self._counter += 1
            if self._counter >= 20:
                self._advance()
        else:
            self._counter = 0

    def _update_point_right(self, frame: FrameResult) -> None:
        right = frame.right
        if right.present and right.gesture == "point" and right.features is not None:
            dx = right.features.thumb_x - right.features.index_x
            dy = right.features.thumb_y - right.features.index_y
            dist = math.sqrt(dx * dx + dy * dy)
            self._point_accum.append(dist)
            if len(self._point_accum) >= 15:
                self._result.hand_size = sum(self._point_accum) / len(self._point_accum)
                self._advance()
        else:
            # Reset if gesture breaks
            self._point_accum = []

    def _update_move_right(self, frame: FrameResult) -> None:
        right = frame.right
        if not right.present or right.features is None:
            self._prev_xy = None
            return
        x, y = right.features.index_x, right.features.index_y
        if self._prev_xy is not None:
            px, py = self._prev_xy
            delta = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            self._traveled += delta
        self._prev_xy = (x, y)
        if self._traveled >= 0.3:
            self._advance()

    def _update_pinch(self, hand: HandState, *, is_right: bool) -> None:
        if is_pinching(hand):
            self._pinch_counter += 1
            dist = _thumb_index_dist(hand)
            if dist < self._pinch_min:
                self._pinch_min = dist
            if self._pinch_counter >= 8:
                if is_right:
                    self._result.pinch_min_right = self._pinch_min
                else:
                    self._result.pinch_min_left = self._pinch_min
                self._advance()
        else:
            self._pinch_counter = 0
            self._pinch_min = float("inf")

    def _update_radial_flick(self, frame: FrameResult) -> None:
        left = frame.left
        pinching = is_pinching(left)

        if self._flick_state == "waiting_pinch":
            if pinching and left.features is not None:
                self._flick_state = "pinched"
                self._flick_origin = (left.features.index_x, left.features.index_y)

        elif self._flick_state == "pinched":
            if not pinching:
                # Released without reaching displacement threshold — reset
                self._flick_state = "waiting_pinch"
                self._flick_origin = None
            elif left.features is not None and self._flick_origin is not None:
                ox, oy = self._flick_origin
                dx = left.features.index_x - ox
                dy = left.features.index_y - oy
                if math.sqrt(dx * dx + dy * dy) >= 0.03:
                    self._flick_state = "moved"

        elif self._flick_state == "moved" and not pinching:
            # Released after moving — success!
            self._advance()

    def _update_peace(self, frame: FrameResult) -> None:
        left_peace = frame.left.present and frame.left.gesture == "peace"
        right_peace = frame.right.present and frame.right.gesture == "peace"
        if left_peace or right_peace:
            self._peace_counter += 1
            if self._peace_counter >= 10:
                self._advance()
        else:
            self._peace_counter = 0


# ---------------------------------------------------------------------------
# TutorialRenderer
# ---------------------------------------------------------------------------


class TutorialRenderer:
    """Draws tutorial overlays onto a BGR frame using cv2."""

    _BAR_H = 60
    _BOTTOM_H = 30

    def draw(self, bgr_frame: NDArray[np.uint8], sm: TutorialStateMachine) -> NDArray[np.uint8]:
        """Return a new BGR frame with tutorial overlay drawn."""
        import cv2

        frame = bgr_frame.copy()
        h, w = frame.shape[:2]

        # --- Top bar ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self._BAR_H), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Step title
        title = f"[{sm.step_number}/7] {sm.step_title}"
        cv2.putText(frame, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Instruction
        cv2.putText(frame, sm.instruction, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Progress bar within the top bar
        bar_y = self._BAR_H - 6
        bar_w = int(w * sm.progress_fraction)
        cv2.rectangle(frame, (0, bar_y), (bar_w, self._BAR_H - 2), (80, 200, 80), -1)

        # --- Bottom status bar ---
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - self._BOTTOM_H), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        # Calibration data summary
        res = sm.result
        parts: list[str] = [f"Step {sm.step_number}/7"]
        if res.hand_size is not None:
            parts.append(f"hand_size={res.hand_size:.3f}")
        if res.pinch_min_right is not None:
            parts.append(f"pinch_R={res.pinch_min_right:.3f}")
        if res.pinch_min_left is not None:
            parts.append(f"pinch_L={res.pinch_min_left:.3f}")
        status = "  ".join(parts)
        cv2.putText(
            frame, status,
            (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )

        # --- Green border pulse on step complete ---
        if sm.progress_fraction >= 1.0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 220, 0), 4)

        return frame


# ---------------------------------------------------------------------------
# Tutorial glue
# ---------------------------------------------------------------------------


class Tutorial:
    """Glue class: owns VideoCapture + Tracker, drives the state machine."""

    def __init__(self, tracker: object, step_timeout_seconds: float = 30.0) -> None:
        self._tracker = tracker
        self._timeout = step_timeout_seconds

    def run(self, cap: object) -> CalibrationResult | None:
        """Run the tutorial loop.

        Returns CalibrationResult on success/skip, None if cap can't be read.
        """
        import cv2

        sm = TutorialStateMachine(step_timeout_seconds=self._timeout)
        renderer = TutorialRenderer()
        window = "handspring · tutorial"

        try:
            while True:
                ret, bgr = cap.read()  # type: ignore[union-attr]
                if not ret or bgr is None:
                    return None

                frame_result: FrameResult = self._tracker.process(bgr)  # type: ignore[union-attr]
                now = time.monotonic()

                sm.update(frame_result, now)

                display = renderer.draw(bgr, sm)
                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # Esc or q
                    sm.skip_all()
                elif key == 32:  # space
                    sm.skip_current(now)

                if sm.done:
                    break

        finally:
            cv2.destroyWindow(window)

        return sm.result
