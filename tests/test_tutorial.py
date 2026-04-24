"""Tests for the tutorial state machine. No cv2 / no camera."""

from __future__ import annotations

from handspring.tutorial import StepId, TutorialStateMachine
from handspring.types import (
    FaceState,
    FrameResult,
    HandFeatures,
    HandState,
    MotionState,
    PoseState,
)


def _hf(x: float = 0.5, y: float = 0.5, thumb_dx: float = 0.1) -> HandFeatures:
    return HandFeatures(
        x=x, y=y, z=0.0,
        openness=1.0, pinch=0.0,
        index_x=x, index_y=y,
        thumb_x=x + thumb_dx, thumb_y=y,
    )


def _hand(
    gesture: str = "none",
    *,
    present: bool = True,
    x: float = 0.5, y: float = 0.5,
    pinch: float = 0.0,
    thumb_dx: float = 0.1,
) -> HandState:
    return HandState(
        present=present,
        features=_hf(x, y, thumb_dx) if present else None,
        gesture=gesture,  # type: ignore[arg-type]
        motion=MotionState(pinching=pinch >= 0.85, dragging=False, drag_dx=0, drag_dy=0, event=None),
    )


def _frame(left: HandState | None = None, right: HandState | None = None) -> FrameResult:
    return FrameResult(
        left=left or _hand(present=False),
        right=right or _hand(present=False),
        face=FaceState(False, None, "neutral", 0, 0),
        pose=PoseState(False, None),
        fps=30.0,
        clap_event=False,
    )


def test_starts_at_detect_right():
    sm = TutorialStateMachine()
    assert sm.current_step == StepId.DETECT_RIGHT


def test_detect_right_advances_after_20_frames_with_right_present():
    sm = TutorialStateMachine()
    for i in range(20):
        sm.update(_frame(right=_hand("open")), now=i * 0.033)
    assert sm.current_step == StepId.POINT_RIGHT


def test_detect_right_resets_if_right_goes_absent():
    sm = TutorialStateMachine()
    for i in range(10):
        sm.update(_frame(right=_hand("open")), now=i * 0.033)
    sm.update(_frame(right=_hand(present=False)), now=0.5)
    # counter reset
    for i in range(15):
        sm.update(_frame(right=_hand("open")), now=0.6 + i * 0.033)
    # still on first step, haven't reached 20 in a row
    assert sm.current_step == StepId.DETECT_RIGHT


def test_point_right_captures_hand_size():
    sm = TutorialStateMachine()
    # blow through step 1
    for i in range(20):
        sm.update(_frame(right=_hand("open")), now=i * 0.033)
    # now point for 15 frames with thumb_dx=0.12 (hand scale proxy)
    start_t = 0.7
    for i in range(15):
        sm.update(
            _frame(right=_hand("point", thumb_dx=0.12)),
            now=start_t + i * 0.033,
        )
    assert sm.current_step == StepId.MOVE_RIGHT
    assert sm.result.hand_size is not None
    assert 0.10 < sm.result.hand_size < 0.14


def test_skip_current_advances_with_marker():
    sm = TutorialStateMachine()
    sm.skip_current(now=0.0)
    assert StepId.DETECT_RIGHT in sm.result.skipped_steps
    assert sm.current_step == StepId.POINT_RIGHT


def test_skip_all_jumps_to_done():
    sm = TutorialStateMachine()
    sm.skip_all()
    assert sm.done
    assert sm.current_step == StepId.DONE
    # All steps except DONE should be in skipped_steps
    assert StepId.DETECT_RIGHT in sm.result.skipped_steps
    assert StepId.PEACE in sm.result.skipped_steps


def test_timeout_skips_step_after_30_seconds():
    sm = TutorialStateMachine(step_timeout_seconds=30.0)
    # Feed one initial frame at t=0 so the step's start_time is set.
    sm.update(_frame(right=_hand(present=False)), now=0.0)
    # Now jump 31 seconds ahead — should auto-skip.
    sm.update(_frame(right=_hand(present=False)), now=31.0)
    assert StepId.DETECT_RIGHT in sm.result.skipped_steps
    assert sm.current_step == StepId.POINT_RIGHT


# ---------------------------------------------------------------------------
# __main__ helper tests
# ---------------------------------------------------------------------------


def test_should_run_tutorial_skip_flag_wins(tmp_path):
    from argparse import Namespace

    from handspring.__main__ import _should_run_tutorial

    config_path = tmp_path / "config.toml"
    args = Namespace(skip_tutorial=True, tutorial=False)
    run, flag = _should_run_tutorial(args, config_path)
    assert run is False
    assert flag is None


def test_should_run_tutorial_force_flag(tmp_path):
    from argparse import Namespace

    from handspring.__main__ import _should_run_tutorial

    config_path = tmp_path / "config.toml"
    config_path.write_text("[cursor]\nsmoothing = 0.5\n")
    args = Namespace(skip_tutorial=False, tutorial=True)
    run, flag = _should_run_tutorial(args, config_path)
    assert run is True


def test_should_run_tutorial_missing_config(tmp_path):
    from argparse import Namespace

    from handspring.__main__ import _should_run_tutorial

    config_path = tmp_path / "nope.toml"
    args = Namespace(skip_tutorial=False, tutorial=False)
    run, flag = _should_run_tutorial(args, config_path)
    assert run is True


def test_should_run_tutorial_replay_flag(tmp_path):
    from argparse import Namespace

    from handspring.__main__ import _should_run_tutorial

    config_path = tmp_path / "config.toml"
    config_path.write_text("[cursor]\nsmoothing = 0.5\n")
    flag = tmp_path / "replay-tutorial.flag"
    flag.write_text("")
    args = Namespace(skip_tutorial=False, tutorial=False)
    run, flag_out = _should_run_tutorial(args, config_path)
    assert run is True
    assert flag_out == flag


def test_apply_calibration_scales_thresholds(tmp_path):
    from handspring.__main__ import _BASELINE_HAND_SIZE, _apply_calibration
    from handspring.config import Config, ConfigStore
    from handspring.tutorial import CalibrationResult

    store = ConfigStore(persist=False, initial=Config())
    base_flick = store.get().radial.flick_threshold
    base_entry = store.get().create.entry_distance
    result = CalibrationResult(hand_size=_BASELINE_HAND_SIZE * 1.5)
    _apply_calibration(store, result)
    assert store.get().radial.flick_threshold == base_flick * 1.5
    assert store.get().create.entry_distance == base_entry * 1.5


def test_apply_calibration_noop_when_hand_size_none():
    from handspring.__main__ import _apply_calibration
    from handspring.config import Config, ConfigStore
    from handspring.tutorial import CalibrationResult

    store = ConfigStore(persist=False, initial=Config())
    before = store.get()
    _apply_calibration(store, CalibrationResult(hand_size=None))
    assert store.get() == before
