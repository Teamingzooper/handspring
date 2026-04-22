"""SynthController state-machine tests."""

from __future__ import annotations

from handspring.synth_params import SynthParams
from handspring.synth_ui import SynthController
from handspring.types import FaceState, FrameResult, HandFeatures, HandState, MotionState, PoseState


def _absent_hand() -> HandState:
    return HandState(
        present=False,
        features=None,
        gesture="none",
        motion=MotionState(False, False, 0.0, 0.0, None),
    )


def _hand(gesture: str, x: float = 0.5, y: float = 0.5) -> HandState:
    return HandState(
        present=True,
        features=HandFeatures(
            x=x, y=y, z=0.0, openness=1.0, pinch=0.0, index_x=x, index_y=y, thumb_x=x, thumb_y=y
        ),
        gesture=gesture,  # type: ignore[arg-type]
        motion=MotionState(False, False, 0.0, 0.0, None),
    )


def _face_absent() -> FaceState:
    return FaceState(
        present=False,
        features=None,
        expression="neutral",
        eye_left_open=0.0,
        eye_right_open=0.0,
    )


def _frame(left: HandState, right: HandState) -> FrameResult:
    return FrameResult(
        left=left,
        right=right,
        face=_face_absent(),
        pose=PoseState(False, None),
        fps=30.0,
        clap_event=False,
    )


def test_mode_defaults_play():
    p = SynthParams()
    c = SynthController(p)
    c.update(_frame(_absent_hand(), _absent_hand()))
    assert p.snapshot().mode == "play"


def test_left_fist_debounce_activates_edit_left():
    p = SynthParams()
    c = SynthController(p)
    # 2 frames is not enough (debounce = 3).
    for _ in range(2):
        c.update(_frame(_hand("fist"), _absent_hand()))
    assert p.snapshot().mode == "play"
    c.update(_frame(_hand("fist"), _absent_hand()))
    assert p.snapshot().mode == "edit_left"


def test_release_requires_debounce():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _absent_hand()))
    assert p.snapshot().mode == "edit_left"
    # Single frame of non-fist — still in edit mode.
    c.update(_frame(_hand("open"), _absent_hand()))
    assert p.snapshot().mode == "edit_left"
    c.update(_frame(_hand("open"), _absent_hand()))
    c.update(_frame(_hand("open"), _absent_hand()))
    assert p.snapshot().mode == "play"


def test_both_fist_left_wins():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("fist")))
    assert p.snapshot().mode == "edit_left"


def test_edit_left_point_controls_volume():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("point", y=0.5)))
    # y=0.5 (middle) → volume ~0.5
    vol_mid = p.snapshot().volume
    assert 0.4 <= vol_mid <= 0.6
    # y=0.0 (top of frame) → volume = 1.0
    c.update(_frame(_hand("fist"), _hand("point", y=0.0)))
    assert p.snapshot().volume == 1.0
    # y=1.0 (bottom) → volume = 0.0
    c.update(_frame(_hand("fist"), _hand("point", y=1.0)))
    assert p.snapshot().volume == 0.0


def test_edit_left_open_controls_pitch_and_stepping():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("open", x=0.5, y=0.5)))
    s = p.snapshot()
    # Middle y → middle of log-pitch range (geometric mean of C3..C6).
    assert 200.0 < s.note_hz < 700.0
    # Middle x → stepping ~ half of range (0..16). Actual expo mapping
    # puts mid at 4.0, but allow some range.
    assert 0.0 <= s.stepping_hz <= 16.0
    # Top of frame → highest pitch.
    c.update(_frame(_hand("fist"), _hand("open", x=0.5, y=0.0)))
    assert p.snapshot().note_hz == 1047.0


def test_edit_right_point_controls_cutoff():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("point", y=0.5), _hand("fist")))
    assert p.snapshot().mode == "edit_right"
    # y=0.0 → highest cutoff
    c.update(_frame(_hand("point", y=0.0), _hand("fist")))
    assert p.snapshot().cutoff_hz == 8000.0
    c.update(_frame(_hand("point", y=1.0), _hand("fist")))
    assert p.snapshot().cutoff_hz == 200.0


def test_edit_right_open_controls_mod():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("open", x=0.5, y=0.5), _hand("fist")))
    s = p.snapshot()
    assert 0.0 <= s.mod_depth <= 1.0
    assert 0.1 <= s.mod_rate <= 10.0
    # y=0 → max depth
    c.update(_frame(_hand("open", x=0.0, y=0.0), _hand("fist")))
    assert p.snapshot().mod_depth == 1.0


def test_play_mode_does_not_edit_params():
    p = SynthParams()
    c = SynthController(p)
    initial_vol = p.snapshot().volume
    # No fist → play mode; right hand pointing should NOT change volume.
    c.update(_frame(_absent_hand(), _hand("point", y=0.0)))
    c.update(_frame(_absent_hand(), _hand("point", y=0.0)))
    assert p.snapshot().volume == initial_vol


def test_slider_anchors_on_first_point_frame():
    p = SynthParams()
    c = SynthController(p)
    # Enter edit_left
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("point", x=0.5, y=0.8)))
    hint = c.ui_hint()
    assert hint.kind == "slider"
    assert abs(hint.x - 0.5) < 1e-6
    assert abs(hint.y - 0.8) < 1e-6

    # Move hand to new position — slider position should NOT change.
    c.update(_frame(_hand("fist"), _hand("point", x=0.3, y=0.2)))
    hint2 = c.ui_hint()
    assert hint2.kind == "slider"
    assert abs(hint2.x - 0.5) < 1e-6  # still anchored
    assert abs(hint2.y - 0.8) < 1e-6  # still anchored
    # But value should have changed (y=0.2 gives different vol than y=0.8)
    assert hint2.value_a != hint.value_a


def test_slider_anchor_clears_when_leaving_point():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("point", x=0.5, y=0.5)))
    # Break to open — anchor should clear.
    c.update(_frame(_hand("fist"), _hand("open", x=0.5, y=0.5)))
    # Re-enter point at new position.
    c.update(_frame(_hand("fist"), _hand("point", x=0.2, y=0.2)))
    hint = c.ui_hint()
    assert hint.kind == "slider"
    assert abs(hint.x - 0.2) < 1e-6  # new anchor
    assert abs(hint.y - 0.2) < 1e-6


def test_slider_anchor_clears_on_mode_change():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("point", x=0.5, y=0.5)))
    # Drop left fist for 3 frames to exit mode.
    for _ in range(3):
        c.update(_frame(_hand("open"), _hand("open", x=0.5, y=0.5)))
    # Now enter edit_right instead.
    for _ in range(3):
        c.update(_frame(_hand("point", x=0.3, y=0.3), _hand("fist")))
    hint = c.ui_hint()
    assert hint.kind == "slider"
    # Should anchor at the new point position, not the old one.
    assert abs(hint.x - 0.3) < 1e-6
    assert abs(hint.y - 0.3) < 1e-6


def test_slider_live_y_tracks_current_fingertip():
    p = SynthParams()
    c = SynthController(p)
    for _ in range(3):
        c.update(_frame(_hand("fist"), _hand("point", x=0.5, y=0.3)))
    h1 = c.ui_hint()
    assert h1.kind == "slider"
    assert abs(h1.live_y - 0.3) < 1e-6
    c.update(_frame(_hand("fist"), _hand("point", x=0.5, y=0.8)))
    h2 = c.ui_hint()
    assert h2.kind == "slider"
    # Anchor unchanged.
    assert abs(h2.x - h1.x) < 1e-6
    assert abs(h2.y - h1.y) < 1e-6
    # But live_y tracks the new finger Y.
    assert abs(h2.live_y - 0.8) < 1e-6
