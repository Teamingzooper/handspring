"""App-level mode toggle driven by sustained mouth-open."""

from __future__ import annotations

from handspring.app_mode import AppModeController


def _step(ctrl: AppModeController, mouth: float, n: int) -> None:
    for _ in range(n):
        ctrl.update(mouth_open=mouth, face_present=True, now=0.0 if n == 0 else None)


def test_default_is_synth():
    c = AppModeController()
    assert c.mode() == "synth"


def test_sustained_mouth_open_toggles_to_jarvis():
    c = AppModeController()
    # 14 frames is not enough (need 15).
    for i in range(14):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    assert c.mode() == "synth"
    c.update(mouth_open=0.8, face_present=True, now=14 * 0.033)
    assert c.mode() == "jarvis"


def test_second_toggle_respects_cooldown():
    c = AppModeController()
    # First toggle.
    for i in range(15):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    assert c.mode() == "jarvis"
    # Immediately hold mouth open: should NOT toggle back (cooldown active).
    for i in range(15):
        c.update(mouth_open=0.8, face_present=True, now=(15 + i) * 0.033)
    assert c.mode() == "jarvis"
    # After 1.5 s cooldown: toggle allowed.
    for i in range(15):
        c.update(mouth_open=0.8, face_present=True, now=2.0 + i * 0.033)
    assert c.mode() == "synth"


def test_brief_mouth_open_does_not_toggle():
    c = AppModeController()
    # 10 frames of open, then closed — should not toggle.
    for i in range(10):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    for i in range(20):
        c.update(mouth_open=0.1, face_present=True, now=(10 + i) * 0.033)
    assert c.mode() == "synth"


def test_face_absent_resets_counter():
    c = AppModeController()
    for i in range(10):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    # Face gone — counter resets.
    c.update(mouth_open=0.8, face_present=False, now=10 * 0.033)
    for i in range(10):
        c.update(mouth_open=0.8, face_present=True, now=(11 + i) * 0.033)
    assert c.mode() == "synth"


def test_mode_transition_callback():
    transitions = []
    c = AppModeController(on_change=lambda mode: transitions.append(mode))
    for i in range(15):
        c.update(mouth_open=0.8, face_present=True, now=i * 0.033)
    assert transitions == ["jarvis"]
