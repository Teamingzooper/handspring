"""SynthParams tests — thread-safe snapshot semantics."""

from __future__ import annotations

from handspring.synth_params import SynthParams


def test_defaults():
    p = SynthParams()
    s = p.snapshot()
    assert s.volume == 0.4
    assert abs(s.note_hz - 440.0) < 1e-6
    assert s.stepping_hz == 0.0
    assert s.cutoff_hz == 3000.0
    assert s.mod_depth == 0.0
    assert s.mod_rate == 1.0
    assert s.mode == "play"


def test_set_volume_clamps():
    p = SynthParams()
    p.set_volume(1.5)
    assert p.snapshot().volume == 1.0
    p.set_volume(-0.2)
    assert p.snapshot().volume == 0.0


def test_set_note_clamps_to_range():
    p = SynthParams()
    p.set_note_hz(50.0)
    assert p.snapshot().note_hz == 131.0
    p.set_note_hz(10_000.0)
    assert p.snapshot().note_hz == 1047.0


def test_set_stepping_clamps():
    p = SynthParams()
    p.set_stepping_hz(-1)
    assert p.snapshot().stepping_hz == 0.0
    p.set_stepping_hz(99)
    assert p.snapshot().stepping_hz == 16.0


def test_set_cutoff_clamps_exponential():
    p = SynthParams()
    p.set_cutoff_hz(100.0)
    assert p.snapshot().cutoff_hz == 200.0
    p.set_cutoff_hz(50000.0)
    assert p.snapshot().cutoff_hz == 8000.0


def test_set_mode_updates_snapshot():
    p = SynthParams()
    p.set_mode("edit_left")
    assert p.snapshot().mode == "edit_left"


def test_snapshot_is_immutable():
    p = SynthParams()
    s = p.snapshot()
    # Dataclass is frozen; assigning should raise.
    import pytest

    with pytest.raises(AttributeError):
        s.volume = 0.9  # type: ignore[misc]
