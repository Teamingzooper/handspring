# handspring v0.4.0 — Gesture-Driven In-Process Synth

**Date:** 2026-04-21
**Status:** Approved
**Parent:** v0.3.0

## Purpose

Put a playable synthesizer directly inside handspring, driven by hand gestures. Fist on either hand enters an edit mode; the non-fist hand's shape and position edit synth parameters live. A small synth panel and floating sliders in the preview window show the current state.

## In scope

### Synth engine (in-process audio)

- Monophonic synth: saw oscillator → one-pole lowpass → amp envelope → tremolo (amplitude mod).
- Parameters (all continuous, one-pole smoothed to avoid zipper noise):
  - `volume` 0..1
  - `note_hz` 131..1047 (C3..C6)
  - `stepping_hz` 0..16 — envelope retrigger rate; 0 = sustained, > 0 = rhythmic pulses (the "buildup" sound).
  - `cutoff_hz` 200..8000 (exponential mapping)
  - `mod_depth` 0..1
  - `mod_rate` 0.1..10 Hz (LFO frequency for tremolo)
- Runs on `sounddevice.OutputStream` at 48 kHz, blocksize 256, mono.
- `sounddevice` promoted from optional dev-dep → runtime dep.
- CLI flag `--no-synth` disables audio output (parameters still track for OSC).

### Gesture-driven editing

Fist-on-hand → edit mode for that hand's "side" of the synth. The non-fist hand edits parameters.

**Mode 1 — Left hand = FIST ("EDIT L"):**

| Right hand | Effect |
|---|---|
| `point` | Volume slider drawn at right index fingertip. Right hand Y → volume. |
| `open` | Right Y → note pitch (inverted: high = high pitch). Right X → stepping rate. |
| anything else | No edit; synth plays current params. |

**Mode 2 — Right hand = FIST ("EDIT R"):**

| Left hand | Effect |
|---|---|
| `point` | Cutoff slider at left index fingertip. Left hand Y → cutoff. |
| `open` | Left Y → mod depth. Left X → mod rate. |
| anything else | No edit. |

**Tie-break:** if both hands are in fist simultaneously, left wins (Mode 1 takes priority).

**Neither fist:** PLAY mode. Synth keeps playing current params.

**Fist debounce:** the gesture must classify as `"fist"` for ≥ 3 consecutive frames (~100 ms) before the mode activates, and for ≥ 3 consecutive non-fist frames before deactivation. Prevents classifier flicker from toggling the mode.

### Preview overlay

**Synth panel** — always visible lower-left corner, compact text block:

```
─ SYNTH ─
vol: 0.60
note: A4 (440 Hz)
step: 0.0 Hz
cutoff: 3500 Hz
mod: 0.00 @ 1.0 Hz
mode: PLAY
```

The `mode` line shows `PLAY` / `EDIT L` / `EDIT R`. In edit mode the panel tints the edited parameter rows green.

**Floating slider** — when edit mode is active and the non-fist hand is pointing:
- A vertical slider draws beside the index fingertip (offset so it's readable).
- Slider shows: parameter name, current value, bar-fill proportional to the normalized value (0..1).
- The slider's value follows the Y position of the fingertip.

**XY controller** — when edit mode is active and the non-fist hand is open:
- Two crosshair lines (horizontal and vertical) draw through the palm center at normalized screen position (palm.x, palm.y).
- Labels at each axis end show which parameter that axis controls, plus its current value.

### OSC additions (always emitted; observable externally)

Per frame (continuous):
- `/synth/volume` float 0..1
- `/synth/note_hz` float
- `/synth/stepping_hz` float
- `/synth/cutoff_hz` float
- `/synth/mod_depth` float
- `/synth/mod_rate` float
- `/synth/mode` string: `"play" | "edit_left" | "edit_right"` — emitted on change (like gesture).

## Out of scope

- Polyphony (monophonic only)
- Multiple waveforms or sample-based voices
- Preset save/load
- Envelope sliders for attack/release/sustain
- Reverb/delay/stereo panning
- Filter resonance (simple one-pole lowpass only)
- Gesture-based synth start/stop (synth always runs while app is up; `--no-synth` disables at boot)
- Per-note velocity
- Pinch visibility fix (still punted; fist-mode label in the panel serves as the primary gesture-state feedback now)

## Autonomous design decisions

1. **Saw wave** — warm harmonic content; works well with a lowpass. Simple integer-phase implementation.
2. **One-pole lowpass** — `y[n] = (1 - α) y[n-1] + α x[n]`, α = cutoff / (cutoff + sample_rate / (2π)). Simple, non-resonant, fast.
3. **Tremolo (not vibrato)** for modulation. Amp-mod is more audible than pitch-mod for most patches and avoids pitch smearing.
4. **Stepping retriggers the amp envelope** at the stepping rate. Between retriggers, the envelope decays naturally.
5. **Smoothing coefficient:** 0.01 per sample (~22 ms time constant). Avoids audible discontinuities when hand jitters.
6. **Fist debounce:** 3 frames in and out.
7. **Default parameter values on startup:** vol=0.4, note=A4 (440 Hz), stepping=0, cutoff=3000 Hz, mod_depth=0, mod_rate=1 Hz. Synth is audible immediately; users can edit from there.
8. **Thread safety:** `SynthParams` class wraps all floats. Writes from main thread acquire `threading.Lock`; the audio callback takes a `snapshot()` under the same lock at the top of each block. Per-block lock is fine for block sizes ≥ 128 samples (1-time hit per ~5 ms).
9. **Pitch Y-axis inversion:** hand at top of frame → high pitch; bottom → low pitch. Matches how piano keys "feel" laid out vertically in intuition.
10. **Note display:** standard note names (C3, C#3, …) with octave number; Hz in parentheses.
11. **Tie-break for both fists:** left wins. Simpler than building a "both" mode.
12. **`--no-synth` behavior:** disables audio output AND the `/synth/*` OSC emissions. Panel shows "synth: off" instead of the full readout.

## Acceptance criteria

1. `pip install -e '.[dev]'` now installs `sounddevice` (previously optional). macOS + Linux.
2. `python -m handspring` boots with an audible A4 sine-like tone (saw filtered).
3. Preview window shows the `─ SYNTH ─` panel in the lower-left.
4. Making a LEFT fist for ~100 ms: panel's `mode` flips to `EDIT L`.
5. While in EDIT L + right hand pointing upward: a vertical slider appears at the right index fingertip labeled "volume". Moving the right hand up/down changes the slider fill AND the audible volume.
6. While in EDIT L + right hand open: moving right hand up raises pitch; moving right/left changes the stepping rate. Pitch display updates (e.g., `note: C5 (523 Hz)`) and stepping value updates.
7. Stepping > 0: you hear the synth pulse at that rate instead of sustaining.
8. Making a RIGHT fist: mode flips to `EDIT R`; left-hand controls now apply to cutoff / mod.
9. `/synth/volume`, `/synth/note_hz`, etc. stream via OSC.
10. `--no-synth` starts the app without audio; synth panel shows "SYNTH: off".
11. All v0.3.0 acceptance criteria still pass (overlays, motion, expressions, hand shapes).
12. `pytest` ≥ 85 tests pass. `ruff check`, `ruff format --check`, `mypy src/` all clean.

## Known tradeoffs

- **Audio and MediaPipe share a Python process.** Under heavy MediaPipe load the audio callback may miss blocks, producing brief dropouts. Acceptable for a tool; not studio-grade.
- **One-pole lowpass has no resonance.** Can't self-oscillate; sounds mild. Upgradeable to a state-variable filter later if needed.
- **Fist classifier ambiguity** (thumb curled vs extended) means some natural fist shapes don't register. We already lowered the thumb sensitivity in v0.1.0; if mode activation proves unreliable, we'd need a more robust closed-hand detector.
- **Volume control while pointing is inherently inverted** (finger up = more volume). Expected / intuitive but worth noting.
