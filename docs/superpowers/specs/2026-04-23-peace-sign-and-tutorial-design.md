# Peace Sign Gesture + First-Run Tutorial

Add two independent-but-related features:

1. **Peace sign → Show Desktop.** Holding a peace sign (index + middle
   extended, ring + pinky curled) for 0.3s triggers macOS Show Desktop.
2. **First-run calibration tutorial.** On first launch (no config file
   yet), walk the user through 7 gestures in the preview window,
   capturing per-user thresholds as we go.

Both ship together as one bundled PR.

## Peace sign → Show Desktop

### Detection

Add a new gesture label `"peace"` to `handspring.gestures.classify_hand`
alongside the existing `fist` / `open` / `point` / `pinch` set.

Classification rules (using existing finger-extended features from
`features.py`):

- Index finger: extended
- Middle finger: extended
- Ring finger: curled
- Pinky finger: curled
- Thumb: either state (don't discriminate)

If a hand matches this pose and doesn't already match `pinch` (which
takes precedence), label it `"peace"`.

### Action

Emit macOS F11 keycode via `os_control.show_desktop()`, a new helper:

```python
def show_desktop() -> None:
    """Send F11 (default Show Desktop binding)."""
    if not _MAC:
        return
    import subprocess
    subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to key code 103'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
```

Keycode 103 = F11 on macOS. If the user has rebound Show Desktop to a
different key, they override via `config.toml`:

```toml
[gestures]
peace_command = "osascript -e 'tell app \"System Events\" to key code 103'"
```

### Trigger logic

In `desktop_controller.py`, add a `_PeaceState` with `start: float | None`
tracking how long a peace pose has been held. Fire the action when held
≥ `peace_hold_seconds` (default 0.3s). Once fired, require the pose to
drop (any frame where neither hand shows peace) before it can fire
again. Suppress during failsafe-disabled mode. Either hand counts.

### Config additions

```python
@dataclass(frozen=True)
class GesturesConfig:
    peace_hold_seconds: float = 0.3
    peace_command: str = ""   # empty = use built-in show_desktop()
```

Plug into `Config` as `gestures: GesturesConfig`.

### Non-goals

- Per-hand peace bindings (both hands do the same thing for v1).
- Peace sign with pinch modifier (leave room for later).
- Cancel-peace-with-quick-release timing. A simple edge-triggered fire is enough.

## First-Run Tutorial

### Trigger

On `python -m handspring` startup:

- If `~/.config/handspring/config.toml` does **not** exist → run the
  tutorial before entering the main loop.
- `--skip-tutorial` → always skip, proceed to main loop.
- `--tutorial` → force-run the tutorial even if config exists.
- The Settings UI grows a "Replay tutorial" button that writes a flag
  the next main-loop start reads and re-runs the tutorial.

### Rendering

The tutorial runs **inside the existing OpenCV preview window**. It uses
`cv2.putText` and `cv2.rectangle` for overlays. No new window type, no
web UI. The main-loop camera read + tracker pipeline runs exactly as in
normal mode; the tutorial is a thin state machine that consumes
`FrameResult`s and draws guidance on top.

Layout: a semi-opaque bar across the top of the preview with the current
step's title + instruction, a status line at the bottom (detected
state), and a progress indicator (e.g., `Step 3 / 7`).

### Steps

Each step captures a sample when it passes, then auto-advances after a
short confirmation pause (~0.8s).

1. **Show your right hand.** Wait until `frame.right.present` is True
   for 20 consecutive frames. Displays a green ring around the detected
   hand.

2. **Point with your right index.** Wait for `frame.right.gesture ==
   "point"` held 15 frames. Captures `hand_size =
   dist(wrist, index_tip)` as a calibration sample.

3. **Move the cursor around.** Wait until the right-hand position has
   traveled a cumulative 0.3 camera units. Confirms cursor tracking
   feels right.

4. **Pinch your right index + thumb.** Wait for `is_pinching(right)`
   True for 8 consecutive frames. Records the user's pinch distance
   range — specifically the min distance observed while pinching, used
   to auto-tune the pinch threshold.

5. **Show your left hand and pinch it.** Same as step 4 but for the left
   hand. Confirms the radial trigger works.

6. **Flick toward the Mission slice.** Wait for a successful radial
   commit of `Mission` (or any slice). Uses the real `DesktopController`
   path but intercepts the commit so we don't actually fire Mission
   Control mid-tutorial.

7. **Make a peace sign.** Wait for `frame.<side>.gesture == "peace"`
   held 10 frames (tests the new detection).

Each step has a 30-second timeout. If the user times out on a step,
offer Skip (arrow-key or any pinch) to move on.

### Calibration output

After all steps complete, write `~/.config/handspring/config.toml` with:

- `create.entry_distance` and `radial.flick_threshold` scaled by the
  captured `hand_size` (relative to the baseline 0.4 camera-space
  reference hand). Larger hand = larger thresholds proportionally.
- All other fields at their dataclass defaults.

If the user skips with Esc or `q` before completion, write defaults
only (no calibration). Either way, the file now exists, so future
launches skip the tutorial.

### Architecture

New module: `handspring.tutorial`.

```python
class Tutorial:
    def __init__(self, tracker: Tracker) -> None: ...
    def run(self, cap: cv2.VideoCapture) -> CalibrationResult: ...
    # Returns CalibrationResult with captured values (hand_size, etc.)
    # or None if skipped.
```

`__main__.py` calls `Tutorial(tracker).run(cap)` before the main loop
when the trigger conditions are met, then writes the calibrated config
via `ConfigStore` and continues.

The tutorial MUST close its OpenCV windows cleanly so the main loop can
create its own without conflict.

### CLI additions

```
python -m handspring [existing options]
  --skip-tutorial          skip first-run tutorial if it would run
  --tutorial               force the tutorial even if config exists
```

### Error handling

- Camera can't open during tutorial: abort tutorial, print error, exit
  non-zero. Don't write defaults in this case — the user should fix
  camera setup.
- User presses Esc during a step: confirm "Skip remaining steps? (y/n)".
  `y` writes defaults, exits tutorial cleanly. `n` continues.
- Any step's timeout (30s): advance with a warning ("step skipped, using
  default"). Don't capture a calibration sample for that step.

## Files touched

- `src/handspring/gestures.py` — add `"peace"` classifier.
- `src/handspring/os_control.py` — add `show_desktop()`.
- `src/handspring/config.py` — add `GesturesConfig`, wire into `Config`.
- `src/handspring/desktop_controller.py` — add `_PeaceState`, fire logic.
- `src/handspring/tutorial.py` — **new.** Tutorial state machine + renderer.
- `src/handspring/__main__.py` — `--skip-tutorial` / `--tutorial` flags,
  trigger logic before main loop.
- `src/handspring/settings_server.py` — add "Replay tutorial" button +
  POST `/api/replay-tutorial` that writes a flag file; main loop reads
  and deletes the flag on next start.
- `tests/test_gestures.py` — peace classifier tests.
- `tests/test_desktop_controller.py` — peace-hold fires action test.
- `tests/test_tutorial.py` — **new.** Drives the tutorial state machine
  with synthetic `FrameResult`s; no camera, no cv2 window.
- `README.md` — brief tutorial + peace-sign doc.

## Testing strategy

**Gesture classifier:** unit tests with synthetic `HandLandmarks` for
each of the 5 finger states, confirm the "peace" label fires only for
the correct shape. Confirm pinch takes precedence if both match.

**Peace fire logic:** inject a held-peace frame stream into
`DesktopController`, confirm `show_desktop` is called exactly once, and
the re-fire requires a drop-and-recover.

**Tutorial:** the `Tutorial` class is split into a pure-logic
`TutorialStateMachine` (frame-in, state-out, no cv2) and a thin
`TutorialRenderer` (draws the state). Tests drive the state machine
with synthetic frames and assert transitions. The renderer is
untested (manual-test only — pyobjc/cv2 window) but kept small enough
to be obvious.

## Non-goals

- Voice or audio cues in the tutorial.
- Saving tutorial progress and resuming later.
- Tutorial localization.
- Sharing calibration across machines.
- Detecting additional poses (rock-on, thumbs-up) — peace is the only
  new one.
