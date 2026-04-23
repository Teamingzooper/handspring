# Flick-Commit Radial Redesign

Rebuild the left-hand command picker from scratch. The current radial
(hold-to-activate + concentric/chip/mini-pinwheel sub-menus) is too slow,
too noisy, and requires too much hand travel. Replace with a flat
6-command directional flick: pinch to show, nudge to aim, release to
commit. Also smooth the two-hand "create window" gesture so the live
ghost rectangle doesn't jitter.

## Problems in the current system

1. **Hold delay**: 0.4s pinch-hold before the menu appears feels laggy.
2. **Noisy hover**: per-frame direction reads cause slices to flicker
   between neighbors, especially near the angular boundaries.
3. **Big hand travel**: reaching sub-chips required 10–25% of the camera
   frame of hand movement. With an 8-slice + sub-menu tree, aiming was
   a two-stage motion.
4. **Visual overload**: root ring + chip row + slice tip + connector
   line is a lot of UI for "pick one of 6 things."

## Design

### Interaction flow

1. **Pinch** left hand → overlay shows 6 labeled slices around the
   pinch origin **instantly** (no hold).
2. **Aim** → move hand a small amount in any direction. Whatever slice
   that direction points at highlights. Hand can drift — hysteresis
   prevents boundary flicker.
3. **Commit** → release pinch. If displacement from pinch origin is
   ≥ `flick_threshold` (default 0.03 in camera units), the highlighted
   command fires. Otherwise, cancel.
4. **Cancel** → release without moving far enough, or start pinching
   while the right hand is clicking / two-hand create is armed.

### The 6 commands (clockwise from 12 o'clock)

| Slot | Command | Rationale |
|-----:|---------|-----------|
| 0 | None (disable gestures) | Top = deliberate (needs clear up-move) |
| 1 | Settings | Opens the settings web UI |
| 2 | Mission Control | |
| 3 | New Finder window | Bottom = easy reach |
| 4 | Scroll mode | |
| 5 | Screenshot (whole screen) | |

Ordering is in `config.toml` (`radial_tree` key). Users can swap commands
or add custom ones via `command = "open -a X"` as before. The
`radial_tree` format stays — we just stop treating any `subs` field as
meaningful. Existing nested configs degrade gracefully: only the root
name is used, subs are ignored.

### Config changes

- **Remove** `radial.hold_seconds`, `radial.sub_threshold`,
  `radial.sub_mini_inner`, `radial.inner_radius` (no longer used).
- **Add** `radial.flick_threshold = 0.03` (camera-space minimum
  displacement from pinch origin to count as a commit).
- **Add** `radial.angular_hysteresis = 0.15` (a slice's neighbor must
  exceed the current slice's angular share by this fraction to take
  over, preventing boundary flicker).
- Defaults only; users can edit via `config.toml` or the settings UI.

### Selection logic

```
on pinch start:
    origin = hand_position
    highlighted = None
    show overlay

per frame while pinching:
    dx = hand_x - origin.x
    dy = hand_y - origin.y
    dist = sqrt(dx² + dy²)
    if dist < flick_threshold / 2:
        highlighted = None               # in dead zone
    else:
        # Angle from origin in [0, 2π), 0 = up, clockwise.
        angle = clockwise_from_up(dx, dy)
        if highlighted is None:
            highlighted = int(angle / slice_size)
        else:
            # Hysteresis: the current slice's angular range is expanded
            # by (1 + hysteresis) on each side. The hand's angle must
            # fall outside the *expanded* range before a new slice can
            # take over.
            cur_center = (highlighted + 0.5) * slice_size
            half = slice_size / 2 * (1 + hysteresis)
            if abs(wrap_pi(angle - cur_center)) > half:
                highlighted = int(angle / slice_size)

on pinch release:
    if highlighted is not None and final_dist >= flick_threshold:
        commit(highlighted)
```

### Visual

- 6 labels arranged around the pinch origin at ~180 px screen radius.
- Current highlight: label pops in size + color (green), background
  fills a pie wedge behind it.
- No inner ring, no sub-ring, no chips, no connector lines.
- A faint centered dot marks the pinch origin so the user knows where
  "neutral" is.
- The `flick_threshold` is drawn as a thin circle around the origin —
  visual cue that "release outside this = commit."

### Smooth create-window gesture

The two-hand pinch + pull-apart create gesture stays but gets an EMA
smoother on the ghost rectangle so the live preview doesn't jitter from
MediaPipe landmark noise:

- Add `create.smoothing = 0.35` to config (same feel as cursor).
- Track `smooth_left` and `smooth_right` fingertip positions inside
  `_CreateState`; update each frame with
  `smooth = α * raw + (1-α) * smooth`.
- Use smoothed values for the ghost rect and for the commit diagonal
  check, so a brief jitter spike can't spuriously commit or cancel.
- Reset smoothing state when the gesture disarms, so the next arm
  starts from the raw position (no interpolation from a stale point).

No behavior change beyond "the rect doesn't shake."

### What stays

- Right-hand cursor + pinch-click behavior.
- Two-hand create gesture (just smoother).
- Both-fist 5s failsafe.
- Settings web UI + config store + file watcher.
- OSC output.
- All OS control primitives in `os_control.py`.

### What's removed

- `radial.hold_seconds`, `radial.sub_threshold`, `radial.inner_radius`,
  `radial.sub_mini_inner`.
- Sub-chip layout + rendering (`compute_sub_layout`, chip constants,
  overlay chip drawing).
- Root-lock-on-crossover logic (no crossover any more).
- "More" root item plus Settings/Reload/Quit sub-items; Settings is now
  a flat root leaf, Reload happens automatically via the mtime watcher,
  Quit via Ctrl+C or the app menu.
- Window tile subs, Desktop L/R subs, Screenshot variants (all
  accessible via Settings UI / custom `command` radial items if users
  want them back).

### Testing

- `test_flick_commit_picks_slice_on_release` — move in a direction,
  release, verify the matching slice's command fires.
- `test_flick_cancels_below_threshold` — release at origin, no fire.
- `test_flick_hysteresis_prevents_flicker` — small angular wobble near
  a boundary keeps the originally-picked slice highlighted.
- `test_flick_no_hold_delay` — menu is active from the first pinching
  frame; no dwell required.
- `test_create_ghost_rect_smooths_jitter` — inject jittery fingertip
  positions, verify the reported ghost rect is smoother than raw.
- Remove old hold-activation / sub-chip / mini-pinwheel tests.

## Files touched

- `src/handspring/config.py` — update `RadialConfig` fields, add
  `CreateConfig.smoothing`.
- `src/handspring/desktop_controller.py` — rewrite `_handle_radial`
  (no hold, no subs, flick + hysteresis commit), add EMA to
  `_CreateState`, drop `compute_sub_layout` and chip constants, simplify
  `_commit_radial` (all commands are leaves), update `radial_state()`
  payload (drop `hovered_sub`).
- `src/handspring/overlay.py` — simplify `_draw_radial_tree`: pinch dot
  + threshold circle + 6 labeled wedges, no countdown arc, no sub
  chips.
- `src/handspring/__main__.py` — update the radial payload shape
  passed to `overlay.set_state`.
- `tests/test_desktop_controller.py` — replace old hover/sub tests
  with flick-commit tests, add create smoothing test.
- `README.md` — update Desktop mode: radial menu section + default
  `radial_tree` example.

## Non-goals

- Adding voice, alternative input modalities, keyboard fallback.
- Recovering every removed sub (tile ops, desktops L/R, screenshot
  variants). Users can add them as custom radial items with commands
  in `config.toml` if they want them back as leaves.
- Changing the right-hand cursor or failsafe behavior.
