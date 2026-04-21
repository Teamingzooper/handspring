"""JARVIS mode: transparent-window creation, grab/drag, tap."""

from __future__ import annotations

from dataclasses import dataclass, replace

from handspring.types import FrameResult, HandState, Side

_MAX_WINDOWS_DEFAULT = 8
_NUM_COLORS = 3
# Palette (BGR tuples for OpenCV).
WINDOW_COLORS = [
    (255, 180, 77),  # blue  #4DB4FF
    (100, 230, 120),  # green
    (220, 110, 210),  # purple
]


@dataclass(frozen=True)
class Window:
    """Rectangle in normalized (0..1) coordinates."""

    id: int
    x: float
    y: float
    width: float
    height: float
    z: int  # z-order index (higher = on top)
    color_idx: int  # index into WINDOW_COLORS

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


class WindowManager:
    def __init__(self, *, max_windows: int = _MAX_WINDOWS_DEFAULT) -> None:
        self._max = max_windows
        self._next_id = 1
        self._next_z = 1
        self._windows: list[Window] = []  # creation order

    def create(self, *, x: float, y: float, width: float, height: float) -> Window:
        w = Window(
            id=self._next_id,
            x=x,
            y=y,
            width=width,
            height=height,
            z=self._next_z,
            color_idx=0,
        )
        self._next_id += 1
        self._next_z += 1
        self._windows.append(w)
        # Evict FIFO when exceeding the cap.
        while len(self._windows) > self._max:
            self._windows.pop(0)
        return w

    def windows(self) -> list[Window]:
        """Return windows in z-order (bottom → top)."""
        return sorted(self._windows, key=lambda w: w.z)

    def get(self, window_id: int) -> Window | None:
        for w in self._windows:
            if w.id == window_id:
                return w
        return None

    def topmost_at(self, px: float, py: float) -> Window | None:
        hit = [w for w in self._windows if w.contains(px, py)]
        if not hit:
            return None
        return max(hit, key=lambda w: w.z)

    def promote(self, window_id: int) -> None:
        w = self.get(window_id)
        if w is None:
            return
        new_w = replace(w, z=self._next_z)
        self._next_z += 1
        self._replace(new_w)

    def move(self, window_id: int, *, dx: float, dy: float) -> None:
        w = self.get(window_id)
        if w is None:
            return
        nx = max(0.0, min(1.0 - w.width, w.x + dx))
        ny = max(0.0, min(1.0 - w.height, w.y + dy))
        self._replace(replace(w, x=nx, y=ny))

    def cycle_color(self, window_id: int) -> None:
        w = self.get(window_id)
        if w is None:
            return
        self._replace(replace(w, color_idx=(w.color_idx + 1) % _NUM_COLORS))

    def _replace(self, new_w: Window) -> None:
        for i, w in enumerate(self._windows):
            if w.id == new_w.id:
                self._windows[i] = new_w
                return


# ---------------------------------------------------------------------------
# Gesture state machines
# ---------------------------------------------------------------------------

_TAP_HOVER_FRAMES = 5  # 5 frames @ 30fps ≈ 150 ms
_TAP_COOLDOWN_SECONDS = 0.4
_MIN_WINDOW_DIAGONAL = 0.1
_PINCH_ON_THRESHOLD = 0.85


@dataclass
class _GrabState:
    window_id: int
    side: Side
    last_palm_x: float
    last_palm_y: float


@dataclass
class _CreateState:
    start_left: tuple[float, float]
    start_right: tuple[float, float]


class JarvisController:
    def __init__(self) -> None:
        self.manager = WindowManager()
        self._grab: _GrabState | None = None
        self._create: _CreateState | None = None
        self._hover_frames: dict[Side, int] = {"left": 0, "right": 0}
        self._last_hover_window: dict[Side, int | None] = {"left": None, "right": None}
        self._tap_cooldown_by_window: dict[int, float] = {}
        self._last_tap_id: int | None = None
        self._tap_set_at_frame: int = -1
        self._frame_count: int = 0
        self._events_out: list[tuple[str, int]] = []

    # ---- Events consumed by __main__ to emit OSC ----

    def pop_events(self) -> list[tuple[str, int]]:
        """Return events accumulated during update() and clear the buffer.

        Event tuples: (kind, window_id). Kinds: "created" | "tap".
        """
        out = self._events_out
        self._events_out = []
        return out

    def last_tap_id(self) -> int | None:
        return self._last_tap_id

    # ---- Main update loop ----

    def update(self, frame: FrameResult, now: float | None = None) -> None:
        # Use a synthetic monotonic timestamp if caller didn't pass one.
        if now is None:
            now = 0.0

        self._frame_count += 1
        # Clear last_tap_id if it was set in a prior update (transient: one frame).
        if self._tap_set_at_frame != self._frame_count - 1:
            self._last_tap_id = None

        self._handle_create(frame, now)
        self._handle_grab(frame, now)
        self._handle_tap(frame, now)

    # ---- 1. Create: both-hand pinch + pull apart ----

    def _handle_create(self, frame: FrameResult, _now: float) -> None:
        left = frame.left
        right = frame.right
        both_pinching = (
            left.present
            and right.present
            and left.features is not None
            and right.features is not None
            and left.features.pinch >= _PINCH_ON_THRESHOLD
            and right.features.pinch >= _PINCH_ON_THRESHOLD
        )
        if both_pinching:
            if self._create is None:
                assert left.features is not None
                assert right.features is not None
                self._create = _CreateState(
                    start_left=(left.features.x, left.features.y),
                    start_right=(right.features.x, right.features.y),
                )
            return

        # If we were in a create gesture and now lost pinch on either hand, commit.
        if self._create is not None and left.features is not None and right.features is not None:
            # Use current (at-release) positions as the rectangle corners.
            x1, y1 = left.features.x, left.features.y
            x2, y2 = right.features.x, right.features.y
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            w = x_max - x_min
            h = y_max - y_min
            diag = (w * w + h * h) ** 0.5
            if diag >= _MIN_WINDOW_DIAGONAL:
                # Clamp aspect ratio to [0.5, 2.0]
                if w > 0 and h > 0:
                    ar = w / h
                    if ar < 0.5:
                        h = w / 0.5
                    elif ar > 2.0:
                        h = w / 2.0
                created = self.manager.create(x=x_min, y=y_min, width=w, height=h)
                self._events_out.append(("created", created.id))
        self._create = None

    # ---- 2. Grab/drag: open → fist while palm is over a window ----

    def _handle_grab(self, frame: FrameResult, _now: float) -> None:
        if self._grab is None:
            # Look for either hand transitioning to fist while over a window.
            for side in ("left", "right"):
                state: HandState = getattr(frame, side)
                if not state.present or state.features is None:
                    continue
                if state.gesture != "fist":
                    continue
                target = self.manager.topmost_at(state.features.x, state.features.y)
                if target is None:
                    continue
                self.manager.promote(target.id)
                self._grab = _GrabState(
                    window_id=target.id,
                    side=side,  # type: ignore[arg-type]
                    last_palm_x=state.features.x,
                    last_palm_y=state.features.y,
                )
                return
            return

        # Already grabbing: track motion, release on open.
        grab = self._grab
        state = getattr(frame, grab.side)
        if not state.present or state.features is None or state.gesture != "fist":
            self._grab = None
            return
        dx = state.features.x - grab.last_palm_x
        dy = state.features.y - grab.last_palm_y
        self.manager.move(grab.window_id, dx=dx, dy=dy)
        self._grab = _GrabState(
            window_id=grab.window_id,
            side=grab.side,
            last_palm_x=state.features.x,
            last_palm_y=state.features.y,
        )

    # ---- 3. Tap: point + hover on window for _TAP_HOVER_FRAMES frames ----

    def _handle_tap(self, frame: FrameResult, now: float) -> None:
        for side in ("left", "right"):
            state: HandState = getattr(frame, side)
            if not state.present or state.features is None or state.gesture != "point":
                self._hover_frames[side] = 0  # type: ignore[index]
                self._last_hover_window[side] = None  # type: ignore[index]
                continue
            target = self.manager.topmost_at(state.features.index_x, state.features.index_y)
            if target is None:
                self._hover_frames[side] = 0  # type: ignore[index]
                self._last_hover_window[side] = None  # type: ignore[index]
                continue
            if self._last_hover_window[side] == target.id:  # type: ignore[index]
                self._hover_frames[side] += 1  # type: ignore[index]
            else:
                self._hover_frames[side] = 1  # type: ignore[index]
                self._last_hover_window[side] = target.id  # type: ignore[index]

            if self._hover_frames[side] >= _TAP_HOVER_FRAMES:  # type: ignore[index]
                last_tap = self._tap_cooldown_by_window.get(target.id, -1.0)
                if now - last_tap >= _TAP_COOLDOWN_SECONDS:
                    self.manager.cycle_color(target.id)
                    self.manager.promote(target.id)
                    self._tap_cooldown_by_window[target.id] = now
                    self._events_out.append(("tap", target.id))
                    self._last_tap_id = target.id
                    self._tap_set_at_frame = self._frame_count
                # Reset hover counter so a continuous hover doesn't retap until cooldown.
                self._hover_frames[side] = 0  # type: ignore[index]
