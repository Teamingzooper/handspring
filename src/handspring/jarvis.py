"""JARVIS mode: transparent-window creation, grab/drag, tap."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
