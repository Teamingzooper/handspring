"""Persistent, live-reloadable user config for handspring.

The config lives at ``~/.config/handspring/config.toml`` by default. On first
run, defaults are written there so users have a file to edit. A background
poller watches the file's mtime and atomically swaps the in-memory snapshot
when it changes, so both hand-edits and UI-driven writes take effect without
a restart.

Consumers read via ``ConfigStore.get()`` each frame. The returned ``Config``
is an immutable snapshot — swap-in-place means no consumer ever sees a
half-written state.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


# ---- Dataclasses ---------------------------------------------------------


@dataclass(frozen=True)
class RadialItem:
    """A single radial menu entry.

    ``command`` is an optional shell command. If set, selecting this leaf
    runs the command (macOS: via ``subprocess.Popen`` with shell=False when
    possible). If unset, the leaf is a built-in action (matched by name in
    desktop_controller).
    """

    name: str
    subs: tuple[str, ...] = ()
    command: str | None = None


@dataclass(frozen=True)
class CursorConfig:
    smoothing: float = 0.35
    inset: float = 0.08


@dataclass(frozen=True)
class RadialConfig:
    # Minimum camera-space displacement from pinch origin for a release to
    # count as a commit. Below = cancel (deliberate "no-op" pinch).
    flick_threshold: float = 0.03
    # Fractional expansion of the current slice's angular range. Higher =
    # more sticky, less flicker near slice boundaries.
    angular_hysteresis: float = 0.15


@dataclass(frozen=True)
class ScrollConfig:
    deadzone: float = 0.12
    max_pixels: int = 30


@dataclass(frozen=True)
class CreateConfig:
    entry_distance: float = 0.08
    min_diagonal: float = 0.15
    # EMA factor for the ghost rect corners (same idea as cursor smoothing).
    smoothing: float = 0.35


@dataclass(frozen=True)
class FailsafeConfig:
    hold_seconds: float = 5.0


@dataclass(frozen=True)
class OverlayConfig:
    enabled: bool = True
    scale: float = 1.0


@dataclass(frozen=True)
class ColorsConfig:
    # RGB triples in 0..255. Consumers may convert to whatever format they need.
    radial_highlight: tuple[int, int, int] = (136, 255, 0)
    radial_outline: tuple[int, int, int] = (200, 200, 200)
    cursor_dot: tuple[int, int, int] = (136, 255, 0)


@dataclass(frozen=True)
class FeaturesConfig:
    tiling: bool = True
    spaces: bool = True
    mission_control: bool = True
    screenshots: bool = True


@dataclass(frozen=True)
class ServerConfig:
    web_port: int = 8765
    settings_port: int = 8766


@dataclass(frozen=True)
class GesturesConfig:
    peace_hold_seconds: float = 0.3
    peace_command: str = ""  # empty = built-in show_desktop()


@dataclass(frozen=True)
class FaceConfig:
    """Face-tracking features.

    ``gate_gestures`` enables the "safety net": when the user's face isn't
    present or isn't roughly facing the camera, cursor moves, clicks, and
    the radial are all suppressed. Prevents accidental fires when the user
    turns to talk to someone or reaches off-screen.

    ``mouth_open_*`` controls the "mouth wide open" → Spotlight shortcut.
    """

    gate_gestures: bool = True
    # Max |yaw| considered "facing the camera" (in [-1, 1] normalized units).
    gate_yaw_tolerance: float = 0.6
    gate_pitch_tolerance: float = 0.5
    # How many frames without a qualifying face before gesture control
    # is suppressed. Prevents blinks / momentary detection losses from
    # jittering the gate state.
    gate_grace_frames: int = 20
    # Mouth-open shortcut.
    mouth_open_threshold: float = 0.55
    mouth_open_hold_seconds: float = 3.0
    # Empty = built-in Spotlight (Cmd+Space); set to any shell command
    # to replace. Use e.g. "open -a Raycast" for a different launcher.
    mouth_open_command: str = ""


def _default_radial_tree() -> tuple[RadialItem, ...]:
    return (
        RadialItem("None"),
        RadialItem("Settings"),
        RadialItem("Mission"),
        RadialItem("Create", ("Finder",)),  # subs[0] = which app to spawn
        RadialItem("Scroll"),
        RadialItem("Screenshot"),
    )


@dataclass(frozen=True)
class Config:
    cursor: CursorConfig = field(default_factory=CursorConfig)
    radial: RadialConfig = field(default_factory=RadialConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    create: CreateConfig = field(default_factory=CreateConfig)
    failsafe: FailsafeConfig = field(default_factory=FailsafeConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    colors: ColorsConfig = field(default_factory=ColorsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gestures: GesturesConfig = field(default_factory=GesturesConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    radial_tree: tuple[RadialItem, ...] = field(default_factory=_default_radial_tree)


# ---- Serialization ------------------------------------------------------


def _to_dict(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        if f.name == "radial_tree":
            out["radial_tree"] = [
                {"name": it.name, "subs": list(it.subs), "command": it.command}
                for it in val
            ]
        elif is_dataclass(val):
            sect: dict[str, Any] = {}
            for sf in fields(val):
                sv = getattr(val, sf.name)
                if isinstance(sv, tuple):
                    sv = list(sv)
                sect[sf.name] = sv
            out[f.name] = sect
        else:
            out[f.name] = val
    return out


def _from_dict(data: dict[str, Any]) -> Config:
    """Build a Config, tolerating missing/extra keys (defaults fill gaps)."""
    cfg = Config()
    updates: dict[str, Any] = {}

    section_types = {f.name: f.type for f in fields(cfg)}  # noqa: F841 (debug aid)

    def _merge_section(name: str, cls: type) -> Any:
        existing = getattr(cfg, name)
        section_data = data.get(name, {})
        if not isinstance(section_data, dict):
            return existing
        kw: dict[str, Any] = {}
        for sf in fields(existing):
            if sf.name in section_data:
                val = section_data[sf.name]
                if isinstance(getattr(existing, sf.name), tuple) and isinstance(val, list):
                    val = tuple(val)
                kw[sf.name] = val
        return replace(existing, **kw) if kw else existing

    for section_name, cls in (
        ("cursor", CursorConfig),
        ("radial", RadialConfig),
        ("scroll", ScrollConfig),
        ("create", CreateConfig),
        ("failsafe", FailsafeConfig),
        ("overlay", OverlayConfig),
        ("colors", ColorsConfig),
        ("features", FeaturesConfig),
        ("server", ServerConfig),
        ("gestures", GesturesConfig),
        ("face", FaceConfig),
    ):
        updates[section_name] = _merge_section(section_name, cls)

    if "radial_tree" in data and isinstance(data["radial_tree"], list):
        items: list[RadialItem] = []
        for entry in data["radial_tree"]:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            subs = entry.get("subs", [])
            if not isinstance(subs, list):
                subs = []
            items.append(
                RadialItem(
                    name=str(entry["name"]),
                    subs=tuple(str(s) for s in subs),
                    command=entry.get("command") or None,
                )
            )
        if items:
            updates["radial_tree"] = tuple(items)

    return replace(cfg, **updates)


def _dump_toml(cfg: Config) -> str:
    """Minimal TOML serializer for our fixed schema (flat sections + one array of tables)."""
    data = _to_dict(cfg)
    lines: list[str] = [
        "# handspring config. Edits here are picked up live.",
        "# The settings web UI writes to this file too.",
        "",
    ]
    # Ordered top-level sections first.
    for section in (
        "cursor",
        "radial",
        "scroll",
        "create",
        "failsafe",
        "overlay",
        "colors",
        "features",
        "server",
        "gestures",
        "face",
    ):
        lines.append(f"[{section}]")
        for k, v in data[section].items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    # radial_tree: array of tables.
    for item in data["radial_tree"]:
        lines.append("[[radial_tree]]")
        lines.append(f"name = {_toml_value(item['name'])}")
        lines.append(f"subs = {_toml_value(item['subs'])}")
        if item.get("command"):
            lines.append(f"command = {_toml_value(item['command'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value: {v!r}")


# ---- Paths + load/save --------------------------------------------------


def default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "handspring" / "config.toml"


def load(path: Path | None = None) -> Config:
    """Load config from ``path`` (or the default location).

    If the file doesn't exist, writes defaults and returns them.
    Malformed files log a warning and fall back to defaults without writing.
    """
    p = path or default_config_path()
    if not p.exists():
        cfg = Config()
        with contextlib.suppress(OSError):
            save(cfg, p)
        return cfg
    try:
        raw = p.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        print(f"handspring: config load failed ({e}); using defaults", file=sys.stderr)
        return Config()
    return _from_dict(data)


def save(cfg: Config, path: Path | None = None) -> None:
    p = path or default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(_dump_toml(cfg), encoding="utf-8")
    os.replace(tmp, p)


# ---- Store + watcher ----------------------------------------------------


class ConfigStore:
    """Thread-safe holder for the current Config.

    Reads are a single attribute access (cheap, lock-free by virtue of
    Python's GIL on pointer assignment). Writes go through ``set()`` which
    also persists to disk.

    Pass ``persist=False`` for an in-memory-only store (useful in tests).
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        persist: bool = True,
        initial: Config | None = None,
    ) -> None:
        self._path = path or default_config_path()
        self._persist_default = persist
        if initial is not None:
            self._cfg = initial
        elif persist:
            self._cfg = load(self._path)
        else:
            self._cfg = Config()
        self._listeners: list[Callable[[Config], None]] = []
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def get(self) -> Config:
        return self._cfg

    def set(self, cfg: Config, *, persist: bool | None = None) -> None:
        with self._lock:
            self._cfg = cfg
            listeners = list(self._listeners)
        do_persist = self._persist_default if persist is None else persist
        if do_persist:
            try:
                save(cfg, self._path)
            except OSError as e:
                print(f"handspring: config save failed ({e})", file=sys.stderr)
        for cb in listeners:
            try:
                cb(cfg)
            except Exception as e:  # noqa: BLE001
                print(f"handspring: config listener error ({e})", file=sys.stderr)

    def reload(self) -> None:
        """Re-read from disk (ignores listener errors)."""
        self.set(load(self._path), persist=False)

    def on_change(self, callback: Callable[[Config], None]) -> None:
        with self._lock:
            self._listeners.append(callback)


class _MtimeWatcher:
    """Poll-based file watcher. No external deps.

    Checks the config file's mtime every ``interval`` seconds. On change,
    calls ``store.reload()``. Survives transient read errors (e.g., the
    file being mid-rename by an editor).
    """

    def __init__(self, store: ConfigStore, *, interval: float = 1.0) -> None:
        self._store = store
        self._interval = interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_mtime: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._last_mtime = self._store.path.stat().st_mtime
        except OSError:
            self._last_mtime = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="handspring-config-watch")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                mtime = self._store.path.stat().st_mtime
            except OSError:
                continue
            if self._last_mtime is None or mtime != self._last_mtime:
                self._last_mtime = mtime
                # Debounce: wait a moment in case the write is still in progress.
                time.sleep(0.05)
                try:
                    self._store.reload()
                except Exception as e:  # noqa: BLE001
                    print(f"handspring: config reload failed ({e})", file=sys.stderr)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def start_watcher(store: ConfigStore, *, interval: float = 1.0) -> _MtimeWatcher:
    w = _MtimeWatcher(store, interval=interval)
    w.start()
    return w
