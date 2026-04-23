"""Tests for handspring.config + settings_server."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from handspring.config import (
    Config,
    ConfigStore,
    RadialItem,
    _dump_toml,
    _from_dict,
    default_config_path,
    load,
    save,
    start_watcher,
)
from handspring.settings_server import SettingsServer, _config_from_json


def test_default_config_has_more_slot() -> None:
    cfg = Config()
    names = [it.name for it in cfg.radial_tree]
    assert "More" in names
    more = next(it for it in cfg.radial_tree if it.name == "More")
    assert more.subs == ("Settings", "Reload", "Quit")


def test_toml_round_trip(tmp_path: Path) -> None:
    cfg = Config()
    p = tmp_path / "cfg.toml"
    save(cfg, p)
    assert p.exists()
    loaded = load(p)
    assert loaded == cfg


def test_load_missing_file_writes_defaults(tmp_path: Path) -> None:
    p = tmp_path / "nope.toml"
    assert not p.exists()
    cfg = load(p)
    assert cfg == Config()
    assert p.exists()


def test_load_malformed_falls_back(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("this is = not [ valid toml }\n", encoding="utf-8")
    cfg = load(p)
    assert cfg == Config()


def test_partial_config_merges_defaults() -> None:
    # Only override cursor.smoothing; everything else should default.
    partial = {"cursor": {"smoothing": 0.9}}
    cfg = _from_dict(partial)
    assert cfg.cursor.smoothing == 0.9
    assert cfg.cursor.inset == Config().cursor.inset
    assert len(cfg.radial_tree) == len(Config().radial_tree)


def test_custom_radial_tree_loads() -> None:
    data = {
        "radial_tree": [
            {"name": "MyApp", "subs": [], "command": "open -a Slack"},
            {"name": "None", "subs": []},
        ]
    }
    cfg = _from_dict(data)
    assert len(cfg.radial_tree) == 2
    assert cfg.radial_tree[0].command == "open -a Slack"


def test_config_store_in_memory_does_not_touch_disk(tmp_path: Path) -> None:
    p = tmp_path / "should-not-exist.toml"
    store = ConfigStore(path=p, persist=False)
    new_cfg = Config(radial_tree=(RadialItem("Solo"),))
    store.set(new_cfg)
    assert store.get().radial_tree[0].name == "Solo"
    assert not p.exists()


def test_config_store_persists_on_set(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    store = ConfigStore(path=p)
    new_cfg = Config(radial_tree=(RadialItem("X"),))
    store.set(new_cfg)
    assert p.exists()
    reread = load(p)
    assert reread.radial_tree[0].name == "X"


def test_listener_fires_on_set(tmp_path: Path) -> None:
    store = ConfigStore(path=tmp_path / "c.toml", persist=False)
    seen: list[Config] = []
    store.on_change(seen.append)
    store.set(Config(radial_tree=(RadialItem("A"),)))
    assert len(seen) == 1
    assert seen[0].radial_tree[0].name == "A"


def test_mtime_watcher_reloads_on_file_change(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    save(Config(), p)
    store = ConfigStore(path=p, persist=False)
    w = start_watcher(store, interval=0.05)
    try:
        # Mutate the file externally.
        time.sleep(0.1)
        altered = Config(radial_tree=(RadialItem("Zed"),))
        save(altered, p)
        # Wait up to a second for the watcher to pick it up.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if store.get().radial_tree[0].name == "Zed":
                break
            time.sleep(0.05)
        assert store.get().radial_tree[0].name == "Zed"
    finally:
        w.stop()


def test_config_from_json_merges_partial() -> None:
    base = Config()
    merged = _config_from_json({"cursor": {"smoothing": 0.7}}, base)
    assert merged.cursor.smoothing == 0.7
    assert merged.cursor.inset == base.cursor.inset


def test_config_from_json_ignores_unknown_keys() -> None:
    base = Config()
    merged = _config_from_json({"nonsense": 123, "cursor": {"zzz": 5}}, base)
    assert merged == base


def test_config_from_json_list_to_tuple() -> None:
    base = Config()
    merged = _config_from_json(
        {"colors": {"radial_highlight": [1, 2, 3]}}, base
    )
    assert merged.colors.radial_highlight == (1, 2, 3)


@pytest.fixture
def running_server(tmp_path: Path):
    store = ConfigStore(path=tmp_path / "c.toml", persist=False)
    # Pick a high port; reuse_address lets tests re-run quickly.
    SettingsServer(store, port=0)
    # port=0 means OS-assigned; start manually to grab the bound port.
    import http.server
    import socketserver

    from handspring.settings_server import _make_handler

    class _S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    s = _S(("127.0.0.1", 0), _make_handler(store))
    port = s.server_address[1]
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    try:
        yield store, port
    finally:
        s.shutdown()
        s.server_close()


def test_server_get_config(running_server) -> None:
    _store, port = running_server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/config") as r:
        data = json.loads(r.read())
    assert "cursor" in data
    assert "radial_tree" in data


def test_server_post_updates_store(running_server) -> None:
    store, port = running_server
    payload = json.dumps({"cursor": {"smoothing": 0.42}}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/config",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        assert r.status == 200
    assert store.get().cursor.smoothing == 0.42


def test_server_post_reset(running_server) -> None:
    store, port = running_server
    store.set(Config(radial_tree=(RadialItem("X"),)))
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/config",
        data=b'{"__reset": true}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req):
        pass
    assert store.get() == Config()


def test_server_serves_html(running_server) -> None:
    _store, port = running_server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
        body = r.read().decode()
    assert "handspring" in body
    assert "<html" in body.lower()


def test_default_config_path_xdg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_config_path() == tmp_path / "handspring" / "config.toml"


def test_default_config_path_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    path = default_config_path()
    assert path.name == "config.toml"
    assert path.parent.name == "handspring"


def test_dump_toml_is_stable() -> None:
    s1 = _dump_toml(Config())
    s2 = _dump_toml(Config())
    assert s1 == s2


def test_radial_config_defaults_match_flick_model() -> None:
    cfg = Config()
    assert cfg.radial.flick_threshold == 0.03
    assert cfg.radial.angular_hysteresis == 0.15
    # Old fields must be gone:
    assert not hasattr(cfg.radial, "hold_seconds")
    assert not hasattr(cfg.radial, "sub_threshold")
    assert not hasattr(cfg.radial, "inner_radius")
    assert not hasattr(cfg.radial, "sub_mini_inner")


def test_create_config_has_smoothing() -> None:
    cfg = Config()
    assert cfg.create.smoothing == 0.35
