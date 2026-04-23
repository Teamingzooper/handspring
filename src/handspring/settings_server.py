"""HTTP settings server for handspring.

Runs separately from the MJPEG preview server. Serves:

- ``GET  /``               → single-page settings UI
- ``GET  /api/config``     → current config as JSON
- ``POST /api/config``     → replace config with JSON body, persists to disk
- ``POST /api/reload``     → reload from disk (discards in-memory changes)

Bound to 127.0.0.1 only. No auth — local machine assumed trusted, same as
the MJPEG server.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
from dataclasses import fields, is_dataclass, replace
from typing import Any

from handspring.config import (
    ColorsConfig,
    Config,
    ConfigStore,
    CreateConfig,
    CursorConfig,
    FailsafeConfig,
    FeaturesConfig,
    OverlayConfig,
    RadialConfig,
    RadialItem,
    ScrollConfig,
    ServerConfig,
    _to_dict,
)

_SECTION_TYPES: dict[str, type] = {
    "cursor": CursorConfig,
    "radial": RadialConfig,
    "scroll": ScrollConfig,
    "create": CreateConfig,
    "failsafe": FailsafeConfig,
    "overlay": OverlayConfig,
    "colors": ColorsConfig,
    "features": FeaturesConfig,
    "server": ServerConfig,
}


def _config_from_json(data: dict[str, Any], base: Config) -> Config:
    """Merge an incoming JSON payload onto ``base``. Unknown keys are ignored."""
    updates: dict[str, Any] = {}
    for section_name, _cls in _SECTION_TYPES.items():
        if section_name not in data or not isinstance(data[section_name], dict):
            continue
        existing = getattr(base, section_name)
        kw: dict[str, Any] = {}
        for sf in fields(existing):
            if sf.name in data[section_name]:
                val = data[section_name][sf.name]
                cur = getattr(existing, sf.name)
                if isinstance(cur, tuple) and isinstance(val, list):
                    val = tuple(val)
                elif isinstance(cur, int) and not isinstance(cur, bool) and isinstance(val, float):
                    val = int(val)
                kw[sf.name] = val
        if kw:
            updates[section_name] = replace(existing, **kw)
    if "radial_tree" in data and isinstance(data["radial_tree"], list):
        items: list[RadialItem] = []
        for entry in data["radial_tree"]:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            subs = entry.get("subs") or []
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
    return replace(base, **updates) if updates else base


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return _to_dict(o)  # type: ignore[arg-type]
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


_INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>handspring · settings</title>
<style>
  :root {
    --bg: #0f1014;
    --panel: #181a21;
    --border: #2a2d38;
    --text: #e8e8ee;
    --muted: #8a8f9c;
    --accent: #88ff00;
    --danger: #ff5864;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.4 -apple-system, BlinkMacSystemFont, sans-serif; }
  header { padding: 20px 28px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; justify-content: space-between; }
  header h1 { margin: 0; font-size: 18px; letter-spacing: 0.5px; }
  header .status { color: var(--muted); font-size: 12px; }
  main { max-width: 900px; margin: 0 auto; padding: 24px; }
  .section { background: var(--panel); border: 1px solid var(--border);
             border-radius: 8px; padding: 18px 22px; margin-bottom: 18px; }
  .section h2 { margin: 0 0 14px 0; font-size: 14px; text-transform: uppercase;
                color: var(--muted); letter-spacing: 1px; }
  .row { display: grid; grid-template-columns: 180px 1fr 60px; gap: 12px;
         align-items: center; padding: 6px 0; }
  .row label { color: var(--muted); font-size: 13px; }
  .row input[type=range] { width: 100%; }
  .row .val { text-align: right; font-variant-numeric: tabular-nums;
              color: var(--accent); font-size: 12px; }
  .row input[type=text], .row input[type=number] { width: 100%;
       background: #0b0c10; color: var(--text); border: 1px solid var(--border);
       padding: 6px 8px; border-radius: 4px; }
  .row input[type=checkbox] { width: 18px; height: 18px; justify-self: start; }
  .tree-item { border: 1px solid var(--border); border-radius: 6px;
               padding: 10px 12px; margin-bottom: 8px; background: #0b0c10; }
  .tree-item .head { display: grid; grid-template-columns: 1fr auto;
                     gap: 8px; align-items: center; }
  .tree-item input.name { font-weight: 600; }
  .tree-item .subs { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .tree-item .subs input { flex: 1 1 120px; min-width: 100px; }
  .tree-item .cmd { margin-top: 8px; }
  button { background: var(--accent); color: #000; border: 0;
           padding: 9px 18px; border-radius: 4px; font-weight: 600;
           cursor: pointer; font-size: 13px; }
  button.secondary { background: transparent; color: var(--text);
                     border: 1px solid var(--border); }
  button.danger { background: var(--danger); color: #fff; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .toolbar { display: flex; gap: 8px; margin-bottom: 18px; }
  .toast { position: fixed; bottom: 20px; right: 20px; padding: 12px 18px;
           background: var(--accent); color: #000; border-radius: 6px;
           font-weight: 600; opacity: 0; transition: opacity 0.2s;
           pointer-events: none; }
  .toast.show { opacity: 1; }
  .toast.err { background: var(--danger); color: #fff; }
</style>
</head>
<body>
<header>
  <h1>handspring · settings</h1>
  <div class=\"status\" id=\"status\">loading…</div>
</header>
<main>
  <div class=\"toolbar\">
    <button id=\"save\">Save</button>
    <button class=\"secondary\" id=\"reload\">Reload from disk</button>
    <button class=\"secondary\" id=\"reset\">Reset defaults</button>
  </div>
  <div id=\"root\"></div>
</main>
<div class=\"toast\" id=\"toast\"></div>

<script>
const SLIDERS = {
  cursor: {
    smoothing: [0, 1, 0.01, 'EMA smoothing (higher = snappier)'],
    inset: [0, 0.3, 0.01, 'Camera → screen edge inset'],
  },
  radial: {
    hold_seconds: [0.1, 2.0, 0.05, 'Pinch-hold to open wheel'],
    inner_radius: [0, 0.1, 0.005, 'Dead zone'],
    sub_threshold: [0.05, 0.3, 0.005, 'Root → sub-ring distance'],
  },
  scroll: {
    deadzone: [0, 0.3, 0.01, 'Middle band with no scroll'],
    max_pixels: [5, 80, 1, 'Max scroll px/frame'],
  },
  create: {
    entry_distance: [0.02, 0.2, 0.005, 'Hand-together threshold'],
    min_diagonal: [0.05, 0.4, 0.01, 'Minimum pull-apart distance'],
  },
  failsafe: {
    hold_seconds: [1, 10, 0.5, 'Both-fist hold to toggle'],
  },
  overlay: {
    scale: [0.5, 3.0, 0.05, 'Overlay size multiplier'],
  },
  server: {
    web_port: [1024, 65535, 1, 'MJPEG preview port'],
    settings_port: [1024, 65535, 1, 'Settings UI port'],
  },
};
const CHECKBOXES = {
  overlay: ['enabled'],
  features: ['tiling', 'spaces', 'mission_control', 'screenshots'],
};
const COLORS = {
  colors: ['radial_highlight', 'radial_outline', 'cursor_dot'],
};

let currentConfig = null;

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const k in attrs) {
    if (k === 'class') el.className = attrs[k];
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), attrs[k]);
    else if (attrs[k] === true) el.setAttribute(k, '');
    else if (attrs[k] !== false && attrs[k] != null) el.setAttribute(k, attrs[k]);
  }
  for (const k of kids) {
    if (k == null) continue;
    if (typeof k === 'string') el.appendChild(document.createTextNode(k));
    else el.appendChild(k);
  }
  return el;
}

function rgbToHex([r,g,b]) {
  return '#' + [r,g,b].map(n => n.toString(16).padStart(2,'0')).join('');
}
function hexToRgb(hex) {
  const m = hex.replace('#','');
  return [parseInt(m.slice(0,2),16), parseInt(m.slice(2,4),16), parseInt(m.slice(4,6),16)];
}

function render() {
  const root = document.getElementById('root');
  root.innerHTML = '';

  for (const section in SLIDERS) {
    const div = h('div', {class:'section'}, h('h2', {}, section));
    for (const key in SLIDERS[section]) {
      const [min, max, step, label] = SLIDERS[section][key];
      const val = currentConfig[section][key];
      const range = h('input', {type:'range', min, max, step, value: val});
      const valSpan = h('span', {class:'val'}, String(val));
      range.addEventListener('input', e => {
        const v = step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value);
        currentConfig[section][key] = v;
        valSpan.textContent = String(v);
      });
      div.appendChild(h('div', {class:'row'},
        h('label', {title: label}, key), range, valSpan));
    }
    root.appendChild(div);
  }

  // Checkboxes
  const featDiv = h('div', {class:'section'}, h('h2', {}, 'features / overlay'));
  for (const section in CHECKBOXES) {
    for (const key of CHECKBOXES[section]) {
      const val = currentConfig[section][key];
      const cb = h('input', {type:'checkbox'});
      cb.checked = !!val;
      cb.addEventListener('change', e => { currentConfig[section][key] = e.target.checked; });
      featDiv.appendChild(h('div', {class:'row'},
        h('label', {}, `${section}.${key}`), cb, h('span')));
    }
  }
  root.appendChild(featDiv);

  // Colors
  const colorDiv = h('div', {class:'section'}, h('h2', {}, 'colors'));
  for (const key of COLORS.colors) {
    const val = currentConfig.colors[key];
    const picker = h('input', {type:'color', value: rgbToHex(val)});
    picker.addEventListener('input', e => {
      currentConfig.colors[key] = hexToRgb(e.target.value);
    });
    colorDiv.appendChild(h('div', {class:'row'},
      h('label', {}, key), picker, h('span')));
  }
  root.appendChild(colorDiv);

  // Radial tree
  const treeDiv = h('div', {class:'section'}, h('h2', {}, 'radial tree'));
  const itemsDiv = h('div', {id:'tree-items'});
  const rerender = () => {
    itemsDiv.innerHTML = '';
    currentConfig.radial_tree.forEach((item, idx) => {
      const nameInput = h('input', {type:'text', class:'name', value: item.name});
      nameInput.addEventListener('input', e => { item.name = e.target.value; });
      const subsInput = h('input', {type:'text',
        placeholder: 'subs, comma-separated',
        value: (item.subs || []).join(', ')});
      subsInput.addEventListener('input', e => {
        item.subs = e.target.value.split(',').map(s => s.trim()).filter(Boolean);
      });
      const cmdInput = h('input', {type:'text',
        placeholder: 'optional shell command (leaf only)',
        value: item.command || ''});
      cmdInput.addEventListener('input', e => { item.command = e.target.value || null; });
      const del = h('button', {class:'danger',
        onclick: () => { currentConfig.radial_tree.splice(idx,1); rerender(); }},
        '×');
      const up = h('button', {class:'secondary',
        onclick: () => {
          if (idx === 0) return;
          const arr = currentConfig.radial_tree;
          [arr[idx-1], arr[idx]] = [arr[idx], arr[idx-1]];
          rerender();
        }}, '↑');
      const down = h('button', {class:'secondary',
        onclick: () => {
          const arr = currentConfig.radial_tree;
          if (idx === arr.length - 1) return;
          [arr[idx+1], arr[idx]] = [arr[idx], arr[idx+1]];
          rerender();
        }}, '↓');
      const head = h('div', {class:'head'}, nameInput,
        h('div', {}, up, down, del));
      const wrap = h('div', {class:'tree-item'}, head,
        h('div', {class:'subs'}, subsInput),
        h('div', {class:'cmd'}, cmdInput));
      itemsDiv.appendChild(wrap);
    });
  };
  rerender();
  const addBtn = h('button', {class:'secondary',
    onclick: () => {
      currentConfig.radial_tree.push({name:'New', subs:[], command:null});
      rerender();
    }}, '+ add item');
  treeDiv.appendChild(itemsDiv);
  treeDiv.appendChild(addBtn);
  root.appendChild(treeDiv);
}

async function load() {
  const r = await fetch('/api/config');
  currentConfig = await r.json();
  document.getElementById('status').textContent = 'loaded';
  render();
}

async function save() {
  const r = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(currentConfig),
  });
  if (r.ok) toast('saved');
  else toast('save failed', true);
}

async function reload() {
  await fetch('/api/reload', {method: 'POST'});
  await load();
  toast('reloaded');
}

function toast(msg, err) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (err ? ' err' : '');
  setTimeout(() => { t.className = 'toast'; }, 1400);
}

document.getElementById('save').addEventListener('click', save);
document.getElementById('reload').addEventListener('click', reload);
document.getElementById('reset').addEventListener('click', async () => {
  if (!confirm('Reset all settings to defaults?')) return;
  await fetch('/api/config', {method: 'POST',
    headers: {'Content-Type':'application/json'}, body: '{\"__reset\": true}'});
  await load();
});
load();
</script>
</body>
</html>
"""


def _make_handler(
    store: ConfigStore,
) -> type[http.server.BaseHTTPRequestHandler]:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, default=_json_default).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = _INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/config":
                self._send_json(200, _to_dict(store.get()))
            else:
                self.send_error(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/reload":
                store.reload()
                self._send_json(200, {"ok": True})
                return
            if self.path == "/api/config":
                length = int(self.headers.get("Content-Length", "0") or 0)
                try:
                    raw = self.rfile.read(length)
                    data = json.loads(raw.decode("utf-8")) if raw else {}
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    self._send_json(400, {"error": f"bad json: {e}"})
                    return
                if isinstance(data, dict) and data.get("__reset"):
                    store.set(Config())
                else:
                    try:
                        new_cfg = _config_from_json(data, store.get())
                    except (TypeError, ValueError) as e:
                        self._send_json(400, {"error": str(e)})
                        return
                    store.set(new_cfg)
                self._send_json(200, {"ok": True})
                return
            self.send_error(404, "not found")

    return _Handler


class _ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class SettingsServer:
    def __init__(self, store: ConfigStore, *, port: int = 8766) -> None:
        self._store = store
        self._port = port
        self._server: _ReusableThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        handler = _make_handler(self._store)
        self._server = _ReusableThreadingHTTPServer(("127.0.0.1", self._port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="handspring-settings"
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
