# CLAUDE.md — Context for Claude Code agents

## Project goal

**PhotoReactionOpt**: a web app for **photoredox reaction optimization**.
Built on top of the LAtelier framework
(https://github.com/fliaght/latelier) — a three-layer Mesop app with a
global task queue. The framework code (`app_mesop.py`, `state.py`,
`styles.py`, `tasks.py`, `shell.py`, `engine.py`) is carried over
unchanged; the `pages/` directory is being specialized for photoredox
workflows.

Engineering goal is preserved from LAtelier: **low marginal cost for
adding a new feature page**. A new page is ~100-300 lines touching
only `pages/<name>.py` and the `PAGES` tuple in `pages/__init__.py`.

## Status of inherited pages

The example pages from LAtelier are kept as working templates. Treat
them as reference, expect to replace them as photoredox features come
online:

| Page | Origin | Planned disposition |
|---|---|---|
| `pages/duel.py` | LAtelier (LLM Duel) | keep / adapt into a reaction model comparator, or remove |
| `pages/pdf_parse.py` | LAtelier (Marker PDF → Markdown) | may be useful for literature-mining; keep for now |
| `pages/settings.py` | LAtelier | generalize beyond LLM sampling params |
| `pages/performance.py` | LAtelier | keep — it renders from TASK_MANAGER which is feature-neutral |
| `pages/about.py` | LAtelier | update model / quant matrix as photoredox modules arrive |

`engine.py` currently registers Gemma models (carry-over). If
photoredox features don't use HuggingFace LLMs, the module can be
shrunk or replaced; the `clear_gpu()` / `load_model()` API is the
only part pages depend on.

## Hardware (inherited test bench)

- 2× NVIDIA RTX 3090 (24 GB each), PCIe, no NVLink.
- Photoredox modules may not need GPU at all. Keep the framework
  tolerant: `engine.clear_gpu` / `load_model` are optional for any
  given page.

## Layering (must keep)

One-way import direction. Never invert.

```
pages/*      depend on → shell, state, styles, tasks, engine
shell        depends on → state, styles, tasks
tasks        depends on → state
styles       depends on → (Mesop only)
state        depends on → (Mesop only)
engine       depends on → (Torch / Transformers / accelerate)
```

- `state.py` — `@me.stateclass State` + `SESSION_CACHE` helpers.
  State is small; large blobs (reaction datasets, prediction outputs)
  live in the cache.
- `styles.py` — colors + style factories + text helpers. No state
  references, no event handlers.
- `tasks.py` — `TaskManager`, `@task`, `EarlyExit`. Global FIFO queue.
- `shell.py` — `Page` dataclass, sidebar, floating status, dispatch.
  Generic: renders whatever list of `Page` you pass.
- `engine.py` — HuggingFace model lifecycle (currently LLM-focused;
  adapt as photoredox models come in).
- `pages/*` — one feature per file; each exports `PAGE: Page`.

## Framework contract

Pages interact with the shell via these APIs. Never hand-roll them.

- **Progress text**: `state.status = "..."`. The floating pill picks
  colors from keywords (ready / busy / loading / waiting / done /
  stop / error / cancel).
- **Performance log**: append to `SESSION_CACHE[KEY_PERF_LOG]`. The
  Performance page reads it.
- **Long task**: decorate a generator with `@task("Label")` (from
  `tasks.py`). The function takes `cancel_event: threading.Event` as
  its first parameter. Inside:
  - `yield` often so Mesop stays responsive.
  - `if cancel_event.is_set(): raise EarlyExit()` — this bypasses
    remaining work but still runs `finally` and the decorator's
    post-finally re-render (vital for re-enabling the sidebar).
- **Cancel the session's task**: `TASK_MANAGER.cancel(state.my_task_id)`.
- **Large data**: `session_set(state, key, value)` / `session_get(state, key, default)`.
  Never store multi-KB strings on `State` — they multiply re-render cost.

## Key design guarantees (from LAtelier)

1. **At most one task runs at a time** across the whole process.
   `TaskManager.try_start()` is the single choke point.
2. **Per-session mutex** (`state.is_busy`) prevents a single user from
   queuing parallel tasks.
3. **Final re-render after any task** is unconditional. `@task`'s
   post-finally yield always fires, so UI controls re-enable even on
   exceptions.
4. **GPU cleanup** in `engine.clear_gpu()` calls
   `accelerate.hooks.remove_hook_from_module` before dropping the
   reference. Without that, secondary-GPU memory leaks on
   `device_map="auto"` models.

## Mesop gotchas we've codified

All of these are already handled inside the framework. They're
catalogued here (and in the shared skill at
`~/.claude/skills/mesop-debugging/SKILL.md`) because each one bites
hard and looks different every time. **Re-encountering any of these
means something upstream was regressed — check before patching
symptoms.**

1. **DOM reuse** — state-dependent `on_click` / `disabled` must encode
   the state in the element's `key`. Shell helpers already use keys
   like `nav-<page>-<busy|ready>`, `sidebar-open-btn-<busy|ready>`.
2. **Finalizer yield** — flipping `state.is_busy = False` without a
   following `yield` leaves the sidebar stuck. `@task` handles this
   unconditionally (post-finally yield).
3. **Closure handler collapse** — `make_handler(page.key)` style
   factories collide because all closures share the same
   `__qualname__`. Use a single module-level dispatcher that reads
   `e.key` (see shell's `_on_nav_click`).
4. **bool serialized as int** — `isinstance(True, int)` is True in
   Python; Mesop miscategorizes bools into `int_value` in the proto.
   `app_mesop.py` monkey-patches `map_code_value` to check bool first.
5. **`ClickEvent` is not user-constructible** — it has seven required
   kwargs. To invoke a click handler from another handler, extract the
   body into a plain helper and `yield from` it.
6. **`on_input` swallows characters under load** — fires on every
   keystroke and triggers a full re-render. Use `on_blur`; shortcut
   handlers read `e.value` from the event directly.
7. **Hot reload is unreliable** — only body edits inside an existing
   `@me.page` function are reliably hot-reloaded. Structural changes
   (new functions, state fields, imports) need a full restart. Always
   use `./restart_mesop.sh`.
8. **VRAM retained on secondary GPU after `device_map="auto"` unload**
   — `accelerate.hooks.AlignDevicesHook` holds refs to dispatched
   submodules. `engine.clear_gpu()` calls `remove_hook_from_module`
   BEFORE dropping the reference. Without that, GPU 1 leaks ~half the
   model.
9. **State is retransmitted on every render** — keep `State` tiny;
   push anything ≥ 1 KB into `SESSION_CACHE` via `session_get` /
   `session_set`. Otherwise every sidebar click reserializes the blob.
10. **Full-screen `on_click` overlays race with rapid re-renders** —
    produce spurious "unknown handler id" warnings. The sidebar overlay
    is purely visual; close via the ◀ button.

## When editing this project

- Before adding state fields: check if a `SESSION_CACHE` key suffices.
- Before writing an event handler: make sure it's a plain module-level
  named function (no lambdas, no closures from factories).
- Before adding a new `me.box` with `on_click` that depends on state:
  include the state in its `key` (e.g. `f"my-btn-{'busy' if busy else 'ready'}"`).
- Before shelling out to a long-running binary: first try a simple
  per-request subprocess (like Marker in `pages/pdf_parse.py`). Only
  reach for a persistent sidecar if the startup cost is prohibitive.
- Before adding a heavy ML dep to the main venv: check if it would
  downgrade `transformers` / `torch`. If yes, put it in its own venv
  and shell out (the pattern used for `marker-pdf` in `venv_marker/`).
- Always restart via `./restart_mesop.sh`, never trust hot reload.
- When finishing a work session, push to the remote:
  `https://github.com/fliaght/PhotoReactionOpt` (public, GPL-3.0).

## Deployment / running (same as LAtelier)

```bash
# First-time setup
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional PDF Parser backend
python3 -m venv venv_marker
source venv_marker/bin/activate
pip install --upgrade pip
pip install -r requirements-marker.txt

# Run
./restart_mesop.sh          # kills previous, waits for VRAM, starts on :7861
# Open http://localhost:7861

# Stop
pkill -TERM -f "mesop app_mesop.py"
```

`restart_mesop.sh` polls `nvidia-smi` until VRAM is below 1 GB before
starting the new instance. On a CPU-only box it will simply not wait
for GPU cleanup (the check still passes trivially).

## Current pages (as inherited)

| Key | Title | File | Purpose |
|---|---|---|---|
| `duel` | LLM Duel | `pages/duel.py` | LLM Duel — inherited placeholder |
| `pdf` | PDF Parser | `pages/pdf_parse.py` | Marker PDF → Markdown — may keep for literature work |
| `settings` | Settings | `pages/settings.py` | Sampling + system prompt — to generalize |
| `log` | Performance | `pages/performance.py` | Queue + resources + log — keep as-is |
| `about` | About | `pages/about.py` | Project info |

## GitHub

- Repo: `https://github.com/fliaght/PhotoReactionOpt` (public, GPL-3.0)
- Base: `https://github.com/fliaght/latelier` (public, GPL-3.0)
- Owner: fliaght (Yi Luo)
