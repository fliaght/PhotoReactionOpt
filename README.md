# PhotoReactionOpt

A web app for photoredox reaction optimization.

> Built on top of the **[LAtelier](https://github.com/fliaght/latelier)** framework — a modular Mesop AI workbench with a three-layer architecture (styles / framework / pages), a global task queue, and a standard recipe for adding new feature pages. This repository inherits LAtelier's framework code (sidebar, task queue, status bar, GPU lifecycle, session cache) and specializes the `pages/` directory for photoredox chemistry workflows.

The framework pieces (`app_mesop.py`, `state.py`, `styles.py`, `tasks.py`, `shell.py`, `engine.py`) are carried over unchanged. The example pages from LAtelier (LLM Duel, PDF Parser) are kept in `pages/` as reference implementations; they will be replaced over time by photoredox-specific tools (reaction design, condition recommender, result browser, …).

![framework](https://img.shields.io/badge/framework-Mesop-6366f1) ![base](https://img.shields.io/badge/based_on-LAtelier-4338ca) ![license](https://img.shields.io/badge/license-GPL--3.0-10b981)

---

## Contents

1. [Installation](#installation)
2. [Running](#running)
3. [Repository layout](#repository-layout)
4. [Architecture overview](#architecture-overview)
5. [Adding a new page in 4 steps](#adding-a-new-page-in-4-steps)
6. [Framework conventions (things every page must follow)](#framework-conventions-things-every-page-must-follow)
7. [License](#license)

---

## Installation

### Hardware

The framework itself has no GPU requirement. The carried-over LLM Duel / PDF Parser pages need a CUDA GPU (tested on 2× RTX 3090); future photoredox pages may run CPU-only depending on the models they call.

### Main app (`venv/`)

```bash
git clone https://github.com/fliaght/PhotoReactionOpt.git
cd PhotoReactionOpt

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### PDF Parser backend (`venv_marker/`) — optional

```bash
python3 -m venv venv_marker
source venv_marker/bin/activate
pip install --upgrade pip
pip install -r requirements-marker.txt
```

---

## Running

```bash
./restart_mesop.sh
```

Open <http://localhost:7861>. Stop with `pkill -TERM -f "mesop app_mesop.py"`.

---

## Repository layout

```
PhotoReactionOpt/
├── app_mesop.py              # Entry: monkey-patches, @me.page, exit handlers
│
├── state.py                  # @me.stateclass State + SESSION_CACHE helpers
├── styles.py                 # Color tokens + style factories
├── tasks.py                  # TaskManager, @task, EarlyExit
├── shell.py                  # Page dataclass, sidebar, status, render_shell
├── engine.py                 # HuggingFace lifecycle (inherited from LAtelier)
│
├── pages/
│   ├── __init__.py           # PAGES tuple — add your photoredox pages here
│   ├── duel.py               # (LAtelier) LLM Duel — template / to replace
│   ├── pdf_parse.py          # (LAtelier) PDF Parser — template / to replace
│   ├── settings.py           # ⚙ Settings
│   ├── performance.py        # 📊 Performance (queue + resources + log)
│   └── about.py              # ℹ About
│
├── requirements.txt
├── requirements-marker.txt
├── restart_mesop.sh
├── LICENSE                   # GPL-3.0
└── README.md
```

---

## Architecture overview

Three-layer separation with a one-way dependency graph. Identical to LAtelier — see that repo for a full write-up.

```
pages/*  →  shell, state, styles, tasks, engine
shell    →  state, styles, tasks
tasks    →  state
styles, state  →  (Mesop only)
```

Every long-running user action flows through one shared `TASK_MANAGER`:

```
click  → pages/<feature>.py::_on_something
       → yield from _run_work()
       → @task wrapper submits QueueEntry, sets is_busy=True
       → polls TASK_MANAGER.try_start() every 0.3 s (yields)
       → when next: executes the body with a cancel_event
       → TASK_MANAGER.finish(); is_busy=False; final yield → sidebar re-enables
```

At most one task runs at a time across all sessions; others queue.

---

## Adding a new page in 4 steps

### 1. Create `pages/<name>.py`

```python
import mesop as me

from shell import Page
from state import State
from styles import card_style, page_container_style, page_header, page_inner_style

def _render():
    state = me.state(State)
    with me.box(style=page_container_style()):
        with me.box(style=page_inner_style()):
            page_header("My Page", "Short description.")
            with me.box(style=card_style()):
                me.text("Content here.")

PAGE = Page(key="my", title="My Page", icon="✨", render=_render)
```

### 2. Wrap long work with `@task`

```python
from tasks import EarlyExit, task

@task("Photoredox Optimization")
def _run_work(cancel_event):
    state = me.state(State)
    state.status = "Step 1..."
    yield
    if cancel_event.is_set():
        raise EarlyExit()
    # ... do work, yield periodically, check cancel_event ...
```

Event handlers invoke it via `yield from`:

```python
def _on_start(e: me.ClickEvent):
    yield from _run_work()
```

### 3. Cache large outputs (don't store on State)

```python
from state import session_get, session_set

KEY_MY_OUTPUT = "my.output"

def _render():
    state = me.state(State)
    output = session_get(state, KEY_MY_OUTPUT, "") or ""
    me.markdown(output)
```

### 4. Register in `pages/__init__.py`

```python
from . import my_page, settings, performance, about  # + whatever else

PAGES = (
    my_page.PAGE,
    settings.PAGE,
    performance.PAGE,
    about.PAGE,
)
```

---

## Framework conventions (things every page must follow)

| Concern | API |
|---|---|
| **Progress messages** | `state.status = "..."` — floating pill picks colors from keywords (ready, loading, done, error, queue, cancel). |
| **Performance log** | `session_set(state, KEY_PERF_LOG, prev + "\n" + new_line)`. |
| **Long task** | `@task("Label")` on a generator that takes `cancel_event` as first arg. Raise `EarlyExit` to abort cleanly. |
| **Cancel the current session's task** | `TASK_MANAGER.cancel(state.my_task_id)`. |
| **Large output** | Use `session_set` / `session_get` — **never store multi-KB strings on `State`**, it multiplies re-render cost. |

### Hard rules

1. Never capture `state` in a module-level closure — get it inside each event handler with `me.state(State)`.
2. Event handlers are plain named module-level functions — no `lambda`, no `functools.partial`, no closure factories. Mesop identifies handlers by `__qualname__` and collapses same-named ones.
3. Keep State tiny — only primitive fields. Anything ≥ 1 KB goes into `SESSION_CACHE`.
4. Interactive elements whose behavior depends on state need a state-encoded `key` — e.g., `key=f"send-btn-{'busy' if state.is_busy else 'ready'}"`.
5. In a `@task` body, use `EarlyExit` — never `return`; `return` skips the decorator's post-finally re-render.
6. Dependency direction is one-way: pages → (shell / state / styles / tasks / engine).
7. After any code change, restart via `./restart_mesop.sh` — Mesop's hot reload is not reliable.

---

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

Derived from [fliaght/latelier](https://github.com/fliaght/latelier), which is also GPL-3.0.
