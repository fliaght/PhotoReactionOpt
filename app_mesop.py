"""
app_mesop.py — PhotoReactionOpt entry point.

Responsibilities
================
- Apply the single Mesop monkey-patch that fixes bool-as-int proto
  encoding (must run before any `@me.page` is defined).
- Register `SIGTERM` / `SIGINT` / `atexit` handlers that call
  `clear_gpu()` before the process exits, so VRAM is released cleanly.
- Register the single `@me.page("/")` route and delegate to
  `render_shell(PAGES)`.

Module layout (see README.md for the full story)
================================================
  app_mesop.py  ← this file.
  state.py      State + SESSION_CACHE.
  styles.py     Color tokens + style factories.
  tasks.py      TaskManager + @task + EarlyExit.
  shell.py      Page dataclass + sidebar + status + render_shell.
  engine.py     HuggingFace model lifecycle.
  pages/        One module per feature page; each exports PAGE: Page.

Adding a new page: create `pages/<name>.py` exporting `PAGE = Page(...)`
and append it to the tuple in `pages/__init__.py`. This file is
unchanged.
"""

# =====================================================================
# 1) Mesop bool-as-int monkey-patch.
# =====================================================================
# Mesop's `map_code_value` checks `isinstance(value, int)` before
# `isinstance(value, bool)`, and `bool` is a subclass of `int` in
# Python, so booleans are wrongly routed into `int_value` and fail
# proto validation. We patch the check to look at `bool` first.
# Must run before any page module is imported.
import mesop as me
from mesop.component_helpers import helper as _mesop_helper
from mesop.protos import ui_pb2 as _mesop_pb


def _patched_map_code_value(value):
    if isinstance(value, bool):
        return _mesop_pb.CodeValue(bool_value=value)
    if isinstance(value, str):
        return _mesop_pb.CodeValue(string_value=value)
    if isinstance(value, int):
        return _mesop_pb.CodeValue(int_value=value)
    if isinstance(value, float):
        return _mesop_pb.CodeValue(double_value=value)
    return None


_mesop_helper.map_code_value = _patched_map_code_value


# =====================================================================
# 2) Process-exit cleanup.
# =====================================================================
import atexit
import signal
import sys

from engine import clear_gpu


def _cleanup_on_exit(*args):
    """Release GPU memory before the process dies.

    Safe to call multiple times (clear_gpu is idempotent). When invoked
    by a signal handler we also call sys.exit(0) so the caller's wait
    returns cleanly.
    """
    try:
        clear_gpu()
    except Exception:
        pass
    if args:  # signal-handler invocation (SIGTERM / SIGINT)
        sys.exit(0)


atexit.register(_cleanup_on_exit)
signal.signal(signal.SIGTERM, _cleanup_on_exit)
signal.signal(signal.SIGINT, _cleanup_on_exit)


# =====================================================================
# 3) Page registration.
# =====================================================================
from pages import PAGES
from shell import render_shell


@me.page(path="/", title="PhotoReactionOpt")
def _root():
    render_shell(list(PAGES))
