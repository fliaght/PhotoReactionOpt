"""
state.py — The single per-session `@me.stateclass` for the whole app,
plus a module-level session cache for large payloads.

Philosophy
==========
Mesop re-serializes the full `State` object to the browser on every
re-render. Keep it **tiny**: strings, numbers, bools, short lists. Any
blob that can grow large (chat history, parsed markdown, perf log)
should live in `SESSION_CACHE` instead. State only needs a cheap flag
(like `pdf_has_result`) to drive conditional rendering.

Add a new field here only when:
  * it's small (< ~1 KB),
  * every page may legitimately read it, OR
  * a single page needs it to trigger Mesop re-renders (e.g. `is_busy`).

Otherwise, prefer `session_get` / `session_set` with a keyed string.
"""

from __future__ import annotations

import uuid

import mesop as me


# =====================================================================
# The shared state
# =====================================================================
@me.stateclass
class State:
    # Session bookkeeping --------------------------------------------------
    session_id: str = ""          # lazy-assigned on first write; keys SESSION_CACHE
    my_task_id: str = ""          # id of this session's task in TASK_MANAGER

    # Shell UI -----------------------------------------------------------
    current_page: str = "duel"    # key of the page currently shown
    sidebar_open: bool = False
    status: str = "Ready"         # text shown in the floating status pill
    is_busy: bool = False         # True while THIS session has a task submitted/running

    # Settings (edited via the Settings page, consumed by page tasks) ------
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64

    # LLM Duel page: tiny fields only (history is cached separately) ------
    model_left: str = "gemma-3-12b-it-heretic (12B, Heretic)"
    quant_left: str = "BF16"
    model_right: str = "gemma-3-12b-it (12B, Original)"
    quant_right: str = "BF16"
    user_input: str = ""

    # PDF Parser page: metadata only (markdown is cached separately) ------
    pdf_filename: str = ""
    pdf_size: int = 0
    pdf_path: str = ""
    pdf_has_result: bool = False  # drives "show output card" conditional
    pdf_force_ocr: bool = False   # pass --force_ocr to marker


# =====================================================================
# Session cache (for data too large to live in State)
# =====================================================================
SESSION_CACHE: dict[str, dict[str, object]] = {}


def ensure_session_id(state: State) -> str:
    """Allocate a session_id if empty, return it."""
    if not state.session_id:
        state.session_id = str(uuid.uuid4())[:8]
    return state.session_id


def session_get(state: State, key: str, default=None):
    sid = state.session_id
    if not sid:
        return default
    return SESSION_CACHE.get(sid, {}).get(key, default)


def session_set(state: State, key: str, value) -> None:
    sid = ensure_session_id(state)
    SESSION_CACHE.setdefault(sid, {})[key] = value


def session_del(state: State, key: str) -> None:
    sid = state.session_id
    if not sid:
        return
    bucket = SESSION_CACHE.get(sid)
    if bucket and key in bucket:
        del bucket[key]


# =====================================================================
# Well-known cache keys (used by multiple modules)
# =====================================================================
KEY_PERF_LOG = "perf_log"                   # shared across all pages
KEY_DUEL_HISTORY_LEFT = "duel.history_left"
KEY_DUEL_HISTORY_RIGHT = "duel.history_right"
KEY_PDF_MARKDOWN = "pdf.markdown"
