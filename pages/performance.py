"""
pages/performance.py — System overview page.

Shows, for any user who visits:
  * Live GPU memory usage (per device)
  * Current task queue with running + pending entries, cancel buttons
  * Recent task history (done / cancelled / error)
  * Performance log (per-line entries appended by tasks)

The data sources are module-level globals (`TASK_MANAGER`, the session
cache's `perf_log` key), so the same information is visible to every
session. Each session still needs to trigger a render to see fresh
state; use the Refresh button.
"""

from __future__ import annotations

import time

import mesop as me

from shell import Page
from state import KEY_PERF_LOG, State, session_get
from styles import (
    C,
    card_style,
    page_container_style,
    page_header,
    section_header,
)
from tasks import (
    TASK_CANCELLED,
    TASK_DONE,
    TASK_ERROR,
    TASK_MANAGER,
    TASK_QUEUED,
    TASK_RUNNING,
)


# =====================================================================
# Event handlers
# =====================================================================
def _on_refresh(e: me.ClickEvent):
    # Touching state is enough to trigger a re-render.
    me.state(State).status = me.state(State).status


def _on_cancel_click(e: me.ClickEvent):
    # The clicked cancel button's key encodes the task id: "cancel-<id>".
    parts = e.key.split("-", 1)
    if len(parts) == 2 and parts[0] == "cancel":
        TASK_MANAGER.cancel(parts[1])


# =====================================================================
# Rendering
# =====================================================================
def _render():
    state = me.state(State)
    entries = TASK_MANAGER.snapshot()
    running = [e for e in entries if e.status == TASK_RUNNING]
    queued = [e for e in entries if e.status == TASK_QUEUED]
    history = [e for e in entries if e.status in (TASK_DONE, TASK_CANCELLED, TASK_ERROR)]

    with me.box(style=page_container_style()):
        with me.box(style=me.Style(
            max_width="960px", width="100%",
            display="flex", flex_direction="column", gap="20px",
        )):
            _page_head()
            _system_card()
            _queue_card(running, queued)
            _history_card(history)
            _log_card(state)


def _page_head():
    with me.box(style=me.Style(
        display="flex", align_items="flex-start", justify_content="space-between",
        gap="16px",
    )):
        with me.box(style=me.Style(flex_grow=1)):
            page_header(
                "Performance",
                "System resources, task queue, history, and run logs.",
            )
        me.button(
            "Refresh", on_click=_on_refresh, type="stroked",
            style=me.Style(height="36px", min_width="100px", margin=me.Margin(top=20)),
        )


# ----- system card -------------------------------------------------------
def _system_card():
    gpu_lines = _gpu_status_lines()

    with me.box(style=card_style(padding=24, gap="12px")):
        section_header("System resources")
        with me.box(style=me.Style(display="flex", flex_direction="column", gap="6px")):
            for line in gpu_lines:
                me.text(line, style=me.Style(
                    font_family="ui-monospace, SFMono-Regular, monospace",
                    font_size="12px", color=C.text_dim, line_height="1.7",
                ))


def _gpu_status_lines() -> list[str]:
    try:
        import torch  # defer import so the page still renders without torch
        if not torch.cuda.is_available():
            return ["CUDA not available"]
        lines = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = (total - free) / 1024**3
            total_gb = total / 1024**3
            name = torch.cuda.get_device_name(i)
            pct = (used / total_gb) * 100 if total_gb else 0
            lines.append(f"GPU{i} ({name}): {used:5.2f} / {total_gb:5.2f} GB  ({pct:4.1f}%)")
        return lines
    except Exception as ex:
        return [f"GPU info unavailable: {ex}"]


# ----- queue card --------------------------------------------------------
def _queue_card(running, queued):
    with me.box(style=card_style(padding=24, gap="12px")):
        header_row = f"Queue  ·  {len(running)} running, {len(queued)} pending"
        section_header(header_row)

        if not running and not queued:
            me.text(
                "No active tasks.",
                style=me.Style(font_size="12px", color=C.text_faint, font_style="italic"),
            )
            return

        for entry in running + queued:
            _queue_row(entry, active=(entry in running))


def _queue_row(entry, *, active: bool):
    now = time.time()
    if active:
        age = now - (entry.started_at or entry.submitted_at)
        status_label = f"Running · {_fmt_dur(age)} elapsed"
        dot = "#3b82f6"
    else:
        age = now - entry.submitted_at
        status_label = f"Queued · waiting {_fmt_dur(age)}"
        dot = "#f59e0b"

    with me.box(style=me.Style(
        display="flex", align_items="center", gap="12px",
        padding=me.Padding.symmetric(horizontal=12, vertical=10),
        background="#fafbfc", border_radius=8,
    )):
        with me.box(style=me.Style(
            width="10px", height="10px", border_radius="50%", background=dot,
            flex_shrink=0,
        )):
            pass
        with me.box(style=me.Style(flex_grow=1, display="flex", flex_direction="column", gap="2px")):
            me.text(
                f"{entry.label}  ·  session {entry.session_id}",
                style=me.Style(font_size="13px", font_weight="600", color=C.text),
            )
            me.text(
                status_label,
                style=me.Style(font_size="11px", color=C.text_muted),
            )
        me.button(
            "Cancel", on_click=_on_cancel_click, type="stroked", color="warn",
            key=f"cancel-{entry.id}",
            style=me.Style(height="32px", min_width="78px"),
        )


# ----- history card ------------------------------------------------------
def _history_card(history):
    with me.box(style=card_style(padding=24, gap="10px")):
        section_header(f"Recent history ({len(history)})")
        if not history:
            me.text(
                "No completed tasks yet.",
                style=me.Style(font_size="12px", color=C.text_faint, font_style="italic"),
            )
            return
        # Newest first.
        for entry in reversed(history[-15:]):
            _history_row(entry)


def _history_row(entry):
    if entry.status == TASK_DONE:
        icon, color = "✓", "#059669"
    elif entry.status == TASK_CANCELLED:
        icon, color = "✕", "#b45309"
    else:
        icon, color = "!", "#dc2626"

    dur = (entry.finished_at or time.time()) - (entry.started_at or entry.submitted_at)
    ago = time.time() - (entry.finished_at or entry.submitted_at)

    with me.box(style=me.Style(
        display="flex", align_items="center", gap="10px",
        padding=me.Padding.symmetric(horizontal=12, vertical=8),
    )):
        me.text(icon, style=me.Style(color=color, font_size="14px", font_weight="700"))
        with me.box(style=me.Style(flex_grow=1, display="flex", flex_direction="column", gap="1px")):
            me.text(
                f"{entry.label}  ·  session {entry.session_id}",
                style=me.Style(font_size="12px", font_weight="500", color=C.text_dim),
            )
            tail = f"{entry.status} · {_fmt_dur(dur)} · {_fmt_dur(ago)} ago"
            if entry.error:
                tail += f"  —  {entry.error[:80]}"
            me.text(tail, style=me.Style(font_size="11px", color=C.text_muted))


# ----- log card ----------------------------------------------------------
def _log_card(state: State):
    log = session_get(state, KEY_PERF_LOG, "") or ""
    with me.box(style=card_style(padding=24, gap="8px")):
        section_header("Performance log")
        if not log:
            me.text(
                "No runs yet.",
                style=me.Style(font_size="12px", color=C.text_faint, font_style="italic"),
            )
        else:
            for line in log.split("\n"):
                me.text(line, style=me.Style(
                    font_family="ui-monospace, SFMono-Regular, monospace",
                    font_size="12px", color=C.text_dim, line_height="1.7",
                ))


# =====================================================================
# Helpers
# =====================================================================
def _fmt_dur(s: float) -> str:
    if s < 1:
        return f"{s*1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m, rem = divmod(s, 60)
    return f"{int(m)}m{rem:.0f}s"


PAGE = Page(key="log", title="Performance", icon="📊", render=_render)
