"""
shell.py — The app chrome that wraps every feature page.

Responsibilities
================
- Define the `Page` dataclass that pages register themselves as.
- Render the collapsible sidebar (logo, nav items, busy-aware disabling).
- Render the floating status pill at top-right.
- Dispatch to the currently-selected page via `render_shell(pages)`.

Nothing here knows about specific pages. Adding a new feature page does
not require changes to this file — the sidebar and dispatch both read
the `pages` argument passed into `render_shell`.

Imports
=======
`shell.py` is allowed to import from: `mesop`, `state`, `styles`,
`tasks`. It must NOT import from anything under `pages/` (that would
create a cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mesop as me

from state import State
from styles import C, edge_button_style


# =====================================================================
# Page registry type
# =====================================================================
@dataclass(frozen=True)
class Page:
    """One feature page registered with the shell.

    Fields
    ------
    key    : unique id matching `state.current_page`. Keep it URL-safe.
    title  : label shown in the sidebar nav.
    icon   : single glyph / emoji shown before the label.
    render : zero-arg callable that draws the page content. Must use
             `me.state(State)` to read/write state.
    """
    key: str
    title: str
    icon: str
    render: Callable[[], None]


# =====================================================================
# Event handlers owned by the shell
# =====================================================================
def on_open_sidebar(e: me.ClickEvent):
    me.state(State).sidebar_open = True


def on_close_sidebar(e: me.ClickEvent):
    me.state(State).sidebar_open = False


def _on_nav_click(e: me.ClickEvent):
    """Single dispatcher for every nav item.

    Each nav box has `key="nav-<page>-<busy|ready>"`. We extract the
    page key from the second segment.

    IMPORTANT: do not use per-item closure factories — Mesop identifies
    handlers by `__qualname__`, and all closures returned by a factory
    share the same qualname. They collapse into one handler, so only
    the last one registered actually works. See mesop-debugging skill
    §3 for the full story.
    """
    parts = e.key.split("-")
    if len(parts) >= 3 and parts[0] == "nav":
        me.state(State).current_page = parts[1]


# =====================================================================
# Logo
# =====================================================================
def _logo_mark():
    """The GA/LP square used in both collapsed and expanded sidebar."""
    with me.box(style=me.Style(
        width="36px", height="36px", border_radius=8,
        background=C.gradient_brand,
        display="flex", align_items="center", justify_content="center",
        flex_shrink=0,
    )):
        me.text("PR", style=me.Style(
            color="white", font_weight="700", font_size="14px",
        ))


# =====================================================================
# Sidebar — collapsed and expanded variants
# =====================================================================
def _sidebar_collapsed():
    """Just the logo, anchored at top-left (x=20, y=20 to match the
    expanded sidebar header padding — prevents the logo from jumping
    when the user toggles the drawer)."""
    with me.box(key="sidebar-collapsed-logo", style=me.Style(
        position="fixed", top="20px", left="20px", z_index=10,
    )):
        _logo_mark()


def _nav_item(page: Page, *, is_busy: bool, current: str):
    """Sidebar nav row. Non-current items become un-clickable while busy.

    DOM-reuse note: the `key` encodes the busy/ready state so Mesop
    actually remounts the element when state flips. Without this, the
    stale `on_click=None` binding would survive and clicks would
    silently do nothing after a task completes.
    """
    active = current == page.key
    disabled = is_busy and not active

    if disabled:
        bg, fg, cursor = C.bg_muted, C.disabled_muted, "not-allowed"
    elif active:
        bg, fg, cursor = C.primary_bg, C.primary_fg, "pointer"
    else:
        bg, fg, cursor = C.bg_card, C.text_dim, "pointer"

    with me.box(
        key=f"nav-{page.key}-{'busy' if disabled else 'ready'}",
        on_click=None if disabled else _on_nav_click,
        style=me.Style(
            padding=me.Padding.symmetric(horizontal=14, vertical=10),
            border_radius=8,
            background=bg,
            color=fg,
            display="flex", align_items="center", gap="10px",
            cursor=cursor,
            font_size="13px",
            font_weight="600" if active else "500",
        ),
    ):
        me.text(page.icon, style=me.Style(font_size="14px", color=fg))
        me.text(page.title, style=me.Style(color=fg))


def _sidebar_expanded(pages: list[Page]):
    """Full drawer: logo + title + nav list."""
    state = me.state(State)
    with me.box(style=me.Style(
        position="fixed", top=0, left=0, bottom=0,
        width="300px", z_index=20,
        background=C.bg_card,
        border=me.Border(
            right=me.BorderSide(width=1, color=C.border, style="solid")
        ),
        box_shadow=C.shadow_drawer,
        display="flex", flex_direction="column",
    )):
        # Brand header
        with me.box(style=me.Style(
            padding=me.Padding.symmetric(horizontal=20, vertical=20),
            border=me.Border(
                bottom=me.BorderSide(width=1, color=C.border_soft, style="solid")
            ),
            display="flex", align_items="center", gap="12px",
            flex_shrink=0,
        )):
            _logo_mark()
            with me.box(style=me.Style(display="flex", flex_direction="column")):
                me.text("PhotoReactionOpt", style=me.Style(
                    font_size="16px", font_weight="700", color=C.text,
                ))
                me.text("Photoredox reaction optimization", style=me.Style(
                    color=C.text_muted, font_size="11px", line_height="1.4",
                ))

        # Navigation
        with me.box(style=me.Style(
            padding=me.Padding.all(12),
            display="flex", flex_direction="column", gap="4px",
            flex_grow=1,
        )):
            for page in pages:
                _nav_item(page, is_busy=state.is_busy, current=state.current_page)


def _sidebar_open_button():
    """Floating ▶ button at the left edge, visible ONLY when closed."""
    state = me.state(State)
    disabled = state.is_busy
    suffix = "busy" if disabled else "ready"
    with me.box(key=f"sidebar-open-wrap-{suffix}", style=me.Style(
        position="fixed", top="50%", left="0px", z_index=30,
    )):
        with me.box(
            key=f"sidebar-open-btn-{suffix}",
            on_click=None if disabled else on_open_sidebar,
            style=edge_button_style(disabled),
        ):
            me.text("▶", style=me.Style(
                color=C.disabled_text if disabled else C.primary,
                font_size="14px",
            ))


def _sidebar_close_button():
    """Floating ◀ button at the sidebar's right edge, visible ONLY when open."""
    state = me.state(State)
    disabled = state.is_busy
    suffix = "busy" if disabled else "ready"
    with me.box(key=f"sidebar-close-wrap-{suffix}", style=me.Style(
        position="fixed", top="50%", left="300px", z_index=30,
    )):
        with me.box(
            key=f"sidebar-close-btn-{suffix}",
            on_click=None if disabled else on_close_sidebar,
            style=edge_button_style(disabled),
        ):
            me.text("◀", style=me.Style(
                color=C.disabled_text if disabled else C.primary,
                font_size="14px",
            ))


def _sidebar_overlay():
    """Passive dimming backdrop. NO on_click — closing is via the ◀ button
    only (see mesop-debugging skill: click-to-close overlays race with
    rapid re-renders and produce spurious 'unknown handler' warnings)."""
    with me.box(
        key="sidebar-overlay",
        style=me.Style(
            position="fixed", top=0, left=0, right=0, bottom=0,
            background="rgba(0, 0, 0, 0.2)", z_index=15,
        ),
    ):
        pass


# =====================================================================
# Floating status pill
# =====================================================================
def _status_colors(status: str) -> tuple[str, str, str]:
    low = status.lower()
    if status == "Ready":
        return "#ecfdf5", "#059669", "#10b981"
    if "stop" in low or "error" in low or "cancel" in low:
        return "#fef2f2", "#dc2626", "#ef4444"
    if "done" in low:
        return "#eef2ff", "#4338ca", "#6366f1"
    if "waiting" in low or "queue" in low:
        return "#fffbeb", "#b45309", "#f59e0b"
    return "#eff6ff", "#1d4ed8", "#3b82f6"  # busy / loading / generating


def _floating_status(status: str):
    """Top-right pill. `position: fixed` so it doesn't push page layout."""
    bg, fg, dot = _status_colors(status)
    with me.box(style=me.Style(
        position="fixed", top="20px", right="24px", z_index=25,
    )):
        with me.box(style=me.Style(
            display="inline-flex", align_items="center", gap="8px",
            background=bg, color=fg, border_radius=999,
            padding=me.Padding.symmetric(horizontal=14, vertical=8),
            font_size="12px", font_weight="500",
            box_shadow=C.shadow_pill,
        )):
            with me.box(style=me.Style(
                width="8px", height="8px", border_radius="50%", background=dot,
            )):
                pass
            me.text(status, style=me.Style(font_size="12px"))


# =====================================================================
# Public entry point
# =====================================================================
def render_shell(pages: list[Page]) -> None:
    """Render the app shell + dispatch to the current page.

    Call this from the `@me.page`-decorated function in `app_mesop.py`.
    """
    state = me.state(State)

    with me.box(style=me.Style(
        height="100vh", width="100%",
        background=C.bg_page,
        position="relative", overflow_x="hidden",
    )):
        # Main content, offset from the left so the sidebar edge button
        # doesn't overlap page UI.
        with me.box(style=me.Style(
            display="flex", flex_direction="column",
            height="100vh", overflow_y="hidden", width="100%",
            padding=me.Padding(left=64),
        )):
            current = next((p for p in pages if p.key == state.current_page), None)
            if current is None and pages:
                current = pages[0]
            if current is not None:
                current.render()

        # Sidebar (expanded or collapsed)
        if state.sidebar_open:
            _sidebar_overlay()
            _sidebar_expanded(pages)
            _sidebar_close_button()
        else:
            _sidebar_collapsed()
            _sidebar_open_button()

        # Floating status pill (does not occupy layout space)
        _floating_status(state.status)
