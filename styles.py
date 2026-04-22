"""
styles.py — Visual tokens and style factory functions.

Goal: every color, radius, shadow used in the app comes from this file.
Pages should import `C` (color tokens) and the factory helpers rather
than inlining raw color strings or repeating style blocks.

If you find yourself writing `background="#ffffff"` or
`border_radius=12` inside a page, consider whether it belongs here
instead.

Nothing in this module depends on Mesop state; it only re-exports
Mesop's `Style` / `Padding` / etc wrapped in convenient factories.
"""

from __future__ import annotations

import mesop as me


# =====================================================================
# Color tokens
# =====================================================================
class C:
    """Flat color palette. Names follow role, not shade."""
    # Surfaces
    bg_page = "#f5f6fa"
    bg_card = "#ffffff"
    bg_muted = "#fafbfc"
    bg_highlight = "#f3f4f6"

    # Borders / dividers
    border = "#eaeaea"
    border_soft = "#f0f0f0"
    border_soft_alt = "#e5e7eb"

    # Text
    text = "#111"
    text_dim = "#555"
    text_muted = "#888"
    text_faint = "#aaa"

    # Brand accent
    primary = "#6366f1"
    primary_bg = "#f3f4ff"
    primary_fg = "#4338ca"
    gradient_brand = "linear-gradient(135deg, #6366f1, #8b5cf6)"

    # Channel accents (LLM Duel left/right)
    accent_left = "#6366f1"
    accent_right = "#10b981"

    # Disabled state
    disabled_bg = "#f3f4f6"
    disabled_text = "#c0c4cc"
    disabled_muted = "#b0b4bd"

    # Shadows
    shadow_card = "0 1px 3px rgba(0,0,0,0.04), 0 2px 8px rgba(0,0,0,0.04)"
    shadow_edge = "2px 0 6px rgba(0, 0, 0, 0.08)"
    shadow_pill = "0 2px 8px rgba(0,0,0,0.08)"
    shadow_drawer = "2px 0 12px rgba(0, 0, 0, 0.06)"


# =====================================================================
# Style factories — use these instead of inline `me.Style(...)` repeats
# =====================================================================
def card_style(padding: int = 24, gap: str = "16px") -> me.Style:
    """Standard white card with shadow and rounded corners."""
    return me.Style(
        background=C.bg_card,
        border_radius=12,
        padding=me.Padding.all(padding),
        box_shadow=C.shadow_card,
        display="flex",
        flex_direction="column",
        gap=gap,
    )


def page_container_style() -> me.Style:
    """Outer scroll area for a page — centers content horizontally."""
    return me.Style(
        flex_grow=1,
        overflow_y="auto",
        padding=me.Padding.all(40),
        display="flex",
        justify_content="center",
    )


def page_inner_style(max_width: str = "720px") -> me.Style:
    """Inner width-bounded column for a page's content."""
    return me.Style(
        max_width=max_width,
        width="100%",
        display="flex",
        flex_direction="column",
        gap="20px",
    )


def edge_button_style(disabled: bool) -> me.Style:
    """Sidebar toggle buttons (▶ / ◀) floating on the left edge."""
    if disabled:
        return me.Style(
            width="36px", height="80px",
            background=C.disabled_bg,
            border=me.Border.all(
                me.BorderSide(width=1, color=C.border_soft_alt, style="solid")
            ),
            border_radius="0 10px 10px 0",
            box_shadow="2px 0 6px rgba(0,0,0,0.04)",
            display="flex", align_items="center", justify_content="center",
            cursor="not-allowed",
            color=C.disabled_text,
        )
    return me.Style(
        width="36px", height="80px",
        background=C.bg_card,
        border=me.Border.all(
            me.BorderSide(width=1, color=C.border_soft_alt, style="solid")
        ),
        border_radius="0 10px 10px 0",
        box_shadow=C.shadow_edge,
        display="flex", align_items="center", justify_content="center",
        cursor="pointer",
        color=C.primary,
    )


# =====================================================================
# Small text primitives
# =====================================================================
def page_header(title: str, subtitle: str = "") -> None:
    """Standard page title block used at the top of every page."""
    me.text(title, style=me.Style(
        font_size="28px", font_weight="700", color=C.text,
        margin=me.Margin(bottom=4),
    ))
    if subtitle:
        me.text(subtitle, style=me.Style(
            font_size="13px", color=C.text_muted, margin=me.Margin(bottom=12),
        ))


def section_header(label: str) -> None:
    """Small bold heading inside a card."""
    me.text(label, style=me.Style(
        font_size="14px", font_weight="600", color=C.text_dim,
    ))
