"""
pages/about.py — Project description, model list, compatibility matrix.

Purely informational: reads no State. All data driven by engine.py
constants so this page reflects the current registry without edits.
"""

from __future__ import annotations

import mesop as me

from engine import MODELS, QUANT_OPTIONS, SUPPORTED_QUANTS, TIER_DESCRIPTIONS
from shell import Page
from styles import (
    C,
    card_style,
    page_container_style,
    page_header,
    section_header,
)


def _render():
    with me.box(style=page_container_style()):
        with me.box(style=me.Style(
            max_width="720px", width="100%",
            display="flex", flex_direction="column", gap="20px",
        )):
            page_header(
                "About",
                "A side-by-side arena for comparing Google Gemma models and their abliterated variants.",
            )

            # Models card
            with me.box(style=card_style(padding=24, gap="10px")):
                section_header("Models")
                for display, mid in MODELS.items():
                    with me.box(style=me.Style(
                        padding=me.Padding.symmetric(horizontal=12, vertical=10),
                        background="#fafbfc", border_radius=8,
                        display="flex", flex_direction="column", gap="2px",
                    )):
                        me.text(display, style=me.Style(
                            font_size="13px", font_weight="600", color="#333",
                        ))
                        me.text(mid, style=me.Style(
                            font_family="ui-monospace, SFMono-Regular, monospace",
                            font_size="11px", color="#888",
                        ))

            # Compatibility matrix card (model tier × quant)
            with me.box(style=card_style(padding=24, gap="12px")):
                section_header("Model × Quantization compatibility")
                me.text(
                    "Which combinations actually work on 2× RTX 3090. "
                    "The Duel page's Quant dropdown is filtered accordingly.",
                    style=me.Style(font_size="12px", color=C.text_muted),
                )
                _compat_matrix()

            # Quantization notes
            with me.box(style=card_style(padding=24, gap="8px")):
                section_header("Quantization modes")
                for q, desc in [
                    ("BF16",     "Full precision. One GPU for 2.3B, split across two for 12B. 26B doesn't fit."),
                    ("BF16+CPU", "BF16 with CPU offload. Slow but works for models that don't fit in VRAM."),
                    ("NF4",      "4-bit NormalFloat on a single GPU. Fastest for 12B on this hardware."),
                    ("INT8",     "8-bit integer on a single GPU. Slow dequantization on RTX 3090."),
                ]:
                    with me.box(style=me.Style(display="flex", gap="12px", align_items="flex-start")):
                        with me.box(style=me.Style(
                            min_width="88px",
                            padding=me.Padding.symmetric(horizontal=8, vertical=4),
                            background=C.primary_bg, color=C.primary_fg,
                            border_radius=6, font_size="11px", font_weight="600",
                            text_align="center",
                        )):
                            me.text(q, style=me.Style(
                                font_size="11px", font_weight="600", color=C.primary_fg,
                            ))
                        me.text(desc, style=me.Style(
                            font_size="13px", color=C.text_dim, line_height="1.5",
                        ))

            # Hardware card
            with me.box(style=card_style(padding=24, gap="8px")):
                section_header("Hardware")
                me.text("2x NVIDIA RTX 3090 (24 GB each), PCIe.",
                        style=me.Style(font_size="13px", color="#555"))
                me.text("Global FIFO queue, single task at a time, sequential load/unload.",
                        style=me.Style(font_size="13px", color="#555"))

            me.text(
                "github.com/fliaght/PhotoReactionOpt  ·  based on github.com/fliaght/latelier",
                style=me.Style(
                    font_size="12px", color="#6366f1",
                    font_family="ui-monospace, monospace",
                    margin=me.Margin(top=8),
                ),
            )


def _compat_matrix():
    """Render SUPPORTED_QUANTS as a tier × quant table."""
    tiers = list(SUPPORTED_QUANTS.keys())

    with me.box(style=me.Style(
        display="flex", flex_direction="column", gap="0",
        border=me.Border.all(me.BorderSide(width=1, color=C.border_soft, style="solid")),
        border_radius=8,
        overflow_y="hidden",
    )):
        # Header row
        with me.box(style=me.Style(
            display="grid",
            grid_template_columns=f"2fr repeat({len(QUANT_OPTIONS)}, 1fr)",
            background=C.bg_muted,
            padding=me.Padding.symmetric(horizontal=12, vertical=10),
            border=me.Border(
                bottom=me.BorderSide(width=1, color=C.border_soft, style="solid")
            ),
        )):
            me.text("Model tier", style=_hdr_style())
            for q in QUANT_OPTIONS:
                me.text(q, style=_hdr_style(center=True))

        # Data rows
        for i, tier in enumerate(tiers):
            supported = set(SUPPORTED_QUANTS[tier])
            row_bg = C.bg_card if i % 2 == 0 else C.bg_muted
            with me.box(style=me.Style(
                display="grid",
                grid_template_columns=f"2fr repeat({len(QUANT_OPTIONS)}, 1fr)",
                background=row_bg,
                padding=me.Padding.symmetric(horizontal=12, vertical=10),
                border=me.Border(
                    bottom=me.BorderSide(width=1, color=C.border_soft, style="solid")
                ),
                align_items="center",
            )):
                # Tier label + description
                with me.box(style=me.Style(display="flex", flex_direction="column", gap="2px")):
                    me.text(_tier_label(tier), style=me.Style(
                        font_size="13px", font_weight="600", color=C.text,
                    ))
                    me.text(TIER_DESCRIPTIONS[tier], style=me.Style(
                        font_size="11px", color=C.text_muted, line_height="1.4",
                    ))
                # Per-quant cells: ✓ or —
                for q in QUANT_OPTIONS:
                    if q in supported:
                        me.text("✓", style=me.Style(
                            font_size="16px", color="#059669", text_align="center", font_weight="700",
                        ))
                    else:
                        me.text("—", style=me.Style(
                            font_size="14px", color=C.text_faint, text_align="center",
                        ))


def _hdr_style(center: bool = False) -> me.Style:
    return me.Style(
        font_size="11px", font_weight="700",
        color=C.text_dim, letter_spacing="0.5px",
        text_align="center" if center else "left",
    )


def _tier_label(tier: str) -> str:
    return {
        "2.3b":    "2.3B dense",
        "12b":     "12B dense",
        "26b_moe": "25.2B MoE",
    }.get(tier, tier)


PAGE = Page(key="about", title="About", icon="ℹ", render=_render)
