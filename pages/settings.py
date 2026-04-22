"""
pages/settings.py — sampling parameters & system prompt.

These values live in `State` and are consumed by other pages (the Duel
pipeline reads them when sending a message). This page only edits them.
"""

from __future__ import annotations

import mesop as me

from shell import Page
from state import State
from styles import (
    card_style,
    page_container_style,
    page_header,
    page_inner_style,
    section_header,
)


# =====================================================================
# Event handlers
# =====================================================================
def _on_system_prompt_blur(e: me.InputBlurEvent):
    me.state(State).system_prompt = e.value


def _on_max_tokens(e: me.SliderValueChangeEvent):
    me.state(State).max_tokens = int(e.value)


def _on_temperature(e: me.SliderValueChangeEvent):
    me.state(State).temperature = round(e.value, 2)


def _on_top_p(e: me.SliderValueChangeEvent):
    me.state(State).top_p = round(e.value, 2)


def _on_top_k(e: me.SliderValueChangeEvent):
    me.state(State).top_k = int(e.value)


# =====================================================================
# Page render
# =====================================================================
def _render():
    state = me.state(State)

    with me.box(style=page_container_style()):
        with me.box(style=page_inner_style()):
            page_header(
                "Settings",
                "Configure prompt and sampling parameters used for generation.",
            )

            # Single card containing System Prompt + Sampling sections.
            with me.box(style=card_style(padding=28, gap="24px")):
                # Section 1 — System Prompt
                with me.box(style=me.Style(display="flex", flex_direction="column", gap="8px")):
                    section_header("System Prompt")
                    me.text(
                        "Behavior instructions sent to the model before the user message.",
                        style=me.Style(font_size="12px", color="#888", margin=me.Margin(bottom=4)),
                    )
                    me.textarea(
                        label="System prompt",
                        value=state.system_prompt,
                        on_blur=_on_system_prompt_blur,
                        min_rows=3, max_rows=8,
                        appearance="outline",
                        style=me.Style(width="100%"),
                    )

                # Divider
                with me.box(style=me.Style(
                    height="1px", background="#f0f0f0",
                    margin=me.Margin.symmetric(vertical=4),
                )):
                    pass

                # Section 2 — Sampling
                with me.box(style=me.Style(display="flex", flex_direction="column", gap="16px")):
                    section_header("Sampling")
                    me.text(
                        "Controls how tokens are chosen during generation.",
                        style=me.Style(font_size="12px", color="#888", margin=me.Margin(bottom=4)),
                    )
                    _slider_row(f"Max Tokens — {state.max_tokens}",
                                state.max_tokens, 32, 2048, 32, _on_max_tokens)
                    _slider_row(f"Temperature — {state.temperature}",
                                state.temperature, 0, 2, 0.05, _on_temperature)
                    _slider_row(f"Top-p — {state.top_p}",
                                state.top_p, 0, 1, 0.05, _on_top_p)
                    _slider_row(f"Top-k — {state.top_k}",
                                state.top_k, 1, 100, 1, _on_top_k)


def _slider_row(label, value, min_v, max_v, step, handler):
    with me.box():
        me.text(label, style=me.Style(font_size="13px", color="#555", margin=me.Margin(bottom=4)))
        me.slider(min=min_v, max=max_v, step=step, value=value,
                  on_value_change=handler, style=me.Style(width="100%"))


PAGE = Page(key="settings", title="Settings", icon="⚙", render=_render)
