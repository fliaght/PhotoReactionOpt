"""
pages/duel.py — LLM Duel: side-by-side model comparison.

Flow
====
User selects two models + quantizations, types a prompt, hits Send.
`_run_duel` is wrapped in `@task("LLM Duel")` which submits to the
global `TASK_MANAGER` (one task at a time across all sessions). When
it's our turn:

  1. Load LEFT model, stream tokens into the left chat, unload.
  2. Load RIGHT model, stream tokens into the right chat, unload.

Early exits (Stop button, errors) raise `EarlyExit` so `@task`'s
post-finally re-render still fires and the sidebar re-enables.

Conventions this page follows
=============================
- Large data (chat histories, perf log) lives in SESSION_CACHE via
  `session_get` / `session_set`. State only holds small metadata.
- Status messages go through `state.status` (read by the floating pill).
- Each milestone (load, generate, unload) appends one line to the
  `KEY_PERF_LOG` cache entry so it shows on the Performance page.
- The Stop button calls `TASK_MANAGER.cancel(state.my_task_id)` which
  sets the `stop` event that this module polls between steps.
"""

from __future__ import annotations

import time
from threading import Event, Thread

import mesop as me
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


class _CancelCriteria(StoppingCriteria):
    """Checked by transformers between forward passes — lets Stop/Cancel
    abort generation even when no tokens are being emitted fast enough
    for the streamer loop to detect it."""

    def __init__(self, cancel_event):
        self._cancel = cancel_event

    def __call__(self, input_ids, scores, **kwargs):
        return self._cancel.is_set()

from engine import (
    MODELS,
    MODEL_NAMES,
    add_to_history,
    build_messages,
    clear_gpu,
    default_quant_for,
    load_model,
    parse_history,
    supported_quants_for,
    tokenizers,
)
from shell import Page
from state import (
    KEY_DUEL_HISTORY_LEFT,
    KEY_DUEL_HISTORY_RIGHT,
    KEY_PERF_LOG,
    State,
    session_get,
    session_set,
)
from styles import C
from tasks import EarlyExit, TASK_MANAGER, task


def _history_left(state: State) -> list[str]:
    return session_get(state, KEY_DUEL_HISTORY_LEFT, []) or []


def _history_right(state: State) -> list[str]:
    return session_get(state, KEY_DUEL_HISTORY_RIGHT, []) or []


def _set_perf_log(state: State, text: str):
    session_set(state, KEY_PERF_LOG, text)


# =====================================================================
# Event handlers — model & quant selectors
# =====================================================================
def _on_model_left(e: me.SelectSelectionChangeEvent):
    state = me.state(State)
    state.model_left = e.value
    # Reset quant to a compatible default if the current one isn't supported.
    if state.quant_left not in supported_quants_for(e.value):
        state.quant_left = default_quant_for(e.value)


def _on_quant_left(e: me.SelectSelectionChangeEvent):
    me.state(State).quant_left = e.value


def _on_model_right(e: me.SelectSelectionChangeEvent):
    state = me.state(State)
    state.model_right = e.value
    if state.quant_right not in supported_quants_for(e.value):
        state.quant_right = default_quant_for(e.value)


def _on_quant_right(e: me.SelectSelectionChangeEvent):
    me.state(State).quant_right = e.value


# =====================================================================
# Event handlers — input / send / stop / clear
# =====================================================================
def _on_input_blur(e: me.InputBlurEvent):
    """Sync user_input on blur only — avoids per-keystroke re-render thrash."""
    me.state(State).user_input = e.value


def _on_clear(e: me.ClickEvent):
    state = me.state(State)
    session_set(state, KEY_DUEL_HISTORY_LEFT, [])
    session_set(state, KEY_DUEL_HISTORY_RIGHT, [])
    session_set(state, KEY_PERF_LOG, "")
    state.user_input = ""
    state.status = "Ready"


def _on_stop(e: me.ClickEvent):
    """Cancel THIS session's task via the global TASK_MANAGER."""
    state = me.state(State)
    if state.my_task_id:
        TASK_MANAGER.cancel(state.my_task_id)
    state.status = "Stop requested..."


def _on_send(e: me.ClickEvent):
    state = me.state(State)
    if not state.user_input.strip():
        return
    yield from _run_duel()


def _on_ctrl_enter(e: me.TextareaShortcutEvent):
    """Ctrl+Enter in the input sends the message."""
    state = me.state(State)
    state.user_input = e.value  # sync from event (faster than waiting for blur)
    if not state.user_input.strip():
        return
    yield from _run_duel()


# =====================================================================
# The main pipeline
# =====================================================================
@task("LLM Duel")
def _run_duel(stop: Event):
    """Run LEFT model, then RIGHT model, streaming tokens into chat panels.

    `stop` is the cancel_event provided by TASK_MANAGER; check it between
    steps (and in the token loop) to abort cleanly.
    """
    state = me.state(State)
    user_msg = state.user_input.strip()
    state.user_input = ""

    # Freeze the model/quant/sampling choices at send time.
    model_id_a = MODELS[state.model_left]
    model_id_b = MODELS[state.model_right]
    tok_a = tokenizers[model_id_a]
    tok_b = tokenizers[model_id_b]
    label_a = f"{state.model_left} ({state.quant_left})"
    label_b = f"{state.model_right} ({state.quant_right})"

    # Pull chat histories from session cache (mutable lists; changes persist).
    hist_left = _history_left(state)
    hist_right = _history_right(state)
    session_set(state, KEY_DUEL_HISTORY_LEFT, hist_left)
    session_set(state, KEY_DUEL_HISTORY_RIGHT, hist_right)

    # Mirror the user message into both panels.
    add_to_history(hist_left, "user", user_msg)
    add_to_history(hist_right, "user", user_msg)

    # Build prompt from left-side history (identical on both sides at this point).
    past = parse_history(hist_left[:-1])
    messages = build_messages(user_msg, past, state.system_prompt)

    log_lines: list[str] = []
    _set_perf_log(state, "")

    try:
        # Phase 1 — LEFT
        state.status = f"[1/6] Loading LEFT: {label_a}..."
        yield
        if stop.is_set():
            state.status = "Stopped"
            raise EarlyExit()

        t0 = time.perf_counter()
        model_a = load_model(model_id_a, state.quant_left)
        load_a = time.perf_counter() - t0
        log_lines.append(f"[LEFT]  Load: {load_a:.1f}s — {label_a}")
        _set_perf_log(state, "\n".join(log_lines))

        state.status = f"[2/6] LEFT generating: {label_a}..."
        yield

        tokens_a, gen_a, speed_a = yield from _stream_tokens(
            model=model_a, tokenizer=tok_a, messages=messages,
            history_list=hist_left, state=state,
            stop=stop, status_prefix=f"[2/6] LEFT generating: {label_a}",
        )
        log_lines.append(f"[LEFT]  Generate: {tokens_a} tokens in {gen_a:.1f}s ({speed_a:.1f} tok/s)")
        _set_perf_log(state, "\n".join(log_lines))

        state.status = "[3/6] Unloading LEFT model, clearing GPU..."
        yield
        del model_a
        clear_gpu()
        log_lines.append("[LEFT]  Unloaded — GPU cleared")
        _set_perf_log(state, "\n".join(log_lines))

        if stop.is_set():
            state.status = "Stopped — models unloaded"
            raise EarlyExit()

        # Phase 2 — RIGHT
        state.status = f"[4/6] Loading RIGHT: {label_b}..."
        yield

        t0 = time.perf_counter()
        model_b = load_model(model_id_b, state.quant_right)
        load_b = time.perf_counter() - t0
        log_lines.append(f"[RIGHT] Load: {load_b:.1f}s — {label_b}")
        _set_perf_log(state, "\n".join(log_lines))

        state.status = f"[5/6] RIGHT generating: {label_b}..."
        yield
        if stop.is_set():
            del model_b
            clear_gpu()
            state.status = "Stopped — models unloaded"
            raise EarlyExit()

        tokens_b, gen_b, speed_b = yield from _stream_tokens(
            model=model_b, tokenizer=tok_b, messages=messages,
            history_list=hist_right, state=state,
            stop=stop, status_prefix=f"[5/6] RIGHT generating: {label_b}",
        )
        log_lines.append(f"[RIGHT] Generate: {tokens_b} tokens in {gen_b:.1f}s ({speed_b:.1f} tok/s)")

        state.status = "[6/6] Unloading RIGHT model, clearing GPU..."
        yield
        del model_b
        clear_gpu()
        log_lines.append("[RIGHT] Unloaded — GPU cleared")

        total = load_a + gen_a + load_b + gen_b
        log_lines.append(f"--- Total: {total:.1f}s ---")
        _set_perf_log(state, "\n".join(log_lines))

        state.status = (
            f"Done — LEFT: {tokens_a} tokens ({speed_a:.1f} tok/s), "
            f"RIGHT: {tokens_b} tokens ({speed_b:.1f} tok/s)"
        )

    finally:
        # Resource cleanup that must always run.
        clear_gpu()


def _stream_tokens(*, model, tokenizer, messages, history_list, state, stop, status_prefix):
    """Generator: run model.generate and `yield` after each token.

    Consumed via `yield from`; its `return` value (count, elapsed, speed)
    is delivered as the value of the `yield from` expression.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=state.max_tokens,
        temperature=state.temperature, top_p=state.top_p, top_k=state.top_k,
        do_sample=state.temperature > 0,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_CancelCriteria(stop)]),
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    add_to_history(history_list, "assistant", "")
    count = 0
    t0 = time.perf_counter()

    try:
        for token in streamer:
            if stop.is_set():
                break
            last = history_list[-1]
            role, content = last.split("|", 1)
            history_list[-1] = f"{role}|{content}{token}"
            count += 1
            state.status = f"{status_prefix} ({count} tokens)..."
            yield  # drive a Mesop re-render
    finally:
        thread.join()

    elapsed = time.perf_counter() - t0
    speed = count / elapsed if elapsed > 0 else 0.0
    return count, elapsed, speed


# =====================================================================
# UI components
# =====================================================================
def _chat_panel(history_strs, label, accent):
    with me.box(style=me.Style(
        flex_grow=1, flex_basis="0", min_width="0",
        display="flex", flex_direction="column",
        background=C.bg_card,
        border_radius=12,
        box_shadow="0 1px 3px rgba(0,0,0,0.04), 0 2px 8px rgba(0,0,0,0.04)",
        overflow_y="hidden",
    )):
        # Header
        with me.box(style=me.Style(
            padding=me.Padding.symmetric(horizontal=16, vertical=12),
            border=me.Border(bottom=me.BorderSide(width=1, color="#eee", style="solid")),
            display="flex", align_items="center", gap="8px",
        )):
            with me.box(style=me.Style(
                width="8px", height="8px", border_radius="50%", background=accent,
            )):
                pass
            me.text(label, style=me.Style(
                font_weight="600", font_size="13px", color="#333",
                white_space="nowrap", overflow_x="hidden", text_overflow="ellipsis",
            ))
        # Messages
        with me.box(style=me.Style(
            flex_grow=1, overflow_y="auto",
            padding=me.Padding.all(16),
            display="flex", flex_direction="column", gap="12px",
        )):
            if not history_strs:
                me.text("No messages yet.", style=me.Style(
                    color="#aaa", font_size="13px", font_style="italic",
                    text_align="center", margin=me.Margin(top=24),
                ))
            for s in history_strs:
                if "|" not in s:
                    continue
                role, content = s.split("|", 1)
                if role == "user":
                    with me.box(style=me.Style(
                        background="#f3f4f6", border_radius=10,
                        padding=me.Padding.symmetric(horizontal=12, vertical=10),
                        align_self="flex-end", max_width="80%",
                    )):
                        me.text(content, style=me.Style(font_size="14px", color="#333"))
                else:
                    with me.box(style=me.Style(
                        background="#ffffff",
                        border=me.Border.all(me.BorderSide(width=1, color="#f0f0f0", style="solid")),
                        border_radius=10,
                        padding=me.Padding.symmetric(horizontal=12, vertical=10),
                        align_self="flex-start", max_width="90%",
                    )):
                        me.text(content, style=me.Style(
                            font_size="14px", color="#333", white_space="pre-wrap",
                        ))


def _model_selector(label, model_val, quant_val, on_model, on_quant, accent):
    # Quant dropdown options are filtered to whatever the chosen model
    # supports. See engine.SUPPORTED_QUANTS for the source of truth.
    quant_choices = supported_quants_for(model_val)
    with me.box(style=me.Style(
        flex_grow=1, flex_basis="0", min_width="0",
        display="flex", flex_direction="column", gap="6px",
    )):
        with me.box(style=me.Style(display="flex", align_items="center", gap="6px")):
            with me.box(style=me.Style(
                width="6px", height="6px", border_radius="50%", background=accent,
            )):
                pass
            me.text(label, style=me.Style(
                font_size="11px", font_weight="600", color="#666", letter_spacing="0.5px",
            ))
        with me.box(style=me.Style(display="flex", gap="8px")):
            me.select(
                label="Model",
                options=[me.SelectOption(label=n, value=n) for n in MODEL_NAMES],
                value=model_val,
                on_selection_change=on_model,
                appearance="outline",
                style=me.Style(flex_grow=1, min_width="0"),
            )
            me.select(
                label="Quant",
                options=[me.SelectOption(label=q, value=q) for q in quant_choices],
                value=quant_val,
                on_selection_change=on_quant,
                appearance="outline",
                style=me.Style(width="120px", flex_shrink=0),
            )


# =====================================================================
# Page render
# =====================================================================
def _render():
    state = me.state(State)

    # Top bar: model selectors
    with me.box(style=me.Style(
        background=C.bg_card,
        border=me.Border(bottom=me.BorderSide(width=1, color=C.border, style="solid")),
        padding=me.Padding.symmetric(horizontal=24, vertical=16),
        display="flex", align_items="flex-end", gap="16px",
    )):
        _model_selector(
            "LEFT MODEL", state.model_left, state.quant_left,
            _on_model_left, _on_quant_left, C.accent_left,
        )
        _model_selector(
            "RIGHT MODEL", state.model_right, state.quant_right,
            _on_model_right, _on_quant_right, C.accent_right,
        )

    # Chat panels
    with me.box(style=me.Style(
        flex_grow=1, overflow_y="hidden",
        padding=me.Padding.all(20),
        display="flex", gap="20px", min_height="0",
    )):
        _chat_panel(_history_left(state), state.model_left, C.accent_left)
        _chat_panel(_history_right(state), state.model_right, C.accent_right)

    # Input area
    with me.box(style=me.Style(
        background=C.bg_card,
        border=me.Border(top=me.BorderSide(width=1, color=C.border, style="solid")),
        padding=me.Padding.symmetric(horizontal=24, vertical=16),
        display="flex", flex_direction="column", gap="10px",
    )):
        me.textarea(
            label="Type your message...  (Ctrl+Enter to send)",
            value=state.user_input,
            on_blur=_on_input_blur,
            min_rows=1, max_rows=6,
            autosize=True,
            appearance="outline",
            shortcuts={me.Shortcut(key="Enter", ctrl=True): _on_ctrl_enter},
            style=me.Style(width="100%"),
        )
        with me.box(style=me.Style(
            display="flex", gap="10px", justify_content="flex-end",
        )):
            me.button("Clear", on_click=_on_clear, type="stroked",
                      style=me.Style(height="40px", min_width="80px"))
            me.button("Stop", on_click=_on_stop, type="stroked", color="warn",
                      disabled=not state.is_busy,
                      style=me.Style(height="40px", min_width="80px"))
            me.button("Send", on_click=_on_send, type="flat", color="primary",
                      disabled=state.is_busy,
                      style=me.Style(height="40px", min_width="96px"))


PAGE = Page(key="duel", title="LLM Duel", icon="⚔", render=_render)
