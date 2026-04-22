"""
engine.py — HuggingFace model loading, tokenization, GPU lifecycle.

Pure-backend module: no Mesop / UI imports. Safe to use from tests or
other front-ends.

Public surface
==============
Registry:
  - `MODELS`            display name → HF repo id
  - `MODEL_NAMES`       ordered list of display names
  - `QUANT_OPTIONS`     all quantization modes the UI ever shows
  - `tokenizers`        pre-loaded tokenizer per HF repo id
  - `SUPPORTED_QUANTS`  per-tier matrix of compatible quants
  - `TIER_DESCRIPTIONS` human-friendly blurb for each tier
  - `supported_quants_for(model)` / `default_quant_for(model)`

Model lifecycle:
  - `load_model(model_id, quant)` — load onto GPU(s); stores a module-
    level reference so `clear_gpu()` can reach it even if the caller
    drops its own binding. Always pair with `clear_gpu()` in a finally.
  - `clear_gpu()` — release the model + reclaim VRAM. Removes accelerate
    dispatch hooks BEFORE dropping the reference; otherwise the
    secondary GPU's hooked submodules stay pinned and leak. See
    `mesop-debugging` skill §8 for the gory details.
  - `ConfigError` — raised by `load_model` when the (model, quant) combo
    is not in `SUPPORTED_QUANTS`. Fails fast instead of silently
    offloading to disk and hanging generation.

Chat helpers:
  - `build_messages(user_msg, history, system_prompt)` — build the list
    passed to `tokenizer.apply_chat_template(...)`.
  - `parse_history` / `add_to_history` — convert between `"role|content"`
    string encoding (safe to store in caches) and dict form.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# =====================================================================
# Registry
# =====================================================================
MODELS: dict[str, str] = {
    "gemma-4-E2B-it (2.3B, Original)": "google/gemma-4-E2B-it",
    "gemma-4-E2B-it-heretic-ara (2.3B, ARA)": "p-e-w/gemma-4-E2B-it-heretic-ara",
    "gemma-3-12b-it (12B, Original)": "google/gemma-3-12b-it",
    "gemma-3-12b-it-heretic (12B, Heretic)": "p-e-w/gemma-3-12b-it-heretic",
    "gemma-4-26B-A4B-it (25.2B MoE, Original)": "google/gemma-4-26B-A4B-it",
}

MODEL_NAMES: list[str] = list(MODELS.keys())
QUANT_OPTIONS: list[str] = ["BF16", "BF16+CPU", "NF4", "INT8"]


# =====================================================================
# Model × quantization compatibility matrix
# =====================================================================
# Single source of truth for which (model, quant) combos actually work
# on the target hardware (2× RTX 3090). The UI reads these to filter
# dropdowns; the About page reads them to render the compatibility
# table. `load_model` raises `ConfigError` if an invalid combo slips
# through (e.g. loaded state with a stale selection).
#
# If you add a new model, update `_model_tier()` and `SUPPORTED_QUANTS`.

def _model_tier(model_id: str) -> str:
    """Coarse size/arch class used for compatibility lookups."""
    name = model_id.lower()
    if "26b" in name:
        return "26b_moe"
    if "12b" in name:
        return "12b"
    return "2.3b"


SUPPORTED_QUANTS: dict[str, list[str]] = {
    "2.3b":    ["BF16", "BF16+CPU", "NF4", "INT8"],
    "12b":     ["BF16", "BF16+CPU", "NF4", "INT8"],
    "26b_moe": ["BF16+CPU"],
}


# Human-friendly description of each tier for the About page.
TIER_DESCRIPTIONS: dict[str, str] = {
    "2.3b":    "2.3B dense (Gemma 4 E2B). Fits on a single GPU in any precision.",
    "12b":     "12B dense (Gemma 3 12B). BF16 needs both GPUs; NF4 is fastest on one.",
    "26b_moe": "25.2B MoE (Gemma 4 26B-A4B, 3.8B active). Only fits with CPU offload.",
}


def supported_quants_for(model_id_or_display: str) -> list[str]:
    """Return the list of quant options compatible with this model.

    Accepts either a display name (key of MODELS) or a raw HF repo id.
    """
    model_id = MODELS.get(model_id_or_display, model_id_or_display)
    return list(SUPPORTED_QUANTS[_model_tier(model_id)])


def default_quant_for(model_id_or_display: str) -> str:
    return supported_quants_for(model_id_or_display)[0]


# =====================================================================
# Tokenizers (pre-loaded at import time)
# =====================================================================
print("[engine] Loading tokenizers...")
tokenizers: dict[str, Any] = {}
for _display, _model_id in MODELS.items():
    print(f"[engine]   {_model_id}")
    tokenizers[_model_id] = AutoTokenizer.from_pretrained(_model_id)
print("[engine] All tokenizers ready.")


# =====================================================================
# Single-model GPU lifecycle
# =====================================================================
current_model_ref: Any = None


def _needs_dual_gpu(model_id: str) -> bool:
    name = model_id.lower()
    return "12b" in name or "26b" in name


class ConfigError(RuntimeError):
    """Raised when a (model, quant) combo is not supported on this hardware."""


def load_model(model_id: str, quant: str):
    """Load a model onto GPU according to the quantization mode.

    Stores the returned object in a module global so `clear_gpu()` can
    find it regardless of where the caller drops its own reference.

    Fails fast if `quant` is not in `SUPPORTED_QUANTS[tier]` for this
    model — prevents configurations that would otherwise hang or OOM.
    """
    global current_model_ref

    # Pre-validation against the compatibility matrix.
    allowed = supported_quants_for(model_id)
    if quant not in allowed:
        raise ConfigError(
            f"{quant} is not supported for {model_id}. "
            f"Valid options: {', '.join(allowed)}."
        )

    if quant == "NF4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="cuda:0",
        )
    elif quant == "INT8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="cuda:0",
        )
    elif quant == "BF16+CPU":
        # Leave ~4 GB per GPU for activations + KV cache during generation.
        # Without this headroom, all weights fit on GPU but the first
        # forward pass OOMs.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory={0: "20GiB", 1: "20GiB", "cpu": "48GiB"},
        )
    else:  # BF16
        if _needs_dual_gpu(model_id):
            # Constrain per-GPU so accelerate won't silently spill to
            # CPU/disk when GPU fills up; an OOM at load time is far
            # better UX than a silent hang during generation.
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto",
                max_memory={0: "23GiB", 1: "23GiB"},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="cuda:0",
            )

    current_model_ref = model
    return model


def clear_gpu(verbose: bool = False):
    """Release the globally-tracked model and free VRAM.

    Must call `remove_hook_from_module` BEFORE dropping the reference —
    otherwise accelerate's AlignDevicesHook retains submodules on
    secondary GPUs.
    """
    global current_model_ref
    if current_model_ref is not None:
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(current_model_ref, recurse=True)
        except Exception:
            pass
        current_model_ref = None

    # Multiple GC passes to collect generator frames, circular refs, etc.
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    if verbose:
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = (total - free) / 1024**3
            print(f"[engine] GPU{i}: {used:.2f} GB used")


# =====================================================================
# Messages / history utilities
# =====================================================================
def build_messages(user_message: str, history: list[dict], system_prompt: str) -> list[dict]:
    """Build a chat-template message list for `apply_chat_template`."""
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def parse_history(strs: list[str]) -> list[dict]:
    """Convert our `"role|content"` string-encoded history to dicts."""
    out: list[dict] = []
    for s in strs:
        if "|" in s:
            role, content = s.split("|", 1)
            out.append({"role": role, "content": content})
    return out


def add_to_history(history_list: list[str], role: str, content: str) -> None:
    history_list.append(f"{role}|{content}")
