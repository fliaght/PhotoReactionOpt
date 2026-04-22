"""
pages/pdf_parse.py — Upload a PDF and parse it with Marker.

Backend design
==============
Marker (`datalab-to/marker`) is a lighter-weight alternative to MinerU
that exposes a reliable `marker_single` CLI. It still pulls in its own
PyTorch + Surya models, so we keep it in a dedicated `venv_marker/`
virtualenv to avoid conflicting with the main app's pinned transformers
version.

We shell out to `venv_marker/bin/marker_single <pdf> --output_dir <dir>
--output_format markdown`. It writes the markdown file into a per-PDF
subfolder of `--output_dir`; we glob for it and read it back.

Setup (one-time)
----------------
    python3 -m venv venv_marker
    source venv_marker/bin/activate
    pip install --upgrade pip
    pip install marker-pdf

The models auto-download on first run.

Page flow
---------
1. User uploads a PDF via `me.uploader`. Bytes are saved to a temp
   file; state only holds the path + metadata.
2. User clicks "Parse". The `@task`-wrapped generator runs
   `marker_single` as a subprocess and reads the produced `.md`.
3. Result is rendered via `me.markdown`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import mesop as me

from shell import Page
from state import (
    KEY_PDF_MARKDOWN,
    KEY_PERF_LOG,
    State,
    session_get,
    session_set,
)
from styles import C, card_style, page_container_style, page_header, section_header
from tasks import EarlyExit, task


# =====================================================================
# Paths
# =====================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MARKER_BIN = _PROJECT_ROOT / "venv_marker" / "bin" / "marker_single"

# All uploads + parse outputs go under a single temp root so we can
# sweep them if needed.
_TMP_ROOT = Path(tempfile.gettempdir()) / "latelier_pdf"
_TMP_ROOT.mkdir(exist_ok=True)


# =====================================================================
# Session-cache wrappers (keep large markdown out of State)
# =====================================================================
def _get_markdown(state: State) -> str:
    return session_get(state, KEY_PDF_MARKDOWN, "") or ""


def _set_markdown(state: State, md: str) -> None:
    session_set(state, KEY_PDF_MARKDOWN, md)


# =====================================================================
# Event handlers
# =====================================================================
def _on_upload(e: me.UploadEvent):
    """Persist the uploaded bytes to a temp file; store only the path in state."""
    state = me.state(State)
    if not e.files:
        return
    f = e.files[0]
    fd, tmp_path = tempfile.mkstemp(
        prefix="upload_",
        suffix=Path(f.name).suffix or ".pdf",
        dir=_TMP_ROOT,
    )
    with os.fdopen(fd, "wb") as fp:
        fp.write(f.getvalue())

    # Drop the previous upload (if any) so the temp dir doesn't grow.
    if state.pdf_path and os.path.exists(state.pdf_path):
        try:
            os.remove(state.pdf_path)
        except OSError:
            pass

    state.pdf_path = tmp_path
    state.pdf_filename = f.name
    state.pdf_size = f.size
    _set_markdown(state, "")
    state.pdf_has_result = False
    state.status = f"Ready to parse: {f.name}"


def _on_toggle_ocr(e: me.CheckboxChangeEvent):
    me.state(State).pdf_force_ocr = e.checked


def _on_clear(e: me.ClickEvent):
    state = me.state(State)
    if state.pdf_path and os.path.exists(state.pdf_path):
        try:
            os.remove(state.pdf_path)
        except OSError:
            pass
    state.pdf_path = ""
    state.pdf_filename = ""
    state.pdf_size = 0
    _set_markdown(state, "")
    state.pdf_has_result = False
    state.status = "Ready"


def _on_parse(e: me.ClickEvent):
    state = me.state(State)
    if not state.pdf_path:
        state.status = "No PDF selected"
        return
    if not _MARKER_BIN.exists():
        state.status = "Marker is not installed (see page instructions)"
        return
    yield from _run_parse()


# =====================================================================
# The parse pipeline
# =====================================================================
@task("PDF Parse")
def _run_parse(cancel_event):
    state = me.state(State)
    pdf_path = Path(state.pdf_path)
    if not pdf_path.exists():
        state.status = "Uploaded file is no longer available"
        raise EarlyExit()

    # Marker writes into `<output_dir>/<pdf_stem>/<pdf_stem>.md`, so we
    # give it a fresh dir per run and clean it up afterwards.
    out_dir = Path(tempfile.mkdtemp(prefix="out_", dir=_TMP_ROOT))

    cmd = [
        str(_MARKER_BIN),
        str(pdf_path),
        "--output_dir", str(out_dir),
        "--output_format", "markdown",
        "--disable_image_extraction",
        "--disable_tqdm",
    ]
    if state.pdf_force_ocr:
        cmd.append("--force_ocr")

    state.status = "Running Marker..." + (" (force OCR)" if state.pdf_force_ocr else "")
    yield
    if cancel_event.is_set():
        raise EarlyExit()

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        state.status = "Marker timed out after 15 minutes"
        raise EarlyExit()
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:].strip()
        state.status = f"Marker failed (exit {proc.returncode})"
        _set_markdown(state, f"```\n{tail or 'no output'}\n```")
        state.pdf_has_result = True
        raise EarlyExit()

    # Find the produced markdown. Marker uses nested dirs; just rglob.
    md_files = list(out_dir.rglob("*.md"))
    if not md_files:
        state.status = "Marker finished but produced no markdown"
        raise EarlyExit()

    # If multiple files exist, prefer the one named like the input.
    md_files.sort(key=lambda p: (p.stem != pdf_path.stem, len(p.name)))
    md_text = md_files[0].read_text(encoding="utf-8", errors="replace")

    _set_markdown(state, md_text)
    state.pdf_has_result = True
    state.status = (
        f"Done — parsed {state.pdf_filename} in {elapsed:.1f}s ({len(md_text)} chars)"
    )

    # Append one line to the Performance Log.
    entry = (
        f"[PDF]  {state.pdf_filename}  "
        f"force_ocr={state.pdf_force_ocr}  {elapsed:.1f}s  {len(md_text)} chars"
    )
    prev = session_get(state, KEY_PERF_LOG, "") or ""
    session_set(state, KEY_PERF_LOG, (prev + "\n" + entry).strip())


# =====================================================================
# Render
# =====================================================================
def _render():
    state = me.state(State)
    with me.box(style=page_container_style()):
        with me.box(style=me.Style(
            max_width="960px", width="100%",
            display="flex", flex_direction="column", gap="20px",
        )):
            page_header(
                "PDF Parser",
                "Upload a PDF and convert it to markdown with Marker.",
            )

            if not _MARKER_BIN.exists():
                _marker_missing_banner()
                return

            _upload_card(state)
            if state.pdf_has_result:
                _output_card(state)


def _marker_missing_banner():
    with me.box(style=card_style(padding=20, gap="8px")):
        me.text("Marker not found", style=me.Style(
            font_size="14px", font_weight="600", color="#b45309",
        ))
        me.text(
            "Install it into a dedicated virtual environment at `venv_marker/`:",
            style=me.Style(font_size="13px", color=C.text_dim),
        )
        me.text(
            "python3 -m venv venv_marker\n"
            "source venv_marker/bin/activate\n"
            "pip install --upgrade pip\n"
            "pip install marker-pdf",
            style=me.Style(
                font_family="ui-monospace, monospace", font_size="12px",
                background="#fafbfc", color="#333",
                padding=me.Padding.all(12), border_radius=8,
                white_space="pre-wrap",
            ),
        )


def _upload_card(state: State):
    with me.box(style=card_style(padding=24, gap="16px")):
        section_header("Input")

        # Upload row
        with me.box(style=me.Style(display="flex", gap="12px", align_items="center")):
            me.uploader(
                label="Choose PDF",
                accepted_file_types=["application/pdf", ".pdf"],
                on_upload=_on_upload,
                type="flat",
                color="primary",
                disabled=state.is_busy,
                style=me.Style(min_width="140px"),
            )
            if state.pdf_filename:
                me.text(
                    f"{state.pdf_filename}  ·  {_human_size(state.pdf_size)}",
                    style=me.Style(font_size="13px", color=C.text_dim),
                )
            else:
                me.text(
                    "No PDF selected.",
                    style=me.Style(font_size="13px", color=C.text_faint, font_style="italic"),
                )

        # Options
        me.checkbox(
            label="Force OCR (re-OCR every page; use for image-only PDFs)",
            checked=state.pdf_force_ocr,
            on_change=_on_toggle_ocr,
            disabled=state.is_busy,
        )

        # Actions
        with me.box(style=me.Style(display="flex", gap="10px", justify_content="flex-end")):
            me.button(
                "Clear", on_click=_on_clear, type="stroked",
                disabled=state.is_busy or (not state.pdf_filename and not state.pdf_has_result),
                style=me.Style(height="40px", min_width="80px"),
            )
            me.button(
                "Parse with Marker", on_click=_on_parse, type="flat", color="primary",
                disabled=state.is_busy or not state.pdf_filename,
                style=me.Style(height="40px", min_width="160px"),
            )


def _output_card(state: State):
    md = _get_markdown(state)
    with me.box(style=card_style(padding=24, gap="12px")):
        section_header("Markdown output")
        me.text(
            f"{len(md)} characters",
            style=me.Style(font_size="11px", color=C.text_muted),
        )
        with me.box(style=me.Style(
            max_height="640px", overflow_y="auto",
            padding=me.Padding.all(16),
            background="#fafbfc", border_radius=8,
            border=me.Border.all(me.BorderSide(width=1, color=C.border_soft, style="solid")),
        )):
            me.markdown(md)


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    return f"{n / 1024 / 1024 / 1024:.2f} GB"


PAGE = Page(key="pdf", title="PDF Parser", icon="📄", render=_render)
