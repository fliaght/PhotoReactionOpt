"""
tasks.py — Global task queue and the `@task` decorator.

Why
====
Mesop sessions run independently: click events from different users
arrive on different threads with their own `State`. If two users click
"Send" at the same time and each loads a 12B model, the GPU OOMs.

This module provides a single process-wide scheduler, `TASK_MANAGER`,
through which every long-running generator event handler is funnelled.
One task runs at a time; others show "Queued (position N)..." and can
be cancelled individually from the Performance page.

The `@task(label)` decorator is the only thing page authors normally
need. See docstring on `task` below.

Key types
---------
* `QueueEntry` — one submitted task. Carries id, owning session, page,
  human-readable label, timestamps, status, and a `cancel_event` that
  the running work checks.
* `TaskManager` — thread-safe FIFO with `submit`, `try_start`,
  `finish`, `cancel`, `position`, `snapshot`.
* `EarlyExit` — sentinel raised inside a task body to abort early
  while still running cleanup AND the post-finally re-render.
"""

from __future__ import annotations

import functools
import threading
import time
import uuid
from dataclasses import dataclass, field

import mesop as me

from state import State, ensure_session_id


# =====================================================================
# Status constants
# =====================================================================
TASK_QUEUED = "queued"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_CANCELLED = "cancelled"
TASK_ERROR = "error"

TERMINAL_STATUSES = (TASK_DONE, TASK_CANCELLED, TASK_ERROR)


# =====================================================================
# Queue entry
# =====================================================================
@dataclass
class QueueEntry:
    id: str
    session_id: str
    page_key: str
    label: str
    submitted_at: float
    started_at: float = 0.0
    finished_at: float = 0.0
    status: str = TASK_QUEUED
    error: str = ""
    # Signal the running task should poll to abort. Not serialized.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


# =====================================================================
# Manager
# =====================================================================
class TaskManager:
    """Single-slot FIFO scheduler shared across sessions.

    Thread-safe. The manager itself does not spawn threads; each
    `@task`-wrapped handler runs on whatever thread Mesop dispatched
    the event on, and cooperates by calling these methods between
    `yield`s.

    Contracts:
      * `submit` always appends a new entry in FIFO order.
      * `try_start` atomically promotes `entry` to RUNNING iff (a) no
        other entry is RUNNING and (b) `entry` is the head of the
        queued list. Returns True on promotion OR if already running.
      * `finish` sets a terminal status and timestamps.
      * `cancel` sets the entry's `cancel_event` and — if the entry was
        still queued — marks it CANCELLED immediately.
      * `snapshot` returns a consistent copy (recent history trimmed
        to `max_history`).
    """

    def __init__(self, max_history: int = 50):
        self._lock = threading.Lock()
        self._entries: list[QueueEntry] = []
        self._max_history = max_history

    # ------------- mutators -------------
    def submit(self, session_id: str, page_key: str, label: str) -> QueueEntry:
        entry = QueueEntry(
            id=str(uuid.uuid4())[:8],
            session_id=session_id,
            page_key=page_key,
            label=label,
            submitted_at=time.time(),
        )
        with self._lock:
            self._entries.append(entry)
            self._trim_locked()
        return entry

    def try_start(self, entry: QueueEntry) -> bool:
        """Promote `entry` from QUEUED to RUNNING if allowed. Idempotent."""
        with self._lock:
            if entry.status == TASK_RUNNING:
                return True
            if entry.status in TERMINAL_STATUSES:
                return False
            for e in self._entries:
                if e.status == TASK_RUNNING and e.id != entry.id:
                    return False  # someone else is running
            for e in self._entries:
                if e.status == TASK_QUEUED:
                    if e.id == entry.id:
                        entry.status = TASK_RUNNING
                        entry.started_at = time.time()
                        return True
                    return False  # someone queued ahead
        return False

    def finish(self, entry: QueueEntry, status: str, error: str = "") -> None:
        assert status in TERMINAL_STATUSES
        with self._lock:
            entry.status = status
            entry.finished_at = time.time()
            if error:
                entry.error = error

    def cancel(self, entry_id: str) -> bool:
        with self._lock:
            for e in self._entries:
                if e.id != entry_id:
                    continue
                if e.status == TASK_QUEUED:
                    e.status = TASK_CANCELLED
                    e.finished_at = time.time()
                    e.cancel_event.set()
                    return True
                if e.status == TASK_RUNNING:
                    e.cancel_event.set()
                    return True
                return False  # already terminal
        return False

    # ------------- readers -------------
    def position(self, entry: QueueEntry) -> int:
        """1-indexed position among QUEUED entries. 0 if running / terminal."""
        if entry.status != TASK_QUEUED:
            return 0
        with self._lock:
            pos = 0
            for e in self._entries:
                if e.status == TASK_QUEUED:
                    pos += 1
                    if e.id == entry.id:
                        return pos
        return 0

    def snapshot(self) -> list[QueueEntry]:
        with self._lock:
            return list(self._entries)

    # ------------- internals -------------
    def _trim_locked(self) -> None:
        active = [e for e in self._entries if e.status not in TERMINAL_STATUSES]
        history = [e for e in self._entries if e.status in TERMINAL_STATUSES]
        history = history[-self._max_history:]
        self._entries = active + history


TASK_MANAGER = TaskManager()


# =====================================================================
# EarlyExit + @task decorator
# =====================================================================
class EarlyExit(Exception):
    """Raise inside a `@task` body to abort gracefully.

    Unlike `return`, which would skip the decorator's post-finally
    re-render and leave the sidebar greyed out, `EarlyExit` is caught
    by the `@task` wrapper. The body's own `try/finally` still runs,
    then the wrapper flips `is_busy=False` and yields once more.
    """


def task(label: str):
    """Wrap a generator event handler in the global task queue.

    Usage
    -----
    Decorate a generator function that takes `cancel_event` as its
    first parameter. Page-level event handlers like `_on_send`
    forward into the decorated function via `yield from`.

        @task("LLM Duel")
        def _run_duel(cancel_event: threading.Event):
            state = me.state(State)
            state.status = "Working..."
            yield
            try:
                if cancel_event.is_set():
                    raise EarlyExit()
                # ... do work, yield periodically ...
            finally:
                release_resources()

    Guarantees
    ----------
    * Returns immediately if `state.is_busy` (one task per session).
    * Submits an entry to `TASK_MANAGER` and blocks (by yielding) until
      it's this task's turn, showing "Queued (position N)..." in status.
    * Forwards the entry's `cancel_event` as the first positional arg
      to the wrapped function.
    * Catches `EarlyExit` (-> TASK_CANCELLED) and any other Exception
      (-> TASK_ERROR, message written to status).
    * ALWAYS runs a final `state.is_busy = False; yield` so the UI
      re-renders with the sidebar enabled.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            state = me.state(State)
            if state.is_busy:
                return  # this session already has a task

            sid = ensure_session_id(state)
            entry = TASK_MANAGER.submit(
                session_id=sid,
                page_key=state.current_page,
                label=label,
            )
            state.is_busy = True
            state.my_task_id = entry.id
            yield  # initial render with is_busy=True

            try:
                # Poll for turn; yield between checks so the UI stays responsive.
                while not TASK_MANAGER.try_start(entry):
                    if entry.cancel_event.is_set():
                        state.status = "Cancelled"
                        raise EarlyExit()
                    pos = TASK_MANAGER.position(entry)
                    state.status = (
                        f"Queued (position {pos})..." if pos > 0 else "Queued..."
                    )
                    yield
                    time.sleep(0.3)

                # Now running.
                yield from fn(entry.cancel_event, *args, **kwargs)
                TASK_MANAGER.finish(entry, TASK_DONE)

            except EarlyExit:
                TASK_MANAGER.finish(entry, TASK_CANCELLED)
            except Exception as ex:
                state.status = f"Error: {ex}"
                TASK_MANAGER.finish(entry, TASK_ERROR, error=str(ex))

            # Always: reset busy flag and yield so sidebar re-enables.
            state.is_busy = False
            state.my_task_id = ""
            yield

        return wrapper
    return decorator
