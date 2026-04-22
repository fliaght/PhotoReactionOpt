"""
pages package — registry of all feature pages.

To add a new page:

1. Create `pages/<name>.py` following this template:

    import mesop as me
    from shell import State, page_header, page_container_style, page_inner_style, Page

    def render():
        state = me.state(State)
        with me.box(style=page_container_style()):
            with me.box(style=page_inner_style()):
                page_header("My Page", "Short description.")
                # ... render UI ...

    # Optional: event handlers for this page go here too.
    # For long tasks, decorate with shell.task and raise shell.EarlyExit
    # for graceful early exit.

    PAGE = Page(key="my", title="My Page", icon="✨", render=render)

2. Add an import below and append to the PAGES tuple.

The `PAGES` order determines sidebar nav order.
"""

from . import duel, pdf_parse, settings, performance, about

PAGES = (
    duel.PAGE,
    pdf_parse.PAGE,
    settings.PAGE,
    performance.PAGE,
    about.PAGE,
)
