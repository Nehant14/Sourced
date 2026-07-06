"""
Lightweight, dependency-free execution trace.

This is deliberately separate from LangSmith: LangSmith (wired up via env
vars per spec section 9) gives you the rich, shareable trace UI for
screenshots/demos. This module gives every API response a visible,
self-contained trace even if LangSmith isn't configured - useful for the
eval harness and for debugging without leaving the terminal.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.state import ResearchState


def log_step(state: ResearchState, node: str, summary: str) -> list[dict]:
    """Returns a single trace entry to be merged by the graph reducer.
    Nodes call this and include the result under the 'trace' key of their
    returned partial state update."""
    return [
        {
            "node": node,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        }
    ]
