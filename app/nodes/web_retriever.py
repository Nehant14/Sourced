"""
web_retriever_node (spec section 4.2 / section 8).

Only reached via the conditional edge out of the planner when needs_web is
true, so no internal `if` guard here. Must return partial results and
continue rather than crash on provider failure.
"""
from __future__ import annotations

from app.providers.search import WebSearchError, WebSearchProvider
from app.state import ResearchState
from app.tracing import log_step


def web_retriever_node(provider: WebSearchProvider):
    def _run(state: ResearchState) -> dict:
        question = state["question"]
        try:
            results = provider.search(question, max_results=5)
            if not results:
                notes = list(state.get("degradation_notes") or [])
                notes.append("Web search returned zero results for this query.")
                return {
                    "web_results": [],
                    "degraded": True,
                    "degradation_notes": notes,
                    "trace": log_step(state, "web_retriever", "0 results returned"),
                }
            return {
                "web_results": results,
                "trace": log_step(state, "web_retriever", f"{len(results)} result(s) retrieved"),
            }
        except WebSearchError as e:
            notes = list(state.get("degradation_notes") or [])
            notes.append(f"Web search provider failed: {e}. Continuing without web results.")
            return {
                "web_results": [],
                "web_error": str(e),
                "degraded": True,
                "degradation_notes": notes,
                "trace": log_step(state, "web_retriever", f"provider error, degraded: {e}"),
            }

    return _run
