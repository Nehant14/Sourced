"""
paper_retriever_node (spec section 4.2 / section 8). Mirrors web_retriever_node.
"""
from __future__ import annotations

from app.providers.arxiv import ArxivError, PaperSearchProvider
from app.state import ResearchState
from app.tracing import log_step


def paper_retriever_node(provider: PaperSearchProvider):
    def _run(state: ResearchState) -> dict:
        question = state["question"]
        try:
            results = provider.search(question, max_results=5)
            if not results:
                notes = list(state.get("degradation_notes") or [])
                notes.append("No arXiv paper coverage found for this query.")
                return {
                    "paper_results": [],
                    "degraded": True,
                    "degradation_notes": notes,
                    "trace": log_step(state, "paper_retriever", "0 papers found"),
                }
            return {
                "paper_results": results,
                "trace": log_step(state, "paper_retriever", f"{len(results)} paper(s) retrieved"),
            }
        except ArxivError as e:
            notes = list(state.get("degradation_notes") or [])
            notes.append(f"arXiv provider failed: {e}. Continuing without paper results.")
            return {
                "paper_results": [],
                "paper_error": str(e),
                "degraded": True,
                "degradation_notes": notes,
                "trace": log_step(state, "paper_retriever", f"provider error, degraded: {e}"),
            }

    return _run
