"""
Graph wiring (spec section 5):

    planner -> (conditional fan-out) -> web_retriever, paper_retriever
    web_retriever, paper_retriever -> reconciler
    reconciler -> synthesizer
    synthesizer -> validator
    validator -> (conditional) -> synthesizer   [retry]
    validator -> (conditional) -> END           [pass, or retries exhausted]

web_retriever and paper_retriever run in parallel when both are needed:
LangGraph executes independent nodes reached in the same superstep
concurrently and joins before reconciler runs - that's native fan-out/fan-in,
not something we have to implement ourselves. (Verify this in your own
LangSmith trace by checking the timestamps overlap - see spec section 9.)
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.nodes.paper_retriever import paper_retriever_node
from app.nodes.planner import planner_node, route_after_planner
from app.nodes.reconciler import reconciler_node
from app.nodes.synthesizer import synthesizer_node
from app.nodes.validator import route_after_validator, validator_node
from app.nodes.web_retriever import web_retriever_node
from app.providers.arxiv import PaperSearchProvider
from app.providers.llm import LLMProvider
from app.providers.search import WebSearchProvider
from app.state import ResearchState


def build_graph(
    llm: LLMProvider,
    web_provider: WebSearchProvider,
    paper_provider: PaperSearchProvider,
):
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node(llm))
    graph.add_node("web_retriever", web_retriever_node(web_provider))
    graph.add_node("paper_retriever", paper_retriever_node(paper_provider))
    graph.add_node("reconciler", reconciler_node(llm))
    graph.add_node("synthesizer", synthesizer_node(llm))
    graph.add_node("validator", validator_node(llm))

    graph.set_entry_point("planner")

    # Fan-out: planner routes to a SUBSET of {web_retriever, paper_retriever}
    # (or straight to reconciler in the never-should-happen neither case).
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "web_retriever": "web_retriever",
            "paper_retriever": "paper_retriever",
            "reconciler": "reconciler",
        },
    )

    # Fan-in: both retrievers feed the reconciler. LangGraph waits for all
    # active branches of the current superstep before running reconciler.
    graph.add_edge("web_retriever", "reconciler")
    graph.add_edge("paper_retriever", "reconciler")

    graph.add_edge("reconciler", "synthesizer")
    graph.add_edge("synthesizer", "validator")

    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "synthesizer": "synthesizer",
            "end": END,
        },
    )

    return graph.compile()
