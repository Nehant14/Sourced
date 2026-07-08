"""
Graph wiring (spec section 5):

    planner -> (conditional fan-out) -> web_retriever, paper_retriever
    web_retriever, paper_retriever -> relevance_filter
    relevance_filter -> (conditional) -> query_reformulator   [confidence too low, retry budget left]
    relevance_filter -> (conditional) -> reconciler            [confidence OK, or retry budget spent]
    query_reformulator -> (conditional fan-out, same rule as planner) -> web_retriever, paper_retriever
    reconciler -> synthesizer
    synthesizer -> validator
    validator -> (conditional) -> synthesizer   [retry]
    validator -> (conditional) -> END           [pass, or retries exhausted]

web_retriever and paper_retriever run in parallel when both are needed:
LangGraph executes independent nodes reached in the same superstep
concurrently and joins before relevance_filter runs - that's native
fan-out/fan-in, not something we have to implement ourselves. (Verify this
in your own LangSmith trace by checking the timestamps overlap - see spec
section 9.)

relevance_filter's retry (via query_reformulator) is a SEPARATE budget
(retrieval_retry_count, capped at 1) from the synthesizer/validator retry
(retry_count, capped at 2) - they retry different things at different
costs and shouldn't be conflated. See relevance_filter.py and
query_reformulator.py docstrings for why.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.nodes.paper_retriever import paper_retriever_node
from app.nodes.planner import planner_node, route_after_planner
from app.nodes.query_reformulator import query_reformulator_node
from app.nodes.reconciler import reconciler_node
from app.nodes.relevance_filter import relevance_filter_node, route_after_relevance_filter
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
    graph.add_node("relevance_filter", relevance_filter_node(llm))
    graph.add_node("query_reformulator", query_reformulator_node(llm))
    graph.add_node("reconciler", reconciler_node(llm))
    graph.add_node("synthesizer", synthesizer_node(llm))
    graph.add_node("validator", validator_node(llm))

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "web_retriever": "web_retriever",
            "paper_retriever": "paper_retriever",
            "reconciler": "reconciler",
        },
    )

    graph.add_edge("web_retriever", "relevance_filter")
    graph.add_edge("paper_retriever", "relevance_filter")

    graph.add_conditional_edges(
        "relevance_filter",
        route_after_relevance_filter,
        {
            "query_reformulator": "query_reformulator",
            "reconciler": "reconciler",
        },
    )

    graph.add_conditional_edges(
        "query_reformulator",
        route_after_planner,
        {
            "web_retriever": "web_retriever",
            "paper_retriever": "paper_retriever",
            "reconciler": "reconciler",
        },
    )

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