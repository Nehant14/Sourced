import pytest

from app.nodes.paper_retriever import paper_retriever_node
from app.nodes.web_retriever import web_retriever_node
from app.providers.arxiv import ArxivError, PaperSearchProvider
from app.providers.search import WebSearchError, WebSearchProvider


def test_web_retriever_returns_results(web_provider):
    node = web_retriever_node(web_provider)
    out = node({"question": "quantum computing breakthroughs"})
    assert len(out["web_results"]) == 3
    assert not out.get("degraded")


def test_web_retriever_zero_results_degrades_not_crashes(web_provider):
    node = web_retriever_node(web_provider)
    out = node({"question": "noresultstest weather today"})
    assert out["web_results"] == []
    assert out["degraded"] is True
    assert out["degradation_notes"]


def test_web_retriever_provider_exception_degrades_not_crashes():
    class FailingProvider(WebSearchProvider):
        def search(self, query, max_results=5):
            raise WebSearchError("simulated timeout")

    node = web_retriever_node(FailingProvider())
    out = node({"question": "anything"})
    assert out["web_results"] == []
    assert out["web_error"] == "simulated timeout"
    assert out["degraded"] is True


def test_paper_retriever_returns_results(paper_provider):
    node = paper_retriever_node(paper_provider)
    out = node({"question": "transformer scaling laws"})
    assert len(out["paper_results"]) == 2


def test_paper_retriever_no_coverage_degrades_not_crashes(paper_provider):
    node = paper_retriever_node(paper_provider)
    out = node({"question": "nopapertest obscure topic"})
    assert out["paper_results"] == []
    assert out["degraded"] is True


def test_paper_retriever_provider_exception_degrades_not_crashes():
    class FailingProvider(PaperSearchProvider):
        def search(self, query, max_results=5):
            raise ArxivError("simulated arxiv outage")

    node = paper_retriever_node(FailingProvider())
    out = node({"question": "anything"})
    assert out["paper_results"] == []
    assert out["paper_error"] == "simulated arxiv outage"
    assert out["degraded"] is True
