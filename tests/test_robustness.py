"""
Spec section 8: "Write these as actual test cases (mock the provider to
raise/timeout/return empty) - this is what makes 'graceful degradation' a
demonstrated fact instead of a claim."
"""
from app.graph import build_graph
from app.nodes.reconciler import reconciler_node
from app.providers.arxiv import ArxivError, PaperSearchProvider
from app.providers.llm import LLMError, LLMProvider
from app.providers.search import WebSearchError, WebSearchProvider
from app.schemas import PaperReference, SourceReference


class AlwaysFailsWeb(WebSearchProvider):
    def search(self, query, max_results=5):
        raise WebSearchError("simulated network failure")


class AlwaysFailsPapers(PaperSearchProvider):
    def search(self, query, max_results=5):
        raise ArxivError("simulated arxiv outage")


class AlwaysFailsLLM(LLMProvider):
    def generate_text(self, prompt, system=None):
        raise LLMError("simulated LLM outage")

    def generate_structured(self, prompt, schema, system=None, context=None):
        raise LLMError("simulated LLM outage")


def test_web_provider_down_degrades_to_papers_only(llm, paper_provider):
    graph = build_graph(llm, AlwaysFailsWeb(), paper_provider)
    out = graph.invoke(
        {"question": "What's the latest research on model scaling laws?", "retry_count": 0, "trace": []}
    )
    final = out["final_answer"]
    assert final is not None, "Must degrade, not crash"
    assert final.degraded is True
    assert any("web" in n.lower() for n in final.degradation_notes)


def test_paper_provider_down_degrades_to_web_only(llm, web_provider):
    graph = build_graph(llm, web_provider, AlwaysFailsPapers())
    out = graph.invoke(
        {"question": "Compare Python vs Rust for backend services", "retry_count": 0, "trace": []}
    )
    final = out["final_answer"]
    assert final is not None, "Must degrade, not crash"
    assert final.degraded is True


def test_both_providers_down_still_produces_a_state_not_a_crash(llm):
    graph = build_graph(llm, AlwaysFailsWeb(), AlwaysFailsPapers())
    out = graph.invoke({"question": "any question", "retry_count": 0, "trace": []})
    # Per spec: "return a clear error state, not a crash, not a hallucinated
    # answer with no sources." We still produce a final_answer object (so the
    # API layer has something well-formed to return), but it must be clearly
    # marked degraded with zero sources used, and confidence must reflect that.
    final = out["final_answer"]
    assert final is not None
    assert final.degraded is True
    assert final.sources_used == {"web": 0, "papers": 0}
    assert final.confidence < 0.6, "Confidence should be heavily penalized with zero sources"


def test_reconciler_survives_malformed_llm_output():
    node = reconciler_node(AlwaysFailsLLM())
    web = SourceReference(
        source_id="web_1",
        title="t",
        url="https://example.com",
        snippet="some snippet",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    out = node({"question": "q", "web_results": [web], "paper_results": []})
    assert out["conflict_detection_unavailable"] is True
    assert out["conflicts"] == []


def test_synthesizer_survives_llm_failure():
    from app.nodes.synthesizer import synthesizer_node

    node = synthesizer_node(AlwaysFailsLLM())
    out = node({"question": "q", "web_results": [], "paper_results": [], "retry_count": 0})
    assert out["draft_answer"], "Should return a fallback message, not crash"


def test_planner_survives_llm_failure():
    from app.nodes.planner import planner_node

    node = planner_node(AlwaysFailsLLM())
    out = node({"question": "q"})
    assert out["needs_web"] is True and out["needs_papers"] is True
