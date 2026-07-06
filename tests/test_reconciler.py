from app.nodes.reconciler import reconciler_node
from app.schemas import PaperReference, SourceReference


def _sample_sources():
    web = SourceReference(
        source_id="web_1",
        title="Some article",
        url="https://example.com/a",
        snippet="RAG is still widely used in production systems as of this year.",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    paper = PaperReference(
        source_id="paper_1",
        title="A Study on Retrieval",
        url="https://arxiv.org/abs/1234.5678",
        arxiv_id="1234.5678",
        authors=["A. Author"],
        published="2025-01-01",
        abstract="RAG remains a widely used technique for grounding LLM outputs.",
    )
    return web, paper


def test_no_conflict_on_agreeing_sources(llm):
    web, paper = _sample_sources()
    node = reconciler_node(llm)
    out = node(
        {
            "question": "How widely used is RAG today?",
            "web_results": [web],
            "paper_results": [paper],
        }
    )
    assert out["conflicts"] == [], "Agreeing sources should not produce a manufactured conflict"


def test_conflict_detected_when_question_signals_disagreement(llm):
    web, paper = _sample_sources()
    node = reconciler_node(llm)
    out = node(
        {
            "question": "Is there a controversy over whether RAG is still necessary?",
            "web_results": [web],
            "paper_results": [paper],
        }
    )
    assert len(out["conflicts"]) >= 1


def test_no_sources_short_circuits_without_llm_call(llm):
    node = reconciler_node(llm)
    out = node({"question": "anything", "web_results": [], "paper_results": []})
    assert out["claims"] == []
    assert out["conflicts"] == []
