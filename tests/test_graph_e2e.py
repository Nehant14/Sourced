from app.graph import build_graph


def test_e2e_simple_question(llm, web_provider, paper_provider):
    graph = build_graph(llm, web_provider, paper_provider)
    out = graph.invoke({"question": "What is the capital of Australia?", "retry_count": 0, "trace": []})
    assert out["final_answer"] is not None
    assert out["final_answer"].confidence > 0
    assert len(out["trace"]) >= 5  # planner, >=1 retriever, reconciler, synthesizer, validator


def test_e2e_deliberate_conflict_question_surfaces_conflict(llm, web_provider, paper_provider):
    graph = build_graph(llm, web_provider, paper_provider)
    out = graph.invoke(
        {
            "question": "Is there a controversy over whether RAG is still necessary given long context windows?",
            "retry_count": 0,
            "trace": [],
        }
    )
    final = out["final_answer"]
    assert len(final.conflicts) >= 1
    # the validator/synthesizer retry loop should have forced the answer to
    # actually mention the disagreement by the time we reach END
    assert any(w in final.answer.lower() for w in ["disagree", "conflict", "differ", "note:"])


def test_e2e_compound_question_gets_sub_questions(llm, web_provider, paper_provider):
    graph = build_graph(llm, web_provider, paper_provider)
    out = graph.invoke(
        {"question": "Compare PostgreSQL vs MongoDB for a web app", "retry_count": 0, "trace": []}
    )
    assert out["final_answer"] is not None


def test_e2e_never_exceeds_max_retries(llm, web_provider, paper_provider):
    graph = build_graph(llm, web_provider, paper_provider)
    out = graph.invoke({"question": "asdkjqwoieqwoi nonsense zzz111", "retry_count": 0, "trace": []})
    assert out["final_answer"].retries_used <= 2
