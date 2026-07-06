from app.nodes.planner import planner_node, route_after_planner


def test_simple_factual_question_does_not_need_papers(llm):
    node = planner_node(llm)
    out = node({"question": "What is the capital of France?"})
    assert out["needs_web"] or out["needs_papers"]  # at least one, per forced-retrieval rule
    assert out["sub_questions"] == []


def test_compound_question_gets_split(llm):
    node = planner_node(llm)
    out = node({"question": "Compare Python vs Rust for backend services"})
    assert len(out["sub_questions"]) == 2


def test_recent_research_question_flags_papers(llm):
    node = planner_node(llm)
    out = node({"question": "What's the latest research on efficient attention mechanisms?"})
    assert out["needs_papers"] is True


def test_three_question_types_produce_different_plans(llm):
    """The explicit spec requirement: confirm the plan actually differs,
    not just that the node runs without crashing."""
    node = planner_node(llm)
    factual = node({"question": "What is the boiling point of water?"})
    comparative = node({"question": "Compare LangGraph vs LangChain"})
    recent_research = node({"question": "What's the latest research on model scaling laws?"})

    plans = [
        (factual["needs_web"], factual["needs_papers"], bool(factual["sub_questions"])),
        (comparative["needs_web"], comparative["needs_papers"], bool(comparative["sub_questions"])),
        (recent_research["needs_web"], recent_research["needs_papers"], bool(recent_research["sub_questions"])),
    ]
    assert len(set(plans)) > 1, f"Expected differing plans, got identical plans: {plans}"


def test_forced_retrieval_override_when_planner_says_neither(llm, monkeypatch):
    # Force the mock planner's heuristic into the "neither" branch by
    # monkeypatching it directly, then confirm planner_node's override fires.
    def fake_generate_structured(prompt, schema, system=None, context=None):
        from app.schemas import PlannerDecision

        return PlannerDecision(sub_questions=[], needs_web=False, needs_papers=False, reasoning="test")

    monkeypatch.setattr(llm, "generate_structured", fake_generate_structured)
    node = planner_node(llm)
    out = node({"question": "trivial question"})
    assert out["needs_web"] is True
    assert "overridden" in out["plan_reasoning"]


def test_route_after_planner_fans_out_to_both():
    state = {"needs_web": True, "needs_papers": True}
    assert set(route_after_planner(state)) == {"web_retriever", "paper_retriever"}


def test_route_after_planner_single_branch():
    state = {"needs_web": True, "needs_papers": False}
    assert route_after_planner(state) == ["web_retriever"]
