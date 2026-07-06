"""
planner_node (spec section 4.1).

Design decision on the "no-retrieval path" question the spec explicitly
raises: we FORCE at least one retrieval path even when the LLM is confident
it already knows the answer. Rationale, documented here rather than buried:
this is a RAG system whose entire value proposition is citation-backed,
reconciled answers. A no-retrieval path would produce answers with zero
citations, which the validator would then always reject anyway (see
validator_node), just burning a wasted retry. So: if the LLM's plan sets
both needs_web and needs_papers to False, we override needs_web to True.
The LLM's own reasoning is preserved in plan_reasoning either way, so this
override is visible in the trace, not silent.
"""
from __future__ import annotations

from app.providers.llm import LLMError, LLMProvider
from app.schemas import PlannerDecision
from app.state import ResearchState
from app.tracing import log_step

PLANNER_SYSTEM_PROMPT = """You are the planning stage of a research assistant.
Given a user's question, decide:
1. Whether it is a compound/comparative question that should be split into
   independent sub-questions (empty list if it's already atomic).
2. Whether it needs general web search (needs_web) - true for current events,
   product/tool comparisons, opinion/news, anything time-sensitive.
3. Whether it needs academic paper search (needs_papers) - true for
   "latest research", technical/scientific claims, anything where an arXiv
   paper would materially help answer accurately.
A question can need both, one, or (rarely, if you are highly confident and
the question is simple trivia) neither - but explain your reasoning either way.
"""


def planner_node(llm: LLMProvider):
    def _run(state: ResearchState) -> dict:
        question = state["question"]
        prompt = f"Question: {question}\n\nProduce the plan."
        try:
            decision: PlannerDecision = llm.generate_structured(
                prompt=prompt,
                schema=PlannerDecision,
                system=PLANNER_SYSTEM_PROMPT,
                context={"question": question},
            )
        except LLMError:
            # Fail-safe default plan if even the planner call fails: try both
            # retrieval paths rather than crash. This is the same "graceful
            # degradation over crash" principle from spec section 8, applied
            # to the planner itself even though the spec only calls it out
            # for the retrievers.
            return {
                "sub_questions": [],
                "needs_web": True,
                "needs_papers": True,
                "plan_reasoning": "planner LLM call failed; defaulting to both retrieval paths",
                "trace": log_step(state, "planner", "LLM call failed, defaulted to needs_web=needs_papers=True"),
            }

        needs_web = decision.needs_web
        needs_papers = decision.needs_papers
        reasoning = decision.reasoning
        if not needs_web and not needs_papers:
            needs_web = True
            reasoning += " [overridden: forcing needs_web=True so the answer has at least one citable source]"

        return {
            "sub_questions": decision.sub_questions,
            "needs_web": needs_web,
            "needs_papers": needs_papers,
            "plan_reasoning": reasoning,
            "retry_count": 0,
            "trace": log_step(
                state,
                "planner",
                f"needs_web={needs_web}, needs_papers={needs_papers}, "
                f"sub_questions={decision.sub_questions}",
            ),
        }

    return _run


def route_after_planner(state: ResearchState) -> list[str]:
    """Conditional edge: fan out to whichever retrievers are actually needed.
    This is the mechanism, not an if-statement inside a single retriever
    node - see spec section 4.2."""
    targets = []
    if state.get("needs_web"):
        targets.append("web_retriever")
    if state.get("needs_papers"):
        targets.append("paper_retriever")
    if not targets:
        # Should not happen given the forced override above, but keep the
        # graph well-defined regardless.
        targets.append("reconciler")
    return targets
