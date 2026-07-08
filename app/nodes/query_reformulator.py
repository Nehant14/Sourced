"""
query_reformulator_node.

Only reached when route_after_relevance_filter (app/nodes/relevance_filter.py)
decides too few relevant sources survived and the retrieval retry budget
hasn't been used yet. Asks the LLM to rewrite the question into a better
search query, then routes back to whichever retriever(s) were originally
needed (reuses route_after_planner - the needs_web/needs_papers decision
doesn't change on retry, only the query does).

Capped at 1 retry (see MAX_RETRIEVAL_RETRIES in relevance_filter.py) -
unlike the synthesizer/validator retry loop, each attempt here re-runs
retrieval + the relevance filter itself, so it costs meaningfully more per
attempt. It's also not always fixable: a corpus with no genuine coverage of
a topic (e.g. arXiv for a biomedical toxicology question) won't produce
relevant results no matter how the query is phrased. One attempt is enough
to catch a genuinely bad query without chasing an unwinnable case.
"""
from __future__ import annotations

from app.providers.llm import LLMError, LLMProvider
from app.schemas import QueryReformulation
from app.state import ResearchState
from app.tracing import log_step

REFORMULATE_SYSTEM_PROMPT = """The initial search query for this question
returned mostly irrelevant results. Rewrite it as a more effective search
query - more specific, using domain terminology if applicable, or rephrased
to target what's actually being asked rather than surface keywords. Return
only the improved query and a short reason."""


def query_reformulator_node(llm: LLMProvider):
    def _run(state: ResearchState) -> dict:
        question = state["question"]
        retry_count = state.get("retrieval_retry_count", 0) or 0

        try:
            reform: QueryReformulation = llm.generate_structured(
                prompt=f"Original question/query: {question}\n\nReformulate for better retrieval.",
                schema=QueryReformulation,
                system=REFORMULATE_SYSTEM_PROMPT,
                context={"question": question},
            )
            new_query = reform.reformulated_query
            reform_note = f"Reformulated query for retry: '{new_query}' ({reform.reasoning})"
        except LLMError as e:
            new_query = question
            reform_note = f"Query reformulation unavailable ({e}); retrying with original query."

        notes = list(state.get("degradation_notes") or [])
        notes.append(reform_note)

        return {
            "web_query_override": new_query,
            "paper_query_override": new_query,
            "retrieval_retry_count": retry_count + 1,
            "degradation_notes": notes,
            "trace": log_step(state, "query_reformulator", reform_note),
        }

    return _run