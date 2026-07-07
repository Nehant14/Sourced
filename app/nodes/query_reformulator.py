"""query_reformulator_node (new): rewrites the retrieval query when relevance
filters indicate retrieval was not sufficiently relevant.
"""
from __future__ import annotations

from app.providers.llm import LLMError, LLMProvider
from app.schemas import QueryReformulationResult
from app.state import ResearchState
from app.tracing import log_step

REFORMULATE_SYSTEM_PROMPT = """You are given a user's question. Rewrite it into a
retrieval query that is likely to return sources relevant to that question.
Keep the intent the same but make the query more focused for search.
"""


def query_reformulator_node(llm: LLMProvider):
    def _run(state: ResearchState) -> dict:
        question = state["question"]
        try:
            reformulation: QueryReformulationResult = llm.generate_structured(
                prompt=f"Question: {question}\n\nRewrite this query for retrieval.",
                schema=QueryReformulationResult,
                system=REFORMULATE_SYSTEM_PROMPT,
                context={"question": question},
            )
            reformulated_query = reformulation.reformulated_query.strip() or question
        except LLMError:
            reformulated_query = question

        retry_count = (state.get("retrieval_retry_count") or 0) + 1
        return {
            "retrieval_query": reformulated_query,
            "retrieval_retry_count": retry_count,
            "trace": log_step(
                state,
                "query_reformulator",
                f"retrieval query reformulated (retrieval_retry_count={retry_count})",
            ),
        }

    return _run
