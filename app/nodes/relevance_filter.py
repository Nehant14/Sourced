"""relevance_filter_node (new): filters retrieved sources before reconciliation.

This node judges every retrieved web and paper source for question relevance
in a single structured LLM call. It preserves the original retrieval results
with per-source relevance annotations, and it produces filtered lists that
are used downstream.
"""
from __future__ import annotations

from app.confidence import compute_confidence
from app.providers.llm import LLMError, LLMProvider
from app.schemas import PaperReference, RelevanceAnalysis, SourceReference
from app.state import ResearchState
from app.tracing import log_step

RELEVANCE_SYSTEM_PROMPT = """You are given a user question and a set of sources.
For each source, judge whether the title plus snippet/abstract is relevant to
answering the question. Return a true/false verdict for each source_id.
Only mark a source relevant if it contains information that helps address the
question directly.
"""

CONFIDENCE_RETRY_THRESHOLD = 0.70


def _effective_web_results(state: ResearchState) -> list[SourceReference]:
    return state.get("filtered_web_results") or state.get("web_results") or []


def _effective_paper_results(state: ResearchState) -> list[PaperReference]:
    return state.get("filtered_paper_results") or state.get("paper_results") or []


def _sources_to_context(state: ResearchState) -> list[dict]:
    sources: list[dict] = []
    for s in state.get("web_results") or []:
        sources.append({
            "source_id": s.source_id,
            "title": s.title,
            "snippet": s.snippet,
            "type": "web",
        })
    for p in state.get("paper_results") or []:
        sources.append({
            "source_id": p.source_id,
            "title": p.title,
            "abstract": p.abstract,
            "type": "paper",
        })
    return sources


def _annotated_source(source: dict, relevant: bool) -> dict:
    return {**source, "relevant_to_question": relevant}


def _filtered_results_by_relevance(
    raw_results: list[SourceReference] | list[PaperReference],
    relevance_map: dict[str, bool],
) -> list[SourceReference] | list[PaperReference]:
    return [r for r in raw_results if relevance_map.get(r.source_id, False)]


def _build_relevance_map(analysis: RelevanceAnalysis) -> dict[str, bool]:
    return {item.source_id: item.relevant_to_question for item in analysis.relevances}


def relevance_filter_node(llm: LLMProvider):
    def _run(state: ResearchState) -> dict:
        sources = _sources_to_context(state)
        if not sources:
            return {
                "filtered_web_results": [],
                "filtered_paper_results": [],
                "web_results_with_relevance": [],
                "paper_results_with_relevance": [],
                "trace": log_step(state, "relevance_filter", "no sources to judge relevance"),
            }

        try:
            analysis: RelevanceAnalysis = llm.generate_structured(
                prompt=(
                    f"Question: {state['question']}\n\n"
                    "Sources:\n"
                    + "\n".join(
                        f"[{src['source_id']}] {src['title']}\n{src.get('snippet', src.get('abstract',''))}"
                        for src in sources
                    )
                ),
                schema=RelevanceAnalysis,
                system=RELEVANCE_SYSTEM_PROMPT,
                context={"question": state["question"], "sources": sources},
            )
        except LLMError:
            return {
                "filtered_web_results": state.get("web_results") or [],
                "filtered_paper_results": state.get("paper_results") or [],
                "web_results_with_relevance": [
                    _annotated_source(s, True) for s in state.get("web_results") or []
                ],
                "paper_results_with_relevance": [
                    _annotated_source(p, True) for p in state.get("paper_results") or []
                ],
                "trace": log_step(state, "relevance_filter", "LLM relevance judgment failed, preserving raw sources"),
            }

        relevance_map = _build_relevance_map(analysis)
        filtered_web = _filtered_results_by_relevance(state.get("web_results") or [], relevance_map)
        filtered_papers = _filtered_results_by_relevance(state.get("paper_results") or [], relevance_map)

        web_annotations = [
            _annotated_source(s.model_dump(), relevance_map.get(s.source_id, False))
            for s in state.get("web_results") or []
        ]
        paper_annotations = [
            _annotated_source(p.model_dump(), relevance_map.get(p.source_id, False))
            for p in state.get("paper_results") or []
        ]

        return {
            "filtered_web_results": filtered_web,
            "filtered_paper_results": filtered_papers,
            "web_results_with_relevance": web_annotations,
            "paper_results_with_relevance": paper_annotations,
            "trace": log_step(
                state,
                "relevance_filter",
                f"{len(filtered_web)} web source(s) and {len(filtered_papers)} paper source(s) deemed relevant",
            ),
        }

    return _run


def route_after_relevance_filter(state: ResearchState) -> str:
    from app.confidence import compute_confidence

    confidence, _ = compute_confidence(state)
    retry_count = state.get("retrieval_retry_count", 0) or 0
    needs_retrieval = bool(state.get("needs_web") or state.get("needs_papers"))
    if confidence < CONFIDENCE_RETRY_THRESHOLD and retry_count < 1 and needs_retrieval:
        return "query_reformulator"
    return "reconciler"
