"""
relevance_filter_node.

Runs right after web_retriever/paper_retriever, before reconciler. Sends
every retrieved source (title + snippet/abstract) to the LLM in a single
batched call and asks which ones actually address the question - not just
whether they share keywords or a general topic, but whether they'd help
answer it. This exists because retrieval "succeeding" (non-zero results)
says nothing about whether those results are useful; a synthesizer forced
to use several off-topic sources will either produce a strained answer or
correctly say the sources don't address the question - but confidence
should reflect that a zero-*relevant*-results case is not the same as a
zero-results-*returned* case.

Design choice: ONE call covering both web + paper sources together, not one
per source and not two separate calls - mirrors the same cost/latency-vs-
accuracy tradeoff already made in reconciler.py; splitting relevance checks
per source doesn't meaningfully improve accuracy for the handful of sources
this system retrieves per query, and scales badly with source count.

Filtered results go into web_results_filtered/paper_results_filtered -
web_results/paper_results themselves are NOT replaced with a filtered
subset, only annotated with a `relevant` flag on each item, so the full
retrieved set (including what got filtered out) stays visible in the final
answer's websites/research_papers fields. Downstream nodes (reconciler,
synthesizer) read the _filtered lists.
"""
from __future__ import annotations

from app.confidence import compute_confidence
from app.providers.llm import LLMError, LLMProvider
from app.schemas import RelevanceAnalysis
from app.state import ResearchState
from app.tracing import log_step

RELEVANCE_SYSTEM_PROMPT = """You are given a research question and a list of
retrieved sources (web results and/or academic paper abstracts). For EACH
source, decide whether it actually helps answer the question - not just
whether it shares keywords or a general topic, but whether a reader could
use it as real evidence for an answer. Mark relevant=false for sources that
are off-topic, tangential, or only superficially related (e.g. sharing one
keyword with the question but addressing a completely different subject).
Give a short, specific `reason` for every verdict, especially irrelevant
ones."""

RELEVANCE_RETRY_CONFIDENCE_THRESHOLD = 0.65
MAX_RETRIEVAL_RETRIES = 1


def _to_context_list(state: ResearchState) -> list[dict]:
    out = []
    for s in state.get("web_results") or []:
        out.append({"source_id": s.source_id, "title": s.title, "text": s.snippet})
    for p in state.get("paper_results") or []:
        out.append({"source_id": p.source_id, "title": p.title, "text": p.abstract})
    return out


def relevance_filter_node(llm: LLMProvider):
    def _run(state: ResearchState) -> dict:
        web_results = state.get("web_results") or []
        paper_results = state.get("paper_results") or []
        sources = _to_context_list(state)

        if not sources:
            return {
                "web_results_filtered": [],
                "paper_results_filtered": [],
                "relevance_discarded_count": 0,
                "relevance_total_count": 0,
                "trace": log_step(state, "relevance_filter", "no sources to filter"),
            }

        source_block = "\n".join(f"[{s['source_id']}] {s['title']}\n{s['text']}" for s in sources)

        try:
            analysis: RelevanceAnalysis = llm.generate_structured(
                prompt=(
                    f"Question: {state['question']}\n\nSources:\n{source_block}\n\n"
                    "For each source_id above, decide relevant true/false."
                ),
                schema=RelevanceAnalysis,
                system=RELEVANCE_SYSTEM_PROMPT,
                context={"sources": sources, "question": state["question"]},
            )
        except LLMError as e:
            notes = list(state.get("degradation_notes") or [])
            notes.append(f"Relevance filter unavailable ({e}); proceeding with all retrieved sources.")
            return {
                "web_results_filtered": web_results,
                "paper_results_filtered": paper_results,
                "relevance_discarded_count": 0,
                "relevance_total_count": len(sources),
                "degradation_notes": notes,
                "trace": log_step(state, "relevance_filter", f"unavailable, proceeding unfiltered: {e}"),
            }

        verdict_by_id = {v.source_id: v for v in analysis.verdicts}

        def _apply(items):
            annotated = []
            for item in items:
                v = verdict_by_id.get(item.source_id)
                relevant = v.relevant if v is not None else True
                annotated.append(item.model_copy(update={"relevant": relevant}))
            return annotated

        annotated_web = _apply(web_results)
        annotated_papers = _apply(paper_results)

        filtered_web = [s for s in annotated_web if s.relevant is not False]
        filtered_papers = [p for p in annotated_papers if p.relevant is not False]

        total = len(annotated_web) + len(annotated_papers)
        discarded = total - len(filtered_web) - len(filtered_papers)

        notes = list(state.get("degradation_notes") or [])
        if discarded:
            notes.append(
                f"{discarded} of {total} retrieved source(s) filtered as not relevant to the "
                f"question ({len(annotated_web) - len(filtered_web)} web, "
                f"{len(annotated_papers) - len(filtered_papers)} papers)."
            )

        return {
            "web_results": annotated_web,
            "paper_results": annotated_papers,
            "web_results_filtered": filtered_web,
            "paper_results_filtered": filtered_papers,
            "relevance_discarded_count": discarded,
            "relevance_total_count": total,
            "degradation_notes": notes,
            "trace": log_step(state, "relevance_filter", f"{discarded}/{total} source(s) filtered as irrelevant"),
        }

    return _run


def route_after_relevance_filter(state: ResearchState) -> str:
    """Conditional edge: if confidence is already too low because too much
    of what was retrieved got filtered out, and we haven't used the
    retrieval retry budget yet, go reformulate the query and try again.
    Otherwise proceed to reconciler as normal."""
    retrieval_retry_count = state.get("retrieval_retry_count", 0) or 0
    if retrieval_retry_count >= MAX_RETRIEVAL_RETRIES:
        return "reconciler"

    provisional_confidence, _ = compute_confidence(state)
    if provisional_confidence < RELEVANCE_RETRY_CONFIDENCE_THRESHOLD:
        return "query_reformulator"
    return "reconciler"