"""
synthesizer_node (spec section 4.4).

Builds the final answer text using actual retrieved snippets/abstracts as
context (not just titles - this is what makes it RAG). On retry, includes
validation_feedback so the model knows specifically what to fix.
"""
from __future__ import annotations

from app.providers.llm import LLMError, LLMProvider
from app.state import ResearchState
from app.tracing import log_step

SYNTH_SYSTEM_PROMPT = """You are the synthesis stage of a research assistant.
Write a clear, direct answer to the user's question using ONLY the provided
sources as evidence. Rules:
- Every non-trivial factual claim must be followed by an inline citation in
  the form [source: <title or url>].
- If a KNOWN CONFLICTS section is provided below, you MUST explicitly surface
  each conflict in your answer, e.g. "Note: sources disagree on X - source A
  says ..., while source B says ... (likely because ...)."
- If FEEDBACK FROM A PREVIOUS REJECTED DRAFT is provided, you must fix that
  specific issue in this draft.
- Do not fabricate sources or claims beyond what's given."""


def _format_sources(state: ResearchState) -> str:
    lines = []
    for s in state.get("filtered_web_results") or state.get("web_results") or []:
        lines.append(f"- [WEB:{s.source_id}] {s.title} ({s.url})\n  {s.snippet}")
    for p in state.get("filtered_paper_results") or state.get("paper_results") or []:
        lines.append(f"- [PAPER:{p.source_id}] {p.title} ({p.url})\n  {p.abstract}")
    return "\n".join(lines) if lines else "(no sources retrieved)"


def _format_conflicts(state: ResearchState) -> str:
    conflicts = state.get("conflicts") or []
    if not conflicts:
        return ""
    lines = ["KNOWN CONFLICTS (you must surface each of these explicitly):"]
    for c in conflicts:
        lines.append(
            f"- {c.claim_summary} | supporting: {c.supporting_source_ids} "
            f"| disputing: {c.disputing_source_ids} | why: {c.explanation}"
        )
    return "\n".join(lines)


def synthesizer_node(llm: LLMProvider):
    def _run(state: ResearchState) -> dict:
        question = state["question"]
        sources_block = _format_sources(state)
        conflicts_block = _format_conflicts(state)
        feedback = state.get("validation_feedback")

        prompt_parts = [f"Question: {question}", f"\nSources:\n{sources_block}"]
        if conflicts_block:
            prompt_parts.append(f"\n{conflicts_block}")
        if feedback:
            prompt_parts.append(
                f"\nFEEDBACK FROM A PREVIOUS REJECTED DRAFT (fix this): {feedback}"
            )
        prompt = "\n".join(prompt_parts)

        try:
            answer_text = llm.generate_text(prompt, system=SYNTH_SYSTEM_PROMPT)
        except LLMError as e:
            answer_text = (
                "I was unable to generate a synthesized answer due to a model error "
                f"({e}). Please retry the request."
            )

        return {
            "draft_answer": answer_text,
            "trace": log_step(
                state,
                "synthesizer",
                f"draft generated (retry_count={state.get('retry_count', 0)}, "
                f"had_feedback={bool(feedback)})",
            ),
        }

    return _run
