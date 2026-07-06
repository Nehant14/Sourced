"""
validator_node (spec section 4.5).

Checks, in order: non-empty answer, citation presence, and - the retry
condition actually tied to something real - that every known conflict is
surfaced in the answer text. Computes confidence here rather than hardcoding
it. Routes back to the synthesizer (up to 2 retries) or to END.
"""
from __future__ import annotations

from app.confidence import compute_confidence
from app.providers.llm import LLMError, LLMProvider
from app.schemas import ResearchAnswer, ValidationResult
from app.state import ResearchState
from app.tracing import log_step

MAX_RETRIES = 2

VALIDATE_SYSTEM_PROMPT = """You are a strict validator for a research
assistant's draft answer. Check:
1. The answer is non-empty and actually addresses the question.
2. Every non-trivial factual claim has an inline citation like [source: ...].
3. If a list of known conflicts is provided, the answer text must explicitly
   surface EACH of them (not just cite sources, but actually say the sources
   disagree and roughly why).
Return passed=false with specific, actionable feedback if any check fails.
List any conflict summaries that are missing from the answer."""


def _conflict_mentioned(answer: str, claim_summary: str) -> bool:
    """Lightweight heuristic: require at least one disagreement-signaling
    word AND some lexical overlap with the conflict's own summary text.
    (The real Anthropic-backed path can additionally use an LLM check via
    generate_structured(ValidationResult) - see validator_node below - this
    heuristic is what MockLLMProvider relies on and is also used as a fast
    pre-filter before the LLM check in the real path.)"""
    lower = answer.lower()
    disagreement_words = ["disagree", "conflict", "differ", "however", "in contrast", "note:"]
    has_signal = any(w in lower for w in disagreement_words)
    summary_words = [w for w in claim_summary.lower().split() if len(w) > 4]
    overlap = any(w in lower for w in summary_words) if summary_words else True
    return has_signal and overlap


def validator_node(llm: LLMProvider):
    def _run(state: ResearchState) -> dict:
        answer = state.get("draft_answer", "") or ""
        conflicts = state.get("conflicts") or []
        retry_count = state.get("retry_count", 0) or 0

        # Fast heuristic checks first (spec allows "a lightweight heuristic
        # or a second LLM call" - we do both: heuristic short-circuits
        # obvious failures without spending a call, LLM call catches subtler
        # ones).
        if not answer.strip():
            result = ValidationResult(passed=False, feedback="Draft answer is empty.")
        elif "[source" not in answer.lower():
            result = ValidationResult(
                passed=False, feedback="No inline citations found. Add [source: title/url] markers."
            )
        else:
            missing = [
                c.claim_summary for c in conflicts if not _conflict_mentioned(answer, c.claim_summary)
            ]
            if missing:
                result = ValidationResult(
                    passed=False,
                    feedback=f"Answer does not surface {len(missing)} known conflict(s): {missing}",
                    missing_conflict_summaries=missing,
                )
            else:
                # Heuristics passed; do the deeper LLM check for anything
                # the heuristic can't catch (e.g. citation-claim mismatch).
                try:
                    result = llm.generate_structured(
                        prompt=(
                            f"Question: {state['question']}\n\nDraft answer:\n{answer}\n\n"
                            f"Known conflicts: {[c.claim_summary for c in conflicts]}\n\nValidate."
                        ),
                        schema=ValidationResult,
                        system=VALIDATE_SYSTEM_PROMPT,
                        context={
                            "draft_answer": answer,
                            "conflicts": [c.model_dump() for c in conflicts],
                        },
                    )
                except LLMError:
                    # If the checker itself fails, don't block the pipeline on
                    # a broken validator - fall back to "heuristics passed"
                    # rather than an infinite-feeling retry loop.
                    result = ValidationResult(passed=True)

        should_retry = (not result.passed) and retry_count < MAX_RETRIES
        new_retry_count = retry_count + 1 if should_retry else retry_count

        confidence, confidence_reasons = compute_confidence(
            {**state, "retry_count": new_retry_count}  # type: ignore[arg-type]
        )

        update: dict = {
            "validation_passed": result.passed,
            "validation_feedback": result.feedback if not result.passed else None,
            "retry_count": new_retry_count,
            "confidence": confidence,
            "trace": log_step(
                state,
                "validator",
                f"passed={result.passed}, retry_count={new_retry_count}, "
                f"confidence={confidence} ({'; '.join(confidence_reasons)})",
            ),
        }

        if result.passed or new_retry_count >= MAX_RETRIES:
            update["final_answer"] = ResearchAnswer(
                question=state["question"],
                answer=answer,
                citations=_extract_citation_strings(answer),
                conflicts=conflicts,
                confidence=confidence,
                retries_used=new_retry_count,
                sources_used={
                    "web": len(state.get("web_results") or []),
                    "papers": len(state.get("paper_results") or []),
                },
                degraded=bool(state.get("degraded")),
                degradation_notes=state.get("degradation_notes") or [],
            )

        return update

    return _run


def _extract_citation_strings(answer: str) -> list[str]:
    """Pull out the literal contents of every [source: ...] marker. This is
    a display convenience, not something downstream logic depends on."""
    import re

    return re.findall(r"\[source:\s*([^\]]+)\]", answer, flags=re.IGNORECASE)


def route_after_validator(state: ResearchState) -> str:
    if state.get("validation_passed") or (state.get("retry_count", 0) >= MAX_RETRIES):
        return "end"
    return "synthesizer"
