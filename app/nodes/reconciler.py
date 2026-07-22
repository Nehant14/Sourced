"""
reconciler_node - the core differentiator.

Implementation note on the "clustering" step: the spec describes three
LLM-driven steps (extract claims / cluster claims / determine agreement per
cluster). We implement extraction as its own call, then merge clustering +
conflict-determination into a single second call (ConflictAnalysis), because
for the source counts this system deals with (a handful of web results +
a handful of papers) doing those as separate calls adds latency and cost
without adding accuracy - the LLM can cluster-and-judge in one pass just as
well. This is a documented simplification, not an accidental shortcut; a
version handling dozens of sources per query would likely want to split them
back out so clustering can be validated independently. See README limitations.

Calibration: the whole point of this node is to NOT manufacture conflicts.
The prompt explicitly instructs the model to report an empty conflicts list
when sources agree, and the mock provider's deterministic rule mirrors that
(see MockLLMProvider._mock_conflicts) so this behavior is testable offline.
"""
from __future__ import annotations

from app.providers.llm import LLMError, LLMProvider
from app.schemas import ConflictAnalysis, ExtractedClaims
from app.state import ResearchState
from app.tracing import log_step

EXTRACT_SYSTEM_PROMPT = """You extract discrete, checkable factual claims from
source snippets/abstracts. Each claim should be one specific assertion, tagged
with the source_id it came from. Skip filler/opinion sentences; extract only
things that could in principle be verified or contradicted by another source.
Extract at most 5 claims per source."""

ANALYZE_SYSTEM_PROMPT = """You are given factual claims extracted from multiple
sources (web and academic). Group claims that are about the same underlying
fact. For each group that has claims from more than one distinct source:
- If the sources AGREE, do not report it as a conflict.
- If the sources DISAGREE (contradict, or one is clearly outdated relative
  to the other, or they report materially different numbers/conclusions for
  the same question), report it as a conflict with a specific explanation
  of what differs and, if apparent, why (recency, methodology, scope, etc).
Be conservative: only report genuine disagreements. Do NOT report a conflict
merely because two sources phrase the same fact differently, emphasize
different aspects, or one is more detailed than the other. If in doubt,
do not report a conflict. It is expected and normal for the conflicts list
to be empty when sources agree."""


def _sources_to_context(state: ResearchState) -> list[dict]:
    web = state.get("web_results_filtered")
    if web is None:
        web = state.get("web_results") or []
    papers = state.get("paper_results_filtered")
    if papers is None:
        papers = state.get("paper_results") or []

    sources = []
    for s in web:
        sources.append({"source_id": s.source_id, "title": s.title, "snippet": s.snippet})
    for p in papers:
        sources.append({"source_id": p.source_id, "title": p.title, "abstract": p.abstract})
    return sources


def reconciler_node(llm: LLMProvider):
    def _run(state: ResearchState) -> dict:
        sources = _sources_to_context(state)
        if not sources:
            return {
                "claims": [],
                "conflicts": [],
                "trace": log_step(state, "reconciler", "no sources available, skipping claim extraction"),
            }

        source_block = "\n\n".join(
            f"[{s['source_id']}] {s.get('title','')}\n{s.get('snippet') or s.get('abstract','')}"
            for s in sources
        )

        # Step 1: extract claims
        try:
            extracted: ExtractedClaims = llm.generate_structured(
                prompt=f"Sources:\n\n{source_block}\n\nExtract claims.",
                schema=ExtractedClaims,
                system=EXTRACT_SYSTEM_PROMPT,
                context={"sources": sources, "question": state["question"]},
            )
        except LLMError:
            return {
                "claims": [],
                "conflicts": [],
                "conflict_detection_unavailable": True,
                "trace": log_step(state, "reconciler", "claim extraction failed, conflict detection unavailable"),
            }

        if not extracted.claims:
            return {
                "claims": [],
                "conflicts": [],
                "trace": log_step(state, "reconciler", "no claims extracted from sources"),
            }

        # Step 2 + 3: cluster + determine agreement/conflict
        claims_block = "\n".join(f"[{c.claim_id} / {c.source_id}] {c.claim_text}" for c in extracted.claims)
        try:
            analysis: ConflictAnalysis = llm.generate_structured(
                prompt=f"Claims:\n\n{claims_block}\n\nAnalyze for genuine conflicts.",
                schema=ConflictAnalysis,
                system=ANALYZE_SYSTEM_PROMPT,
                context={
                    "claims": [c.model_dump() for c in extracted.claims],
                    "question": state["question"],
                },
            )
        except LLMError:
            return {
                "claims": extracted.claims,
                "conflicts": [],
                "conflict_detection_unavailable": True,
                "trace": log_step(state, "reconciler", "conflict analysis failed, conflict detection unavailable"),
            }

        return {
            "claims": extracted.claims,
            "conflicts": analysis.conflicts,
            "trace": log_step(
                state,
                "reconciler",
                f"{len(extracted.claims)} claim(s) extracted, {len(analysis.conflicts)} conflict(s) found",
            ),
        }

    return _run