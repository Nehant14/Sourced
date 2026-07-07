"""
Pydantic models for every structured object that flows through the graph.

These are used two ways:
  1. As the state's field types (documentation + runtime validation when we
     choose to validate).
  2. As the `schema` argument to LLMProvider.generate_structured(...) - i.e.
     these ARE the function-calling / structured-output schemas. We never
     regex-parse free text out of the LLM for anything machine-readable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------

class SourceReference(BaseModel):
    """A single web search result."""
    source_id: str = Field(description="Stable short id, e.g. 'web_1'")
    title: str
    url: str
    snippet: str
    retrieved_at: datetime


class PaperReference(BaseModel):
    """A single arXiv paper result."""
    source_id: str = Field(description="Stable short id, e.g. 'paper_1'")
    title: str
    url: str
    arxiv_id: str
    authors: list[str]
    published: Optional[str] = None
    abstract: str


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class PlannerDecision(BaseModel):
    """Structured output the planner LLM call must return."""
    sub_questions: list[str] = Field(
        default_factory=list,
        description="If the question is compound, break it into independent "
        "sub-questions. Empty list if the question is already atomic.",
    )
    needs_web: bool = Field(description="Whether general web search is needed.")
    needs_papers: bool = Field(description="Whether arXiv paper search is needed.")
    reasoning: str = Field(
        description="One or two sentences on why this plan was chosen."
    )


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class ExtractedClaim(BaseModel):
    claim_id: str
    claim_text: str
    source_id: str = Field(description="id of the SourceReference/PaperReference this came from")


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim]


class ConflictRecord(BaseModel):
    claim_summary: str = Field(description="The underlying fact/question the sources disagree about")
    supporting_source_ids: list[str]
    disputing_source_ids: list[str]
    explanation: str = Field(
        description="Why the sources differ, e.g. recency, methodology, scope"
    )


class ConflictAnalysis(BaseModel):
    """Structured output of the claim-clustering + conflict-detection LLM call."""
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    agreement_summary: str = Field(
        default="",
        description="Brief note on what sources agreed on, even if no conflicts found",
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    passed: bool
    feedback: Optional[str] = Field(
        default=None, description="Specific, actionable reason for rejection"
    )
    missing_conflict_summaries: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------

class ResearchAnswer(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(
        default_factory=list, description="URLs / source titles actually cited in the answer"
    )
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    confidence: float
    retries_used: int
    sources_used: dict[str, int] = Field(
        default_factory=dict, description="e.g. {'web': 3, 'papers': 2}"
    )
    websites: list[SourceReference] = Field(
        default_factory=list, description="Web Sources"
    )
    research_papers: list[PaperReference] = Field(
        default_factory=list, description="Research Papers"
    )
    degraded: bool = Field(
        default=False, description="True if one or more providers failed and we continued anyway"
    )
    degradation_notes: list[str] = Field(default_factory=list)