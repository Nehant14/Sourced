"""
The LangGraph state. Every node takes this (or a subset) in and returns a
partial dict of the keys it updates - LangGraph merges those into the running
state. We deliberately keep this a plain TypedDict (not BaseModel) because
that's what langgraph's StateGraph is built around; validation of the
individual objects inside it happens via the pydantic models in schemas.py
at the point they're constructed.

Design rule (see build spec section 3): every field a downstream node needs
must live here. Nodes should not reach for hidden globals or extra args.
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from app.schemas import (
    ConflictRecord,
    ExtractedClaim,
    PaperReference,
    ResearchAnswer,
    SourceReference,
)


class ResearchState(TypedDict, total=False):
    # --- input ---
    question: str

    # --- planner output ---
    sub_questions: list[str]
    needs_web: bool
    needs_papers: bool
    plan_reasoning: str

    # --- retrieval output ---
    web_results: list[SourceReference]
    paper_results: list[PaperReference]

    # --- degradation bookkeeping (section 8) ---
    web_error: Optional[str]
    paper_error: Optional[str]
    degraded: bool
    degradation_notes: list[str]

    # --- reconciler output ---
    claims: list[ExtractedClaim]
    conflicts: list[ConflictRecord]
    conflict_detection_unavailable: bool

    # --- synthesizer / validator loop ---
    draft_answer: str
    confidence: float
    validation_passed: bool
    validation_feedback: Optional[str]
    retry_count: int

    # --- trace (lightweight, LangSmith-independent execution log) ---
    trace: Annotated[list[dict], operator.add]

    # --- final ---
    final_answer: ResearchAnswer
