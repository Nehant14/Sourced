"""
Curated evaluation set (spec section 6). ~24 questions across categories.

Each entry records:
  - category: for grouping in the report
  - expected_conflict: whether a genuine source disagreement should be found
  - notes: what a human grader should look for

The "deliberately conflicting" questions use language patterns
(disagree/controversy/outdated/etc.) that MockLLMProvider's calibrated rule
also keys off of, so mock-mode runs exercise the same conflict/no-conflict
split a live run would - see app/providers/llm.py::_mock_conflicts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalQuestion:
    question: str
    category: str
    expected_conflict: bool
    notes: str


EVAL_QUESTIONS: list[EvalQuestion] = [
    # --- Simple factual: minimal retrieval, no conflict expected ---
    EvalQuestion(
        "What year was the transformer architecture introduced?",
        "simple_factual",
        False,
        "Should retrieve minimally; single well-established fact, no disagreement expected.",
    ),
    EvalQuestion(
        "What is the boiling point of water at sea level?",
        "simple_factual",
        False,
        "Trivial fact; conflict detector should stay quiet.",
    ),
    EvalQuestion(
        "Who founded OpenAI?",
        "simple_factual",
        False,
        "Well-established fact with likely-consistent sources.",
    ),
    EvalQuestion(
        "What does the acronym RAG stand for in the context of LLMs?",
        "simple_factual",
        False,
        "Definitional; sources should agree.",
    ),
    EvalQuestion(
        "What is the capital of Australia?",
        "simple_factual",
        False,
        "Trivial geography fact; sanity check for false-positive conflicts.",
    ),
    # --- Compound / comparative: should split into sub-questions ---
    EvalQuestion(
        "Compare LangGraph vs plain LangChain chains for building agents.",
        "compound_comparative",
        False,
        "Planner should split into two sub-questions; not expected to surface a real conflict.",
    ),
    EvalQuestion(
        "What's the difference between RAG and fine-tuning for domain adaptation?",
        "compound_comparative",
        False,
        "Two distinct approaches to compare; agreement across sources expected.",
    ),
    EvalQuestion(
        "Compare the training data size of GPT-3 versus GPT-4.",
        "compound_comparative",
        False,
        "Comparative but factual; sources should broadly agree on public figures.",
    ),
    EvalQuestion(
        "How does PostgreSQL compare to MongoDB for a typical web app?",
        "compound_comparative",
        False,
        "Classic comparison question; planner should split it.",
    ),
    # --- Fast-moving ML/research topics with deliberately conflicting sources ---
    EvalQuestion(
        "Is retrieval-augmented generation still necessary now that context "
        "windows are much larger, or has that debate been settled?",
        "deliberate_conflict",
        True,
        "Genuinely contested topic; web opinion pieces and papers plausibly disagree.",
    ),
    EvalQuestion(
        "Do papers and recent commentary disagree on whether chain-of-thought "
        "prompting reliably improves reasoning accuracy?",
        "deliberate_conflict",
        True,
        "Known area of active disagreement/controversy in the literature vs. blog takes.",
    ),
    EvalQuestion(
        "Is there a current controversy over whether synthetic data causes model collapse?",
        "deliberate_conflict",
        True,
        "Explicitly framed as controversial; should trigger conflict detection.",
    ),
    EvalQuestion(
        "Is an older benchmark result on LLM reasoning now outdated compared to newer evals?",
        "deliberate_conflict",
        True,
        "Recency-based disagreement pattern (spec's exact example scenario).",
    ),
    EvalQuestion(
        "Do researchers disagree about whether scaling laws still hold at current model sizes?",
        "deliberate_conflict",
        True,
        "Contested empirical claim likely to surface conflicting source framing.",
    ),
    # --- Edge cases ---
    EvalQuestion(
        "asdkjqwoieqwoi nonsense query zzz111",
        "edge_case",
        False,
        "Nonsense query; system should degrade gracefully, not hallucinate confidently.",
    ),
    EvalQuestion(
        "What does the nopapertest marker paper say about quantum gravity?",
        "edge_case",
        False,
        "Deliberately triggers the mock 'no paper coverage' path (see MockArxivProvider).",
    ),
    EvalQuestion(
        "What does the noresultstest marker say about the weather today?",
        "edge_case",
        False,
        "Deliberately triggers the mock 'zero web results' path (see MockWebSearchProvider).",
    ),
    EvalQuestion(
        "",
        "edge_case",
        False,
        "Empty question; API layer should reject with 400 before reaching the graph.",
    ),
    # --- Additional factual + comparative fill to reach ~20-24 ---
    EvalQuestion(
        "What is the latest research on efficient attention mechanisms?",
        "recent_research",
        False,
        "Should trigger needs_papers=True; agreement expected unless framed as contested.",
    ),
    EvalQuestion(
        "What are the current state of the art approaches to long-context reasoning?",
        "recent_research",
        False,
        "Paper-heavy query; check citation density.",
    ),
    EvalQuestion(
        "What happened in AI news this week?",
        "current_events",
        False,
        "Should trigger needs_web=True strongly; time-sensitive.",
    ),
    EvalQuestion(
        "Compare Python and Rust for building a high-throughput API service.",
        "compound_comparative",
        False,
        "Standard comparison; check sub-question splitting.",
    ),
    EvalQuestion(
        "Who is the current CEO of Anthropic?",
        "simple_factual",
        False,
        "Current-role fact; web search should be favored over papers.",
    ),
    EvalQuestion(
        "Is there ongoing debate about the reliability of LLM-as-judge evaluation methods?",
        "deliberate_conflict",
        True,
        "Framed explicitly as debated; second deliberate-conflict-category anchor question.",
    ),
]
