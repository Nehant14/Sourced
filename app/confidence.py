"""
Confidence scoring (spec section 7).

Deliberately simple and fully explainable: start at 1.0, subtract a fixed
penalty for each factor that should reduce trust in the answer. The goal is
that for any given answer you can point at exactly which penalties fired and
why - not squeeze out a precise-looking number.
"""
from __future__ import annotations

from app.state import ResearchState

BASE_CONFIDENCE = 1.0

PENALTY_PER_UNRESOLVED_CONFLICT = 0.15
PENALTY_ZERO_WEB_RESULTS_WHEN_NEEDED = 0.20
PENALTY_ZERO_PAPER_RESULTS_WHEN_NEEDED = 0.20
PENALTY_PER_RETRY = 0.10
PENALTY_CONFLICT_DETECTION_UNAVAILABLE = 0.10
PENALTY_PROVIDER_DEGRADED = 0.10

MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 1.0


def compute_confidence(state: ResearchState) -> tuple[float, list[str]]:
    """Returns (score, explanation_lines) so callers/README/eval can show
    exactly why a given answer scored what it did."""
    score = BASE_CONFIDENCE
    reasons: list[str] = []

    conflicts = state.get("conflicts") or []
    if conflicts:
        penalty = PENALTY_PER_UNRESOLVED_CONFLICT * len(conflicts)
        score -= penalty
        reasons.append(f"-{penalty:.2f}: {len(conflicts)} unresolved source conflict(s)")

    if state.get("needs_web") and not state.get("web_results"):
        score -= PENALTY_ZERO_WEB_RESULTS_WHEN_NEEDED
        reasons.append(f"-{PENALTY_ZERO_WEB_RESULTS_WHEN_NEEDED:.2f}: web search returned no results")

    if state.get("needs_papers") and not state.get("paper_results"):
        score -= PENALTY_ZERO_PAPER_RESULTS_WHEN_NEEDED
        reasons.append(f"-{PENALTY_ZERO_PAPER_RESULTS_WHEN_NEEDED:.2f}: paper search returned no results")

    retry_count = state.get("retry_count") or 0
    if retry_count:
        penalty = PENALTY_PER_RETRY * retry_count
        score -= penalty
        reasons.append(f"-{penalty:.2f}: validator required {retry_count} retry/retries")

    if state.get("conflict_detection_unavailable"):
        score -= PENALTY_CONFLICT_DETECTION_UNAVAILABLE
        reasons.append(f"-{PENALTY_CONFLICT_DETECTION_UNAVAILABLE:.2f}: conflict detection was unavailable")

    if state.get("degraded"):
        score -= PENALTY_PROVIDER_DEGRADED
        reasons.append(f"-{PENALTY_PROVIDER_DEGRADED:.2f}: one or more providers degraded/failed")

    score = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, score))
    if not reasons:
        reasons.append("no penalties applied - full base confidence")
    return round(score, 3), reasons
