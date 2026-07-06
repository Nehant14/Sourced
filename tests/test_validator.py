from app.nodes.validator import route_after_validator, validator_node
from app.schemas import ConflictRecord


def test_empty_answer_rejected(llm):
    node = validator_node(llm)
    out = node({"question": "q", "draft_answer": "", "retry_count": 0})
    assert out["validation_passed"] is False
    assert out["retry_count"] == 1


def test_missing_citation_rejected(llm):
    node = validator_node(llm)
    out = node({"question": "q", "draft_answer": "Here is an answer with no citations.", "retry_count": 0})
    assert out["validation_passed"] is False


def test_unmentioned_conflict_forces_retry(llm):
    conflict = ConflictRecord(
        claim_summary="whether attention mechanisms scale efficiently",
        supporting_source_ids=["web_1"],
        disputing_source_ids=["paper_1"],
        explanation="one source predates a later benchmark",
    )
    node = validator_node(llm)
    out = node(
        {
            "question": "q",
            "draft_answer": "Attention mechanisms are efficient. [source: web_1]",
            "conflicts": [conflict],
            "retry_count": 0,
        }
    )
    assert out["validation_passed"] is False
    assert "conflict" in (out["validation_feedback"] or "").lower()
    assert out["retry_count"] == 1


def test_conflict_properly_surfaced_passes(llm):
    conflict = ConflictRecord(
        claim_summary="whether attention mechanisms scale efficiently",
        supporting_source_ids=["web_1"],
        disputing_source_ids=["paper_1"],
        explanation="one source predates a later benchmark",
    )
    node = validator_node(llm)
    out = node(
        {
            "question": "q",
            "draft_answer": (
                "Note: sources disagree on whether attention mechanisms scale efficiently - "
                "web sources say yes [source: web_1], while a paper disputes this [source: paper_1]."
            ),
            "conflicts": [conflict],
            "retry_count": 0,
        }
    )
    assert out["validation_passed"] is True
    assert out["final_answer"] is not None


def test_retries_exhausted_produces_final_answer_anyway(llm):
    node = validator_node(llm)
    out = node({"question": "q", "draft_answer": "", "retry_count": 2})
    assert out["retry_count"] == 2
    assert out["final_answer"] is not None, "Must not loop forever - finalize once retries are exhausted"


def test_route_after_validator():
    assert route_after_validator({"validation_passed": True, "retry_count": 0}) == "end"
    assert route_after_validator({"validation_passed": False, "retry_count": 1}) == "synthesizer"
    assert route_after_validator({"validation_passed": False, "retry_count": 2}) == "end"
