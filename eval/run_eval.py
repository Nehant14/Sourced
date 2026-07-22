"""
Evaluation harness

Usage:
    python -m eval.run_eval                # mock mode, no API keys needed
    python -m eval.run_eval --live         # real providers (needs env vars set)
    python -m eval.run_eval --out report.json

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from eval.questions import EVAL_QUESTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use real providers instead of mocks")
    parser.add_argument("--out", default=None, help="Optional path to write a JSON report")
    args = parser.parse_args()

    if not args.live:
        os.environ["MOCK_MODE"] = "true"

    # Imports deliberately deferred until after MOCK_MODE is set, since
    # build_llm_provider()/build_web_search_provider() read env vars at
    # call time inside app.main - but app.main builds them at import time,
    # so we build our own graph here instead of importing app.main directly.
    from app.graph import build_graph
    from app.providers.arxiv import build_paper_search_provider
    from app.providers.llm import build_llm_provider
    from app.providers.search import build_web_search_provider

    llm = build_llm_provider()
    web_provider = build_web_search_provider()
    paper_provider = build_paper_search_provider()
    graph = build_graph(llm, web_provider, paper_provider)

    results = []
    n_conflict_expected = 0
    n_conflict_expected_and_found = 0
    n_no_conflict_expected = 0
    n_no_conflict_expected_and_clean = 0
    n_retried = 0
    n_citation_present = 0
    n_run = 0
    n_errors = 0

    for eq in EVAL_QUESTIONS:
        if not eq.question.strip():
            # This is the empty-question edge case - it's meant to be
            # rejected at the API layer (see app/main.py), not run through
            # the graph directly. Record it as skipped, not as a graph error.
            results.append(
                {
                    "question": eq.question,
                    "category": eq.category,
                    "skipped_reason": "empty question - validated at API layer, not graph layer",
                }
            )
            continue

        n_run += 1
        start = time.time()
        try:
            out = graph.invoke({"question": eq.question, "retry_count": 0, "trace": []})
            elapsed = time.time() - start
            final = out.get("final_answer")
            answer_text = final.answer if final else out.get("draft_answer", "")
            conflicts_found = len(final.conflicts) if final else len(out.get("conflicts") or [])
            retries_used = final.retries_used if final else out.get("retry_count", 0)
            confidence = final.confidence if final else out.get("confidence")

            has_citation = "[source" in answer_text.lower()
            if has_citation:
                n_citation_present += 1

            if eq.expected_conflict:
                n_conflict_expected += 1
                if conflicts_found > 0:
                    n_conflict_expected_and_found += 1
            else:
                n_no_conflict_expected += 1
                if conflicts_found == 0:
                    n_no_conflict_expected_and_clean += 1

            if retries_used > 0:
                n_retried += 1

            results.append(
                {
                    "question": eq.question,
                    "category": eq.category,
                    "expected_conflict": eq.expected_conflict,
                    "conflicts_found": conflicts_found,
                    "retries_used": retries_used,
                    "confidence": confidence,
                    "has_citation_marker": has_citation,
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            results.append({"question": eq.question, "category": eq.category, "error": str(e)})

    def pct(numerator, denominator):
        return round(100 * numerator / denominator, 1) if denominator else None

    report = {
        "mode": "live" if args.live else "mock",
        "n_questions_run": n_run,
        "n_errors": n_errors,
        "metrics": {
            "conflict_detection_recall_pct": pct(n_conflict_expected_and_found, n_conflict_expected),
            "conflict_detection_true_negative_pct": pct(
                n_no_conflict_expected_and_clean, n_no_conflict_expected
            ),
            "retry_trigger_rate_pct": pct(n_retried, n_run),
            "citation_marker_presence_pct": pct(n_citation_present, n_run),
        },
        "note": (
            "citation_marker_presence_pct is a proxy (does the answer contain "
            "[source: ...] markers at all), NOT the same as citation accuracy "
            "(does each citation actually support the adjacent claim). The "
            "latter requires manual sampling per spec section 6 - do that "
            "separately and record it in the README by hand."
        ),
        "results": results,
    }

    print(json.dumps(report, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nWrote report to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
