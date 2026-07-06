"""
FastAPI app (spec section 10). Single endpoint: POST /research.

Run locally:
    MOCK_MODE=true uvicorn app.main:app --reload

Run against real providers:
    LLM_PROVIDER=anthropic WEB_SEARCH_PROVIDER=tavily PAPER_SEARCH_PROVIDER=arxiv \\
    ANTHROPIC_API_KEY=... TAVILY_API_KEY=... \\
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph import build_graph
from app.providers.arxiv import build_paper_search_provider
from app.providers.llm import build_llm_provider
from app.providers.search import build_web_search_provider
from app.schemas import ResearchAnswer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research_assistant")

app = FastAPI(title="Multi-Perspective Research Assistant")

# Providers and the compiled graph are built once at import/startup time,
# not per-request - that's the whole point of dependency injection into the
# node closures in app/graph.py.
_llm = build_llm_provider()
_web_provider = build_web_search_provider()
_paper_provider = build_paper_search_provider()
_graph = build_graph(_llm, _web_provider, _paper_provider)


class ResearchRequest(BaseModel):
    question: str


class ResearchResponse(BaseModel):
    final_answer: ResearchAnswer
    trace: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    initial_state = {"question": req.question, "retry_count": 0, "trace": []}
    try:
        result = _graph.invoke(initial_state)
    except Exception as e:  # noqa: BLE001
        logger.exception("Graph execution failed")
        raise HTTPException(status_code=500, detail=f"research pipeline failed: {e}") from e

    final_answer = result.get("final_answer")
    if final_answer is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed without producing a final_answer - this is a bug, not "
            "an expected degradation path.",
        )

    return ResearchResponse(final_answer=final_answer, trace=result.get("trace", []))
