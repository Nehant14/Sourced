# Multi-Perspective Research Assistant

A research assistant that answers a question by retrieving from **two
independent corpora** — general web search (Tavily) and academic papers
(arXiv) — using a **LangGraph** state machine, then explicitly reconciles
disagreements between what the sources say before synthesizing an answer. A
validator checks the synthesized answer for citation coverage and conflict
coverage, and can send it back for a rewrite (up to 2 times) before
finalizing.

---

## 1. What it does

Ask it a question. A planner decides whether it needs web search, paper
search, or both, and whether the question should be split into
sub-questions. The relevant retrievers run **in parallel**. A reconciler
extracts factual claims from every retrieved source and checks whether any
of them genuinely disagree — calibrated to stay quiet when sources agree,
not to manufacture drama. A synthesizer writes a cited answer that surfaces
any real conflict explicitly. A validator checks citation coverage and
conflict coverage, routing back to the synthesizer with specific feedback
when needed. Every answer ships with a derived confidence score and a full
execution trace.

## 2. Architecture

```
planner
  │  (conditional fan-out on needs_web / needs_papers)
  ├──> web_retriever ───┐
  └──> paper_retriever ─┼──> reconciler ──> synthesizer ──> validator
                         │                                     │
                         │           ┌─────── retry (≤2) ──────┤
                         │           ▼                         │
                         └────── synthesizer <──────────────────
                                                                 │
                                                          pass / retries exhausted
                                                                 ▼
                                                                END
```

- `web_retriever` and `paper_retriever` only run when the planner flags them
  as needed (a conditional edge, not an `if` inside an always-run node), and
  run **in parallel** when both are needed — LangGraph fans out independent
  branches within a superstep and joins them before `reconciler` runs.
- `validator` routes back to `synthesizer` (carrying specific feedback) or to
  `END`, capped at 2 retries.

State schema, node responsibilities, and design-decision rationale are
documented as docstrings directly in `app/state.py`,
`app/nodes/planner.py`, and `app/nodes/reconciler.py` — that's the
authoritative source, this README summarizes it.

## 3. Evaluation results (mock mode)

These numbers came from a real run of `python -m eval.run_eval` against the
24-question curated set in `eval/questions.py`, using `MockLLMProvider` (no
API keys, no network required). This confirms the harness and control flow
work end-to-end.

| Metric | Result (mock mode) |
|---|---|
| Questions run | 23 of 24 |
| Errors | 0 |
| Conflict detection recall (on `deliberate_conflict` questions) | 100% |
| Conflict detection true-negative rate (on should-agree questions) | 100% |
| Citation-marker presence rate | 100% |

## 4. Example: catching a genuine disagreement

Question (`deliberate_conflict` category): *"Is there ongoing debate about
the reliability of LLM-as-judge evaluation methods?"*

Result: 1 conflict found, confidence 0.85, citation marker present, 0
retries needed — the reconciler → synthesizer → validator path correctly
threads a flagged conflict through to the final answer.

## 5. Test status

```
pytest -v
30 passed, 4 failed
```

## 6. Definition of done (from the build spec)

| Question | Status |
|---|---|
| "Why does the planner branch differently for these two example queries?" | ✅ Demonstrated in `tests/test_planner.py::test_three_question_types_produce_different_plans`, runnable with zero keys. |
| "Show me a case where the sources disagreed and the system caught it" | ✅ Demonstrated in mock mode (section 4) and `tests/test_graph_e2e.py::test_e2e_deliberate_conflict_question_surfaces_conflict`. |
| "What happens if arXiv is down right now?" | ✅ Degrades to web-only (or a clearly-marked zero-source state if both fail) — the system never crashes. |

## 8. Local setup

```bash
git clone <this repo>
cd research_assistant
pip install -r requirements.txt
cp .env.example .env    # fill in ANTHROPIC_API_KEY and TAVILY_API_KEY, or...
```

**Zero-key mode** (mock providers everywhere, fully offline):

```bash
MOCK_MODE=true uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "Is there a controversy over whether RAG is still necessary given long context windows?"}'
```

**Real providers (Gemini ):**

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=AIza-...
export GEMINI_MODEL=gemini-1.5-flash
uvicorn app.main:app --reload
```

**Real providers (Anthropic):**

```bash
export LLM_PROVIDER=anthropic WEB_SEARCH_PROVIDER=tavily PAPER_SEARCH_PROVIDER=arxiv
export ANTHROPIC_API_KEY=sk-ant-...
export TAVILY_API_KEY=tvly-...
uvicorn app.main:app --reload
```

Run tests (works in mock mode, zero keys needed):

```bash
pytest -v
```

Run the eval harness:

```bash
python -m eval.run_eval               # mock mode
python -m eval.run_eval --live        # real providers, needs keys set
```


### LangSmith tracing

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_PROJECT=multi-perspective-research-assistant
```

With these set, every `graph.invoke(...)` call is automatically traced in
LangSmith with no code changes.

---

## Repo layout

```
app/
  schemas.py          # pydantic models = the structured-output schemas
  state.py             # ResearchState TypedDict (spec section 3)
  confidence.py        # derived confidence score (spec section 7)
  tracing.py           # lightweight execution log, independent of LangSmith
  graph.py             # StateGraph wiring (spec section 5)
  main.py              # FastAPI app, POST /research
  providers/
    llm.py             # AnthropicProvider + GeminiProvider + MockLLMProvider
    search.py          # TavilyProvider + MockWebSearchProvider
    arxiv.py            # ArxivProvider (stdlib XML parsing) + MockArxivProvider
  nodes/
    planner.py
    web_retriever.py
    paper_retriever.py
    reconciler.py
    synthesizer.py
    validator.py
eval/
  questions.py          # 24-question curated eval set
  run_eval.py            # metrics harness
tests/
  test_planner.py
  test_retrievers.py
  test_reconciler.py
  test_validator.py
  test_graph_e2e.py
  test_llm_providers.py
  test_robustness.py
requirements.txt
.env.example
render.yaml
Procfile
```
