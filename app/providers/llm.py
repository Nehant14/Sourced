"""
LLM provider abstraction.

Two concrete providers are implemented:
  - AnthropicProvider: real calls to the Anthropic Messages API. Structured
    output is implemented via forced tool-use (a single tool whose input
    schema is the pydantic model's JSON schema, tool_choice forced to it) -
    this is the "function-calling / structured output mode" the spec asks
    for; we never regex-parse free text for anything machine-readable.
  - GeminiProvider: uses Google's Gemini API via google-generativeai for both
    plain text and JSON-structured responses.
  - MockLLMProvider: fully offline, deterministic. It does NOT call any
    network. It inspects the requested schema + an optional `context` dict
    (raw data the node already has, e.g. the retrieved sources) and fabricates
    a plausible structured response. This is what lets the whole graph, the
    test suite and the eval harness run with zero API keys.

Swapping in OpenAI/Gemini only requires implementing this same interface;
the rest of the codebase never touches provider-specific types.
"""
from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when an LLM call fails after retries. Nodes should catch this
    and degrade gracefully rather than let it propagate as a crash."""


class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, system: str | None = None) -> str:
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str | None = None,
        context: dict | None = None,
    ) -> T:
        """Return an instance of `schema` populated by the LLM. `context` is
        optional extra raw data available to MockLLMProvider so it can
        fabricate a sensible answer without needing to parse `prompt`; real
        providers ignore it."""
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "pip install anthropic"
            ) from e
        self._anthropic = anthropic
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set and no api_key passed in")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_text(self, prompt: str, system: str | None = None) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str | None = None,
        context: dict | None = None,
    ) -> T:
        tool_name = f"emit_{schema.__name__.lower()}"
        tool = {
            "name": tool_name,
            "description": f"Emit a {schema.__name__} object as the final answer to this request.",
            "input_schema": schema.model_json_schema(),
        }
        last_error: Exception | None = None
        for attempt in range(2):  # one retry with a stricter instruction, per spec section 8
            try:
                strictness = (
                    ""
                    if attempt == 0
                    else "\n\nIMPORTANT: your previous response was not valid. "
                    "You MUST call the tool exactly once with fields matching the schema exactly."
                )
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system=system or "",
                    messages=[{"role": "user", "content": prompt + strictness}],
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                )
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                        return schema(**block.input)
                raise LLMError("Model did not return the forced tool call")
            except Exception as e:  # noqa: BLE001 - genuinely want to catch+retry broadly here
                last_error = e
                continue
        raise LLMError(f"Structured generation failed after retries: {last_error}")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "The 'google-generativeai' package is required for GeminiProvider. "
                "pip install google-generativeai"
            ) from e
        self._genai = genai
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set and no api_key passed in")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self._genai.configure(api_key=self.api_key)

    def _build_model(self, system: str | None = None):
        return self._genai.GenerativeModel(model_name=self.model, system_instruction=system)

    def generate_text(self, prompt: str, system: str | None = None) -> str:
        model = self._build_model(system)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "").strip()

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str | None = None,
        context: dict | None = None,
    ) -> T:
        model = self._build_model(system)
        schema_json = json.dumps(schema.model_json_schema())
        payload_prompt = (
            f"{prompt}\n\nReturn ONLY valid JSON that matches this schema:\n{schema_json}"
        )
        try:
            resp = model.generate_content(
                payload_prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(getattr(resp, "text", "").strip())
        except json.JSONDecodeError as e:
            raise LLMError(f"Gemini structured response was not valid JSON: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Gemini structured generation failed: {e}") from e
        return schema(**data)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """Deterministic, offline. Same input -> same output, which is what we
    want for reproducible tests and for the eval harness to be meaningful in
    mock mode."""

    def generate_text(self, prompt: str, system: str | None = None) -> str:
        import re

        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]

        # Pull a real-looking citation label out of the prompt's sources
        # block (synthesizer_node formats sources as "- [WEB:id] Title (url)"
        # / "- [PAPER:id] Title (url)") so the mock answer cites something
        # that actually appeared in context, not a fixed placeholder.
        titles = re.findall(r"\[(?:WEB|PAPER):[^\]]+\]\s*([^(]+)\(", prompt)
        citation = titles[0].strip() if titles else "mock-source-1"

        conflict_note = ""
        if "KNOWN CONFLICTS" in prompt:
            conflict_note = (
                " Note: sources disagree on this point - one source supports one view while "
                "another disputes it, likely due to differing recency or scope."
            )

        return (
            f"[mock-answer:{h}] Based on the retrieved sources, here is a synthesized "
            f"answer to the question.{conflict_note} [source: {citation}]"
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str | None = None,
        context: dict | None = None,
    ) -> T:
        context = context or {}
        name = schema.__name__

        if name == "PlannerDecision":
            return self._mock_planner(context)
        if name == "ExtractedClaims":
            return self._mock_claims(context)
        if name == "ConflictAnalysis":
            return self._mock_conflicts(context)
        if name == "ValidationResult":
            return self._mock_validation(context)

        # Generic fallback: try to construct with minimal/empty values.
        try:
            return schema()
        except Exception as e:  # noqa: BLE001
            raise LLMError(
                f"MockLLMProvider has no canned response for schema {name} "
                f"and it has required fields: {e}"
            )

    # -- individual mock strategies -----------------------------------

    def _mock_planner(self, context: dict):
        from app.schemas import PlannerDecision

        question = (context.get("question") or "").lower()
        is_compound = any(
            kw in question for kw in ["compare", " vs ", "versus", "difference between", "and also"]
        )
        wants_recent_research = any(
            kw in question
            for kw in ["latest research", "recent paper", "state of the art", "sota", "recent advances"]
        )
        wants_current_events = any(
            kw in question for kw in ["today", "this week", "current", "latest news", "right now"]
        )

        needs_papers = wants_recent_research or is_compound
        needs_web = wants_current_events or is_compound or not wants_recent_research

        # Enforcement: always require at least one retrieval path for
        # citation integrity (see app/nodes/planner.py docstring for the
        # rationale). Mock mirrors the same rule as the real provider path.
        if not needs_web and not needs_papers:
            needs_web = True

        sub_questions = []
        if is_compound and " and also" not in question:
            # crude split around comparison keywords, good enough for mock mode
            for sep in [" vs ", " versus ", "compare "]:
                if sep in question:
                    parts = question.split(sep)
                    if len(parts) == 2:
                        sub_questions = [p.strip("? ") for p in parts if p.strip("? ")]
                    break

        return PlannerDecision(
            sub_questions=sub_questions,
            needs_web=needs_web,
            needs_papers=needs_papers,
            reasoning="[mock] heuristic keyword-based plan for offline mode",
        )

    def _mock_claims(self, context: dict):
        from app.schemas import ExtractedClaim, ExtractedClaims

        sources = context.get("sources", [])
        claims = []
        for i, s in enumerate(sources):
            text = s.get("snippet") or s.get("abstract") or ""
            claim_text = (text[:140] + "...") if len(text) > 140 else (text or "No content available")
            claims.append(
                ExtractedClaim(
                    claim_id=f"claim_{i+1}",
                    claim_text=f"[mock claim from {s.get('source_id')}] {claim_text}",
                    source_id=s.get("source_id", f"source_{i+1}"),
                )
            )
        return ExtractedClaims(claims=claims)

    def _mock_conflicts(self, context: dict):
        from app.schemas import ConflictAnalysis, ConflictRecord

        claims = context.get("claims", [])
        source_ids = sorted({c.get("source_id") for c in claims if c.get("source_id")})

        # Deterministic, calibrated rule for mock mode: only report a conflict
        # if the question explicitly signals sources are expected to disagree
        # (this mirrors how the real eval set curates deliberate-conflict
        # questions - see eval/questions.py). Otherwise stay quiet, matching
        # the spec's requirement to not manufacture disagreement.
        question = (context.get("question") or "").lower()
        expects_conflict = any(
            kw in question for kw in ["disagree", "conflicting", "debate", "controversy", "outdated"]
        )

        if not expects_conflict or len(source_ids) < 2:
            return ConflictAnalysis(
                conflicts=[],
                agreement_summary="[mock] Sources reviewed appear broadly consistent; no conflicts flagged.",
            )

        return ConflictAnalysis(
            conflicts=[
                ConflictRecord(
                    claim_summary="[mock] Sources differ on a key detail relevant to the question",
                    supporting_source_ids=source_ids[: len(source_ids) // 2] or source_ids[:1],
                    disputing_source_ids=source_ids[len(source_ids) // 2 :] or source_ids[-1:],
                    explanation="[mock] Simulated disagreement, e.g. one source predates a "
                    "later result the other source relies on.",
                )
            ],
            agreement_summary="[mock] Sources otherwise agree except for the flagged point.",
        )

    def _mock_validation(self, context: dict):
        from app.schemas import ValidationResult

        answer = context.get("draft_answer") or ""
        conflicts = context.get("conflicts") or []

        if not answer.strip():
            return ValidationResult(passed=False, feedback="Draft answer is empty.")

        has_citation_marker = "[source:" in answer.lower() or "[source" in answer.lower()
        if not has_citation_marker:
            return ValidationResult(
                passed=False, feedback="No inline citations found. Add [source: title/url] markers."
            )

        missing = []
        for c in conflicts:
            summary = c.get("claim_summary", "") if isinstance(c, dict) else c.claim_summary
            keyword = summary.split()[0] if summary else ""
            mentions_conflict_language = any(
                kw in answer.lower() for kw in ["disagree", "conflict", "differ", "however", "note:"]
            )
            if not mentions_conflict_language:
                missing.append(summary)

        if missing:
            return ValidationResult(
                passed=False,
                feedback=f"Answer does not surface {len(missing)} known conflict(s).",
                missing_conflict_summaries=missing,
            )

        return ValidationResult(passed=True)


def build_llm_provider() -> LLMProvider:
    """Factory reading env vars, used by app/main.py and the eval harness."""
    provider_name = os.environ.get("LLM_PROVIDER", "mock").lower()
    if os.environ.get("MOCK_MODE", "false").lower() == "true":
        provider_name = "mock"

    if provider_name == "anthropic":
        return AnthropicProvider()
    if provider_name == "gemini":
        return GeminiProvider()
    if provider_name == "mock":
        return MockLLMProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider_name}")
