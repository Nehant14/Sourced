"""
Web search provider abstraction: TavilyProvider (real) and
MockWebSearchProvider (offline, deterministic).
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx

from app.schemas import SourceReference


class WebSearchError(Exception):
    pass


class WebSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        ...


class TavilyProvider(WebSearchProvider):
    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not set and no api_key passed in")
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        try:
            resp = httpx.post(
                self.ENDPOINT,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            # Callers (web_retriever_node) are responsible for catching this
            # and degrading gracefully - see section 8 of the spec.
            raise WebSearchError(str(e)) from e

        results = []
        now = datetime.now(timezone.utc)
        for i, item in enumerate(data.get("results", [])[:max_results]):
            results.append(
                SourceReference(
                    source_id=f"web_{i+1}",
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:500],
                    retrieved_at=now,
                )
            )
        return results


class MockWebSearchProvider(WebSearchProvider):
    """Deterministic fake results. A query containing the token
    'noresultstest' returns an empty list, to exercise the zero-results
    degradation path in tests without needing a special flag."""

    def search(self, query: str, max_results: int = 5) -> list[SourceReference]:
        if "noresultstest" in query.lower():
            return []

        now = datetime.now(timezone.utc)
        results = []
        for i in range(min(max_results, 3)):
            h = hashlib.sha256(f"{query}-{i}".encode()).hexdigest()[:6]
            results.append(
                SourceReference(
                    source_id=f"web_{i+1}",
                    title=f"[mock] Result {i+1} for '{query}' ({h})",
                    url=f"https://example.com/mock/{h}",
                    snippet=(
                        f"This is a mock web snippet about '{query}', result {i+1}. "
                        "It stands in for real search content in offline/test mode."
                    ),
                    retrieved_at=now,
                )
            )
        return results


def build_web_search_provider() -> WebSearchProvider:
    provider_name = os.environ.get("WEB_SEARCH_PROVIDER", "mock").lower()
    if os.environ.get("MOCK_MODE", "false").lower() == "true":
        provider_name = "mock"

    if provider_name == "tavily":
        return TavilyProvider()
    if provider_name == "mock":
        return MockWebSearchProvider()
    raise ValueError(f"Unknown WEB_SEARCH_PROVIDER: {provider_name}")
