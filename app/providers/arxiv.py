"""
arXiv API client. Uses the public Atom feed endpoint and stdlib
xml.etree.ElementTree for parsing - no extra dependency needed, per spec
section 4.2.

A 'mock' mode is included here too (MockArxivProvider) so paper retrieval
can be tested/run fully offline, mirroring the web search provider split.
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import httpx

from app.schemas import PaperReference

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"


class ArxivError(Exception):
    pass


class PaperSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[PaperReference]:
        ...


class ArxivProvider(PaperSearchProvider):
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> list[PaperReference]:
        params = {
            "search_query": f"all:{query}",
            "max_results": max_results,
        }
        try:
            resp = httpx.get(ARXIV_ENDPOINT, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ArxivError(str(e)) from e

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as e:
            raise ArxivError(f"Malformed Atom XML from arXiv: {e}") from e

        results: list[PaperReference] = []
        for i, entry in enumerate(root.findall("atom:entry", ATOM_NS)):
            title_el = entry.find("atom:title", ATOM_NS)
            summary_el = entry.find("atom:summary", ATOM_NS)
            id_el = entry.find("atom:id", ATOM_NS)
            published_el = entry.find("atom:published", ATOM_NS)
            authors = [
                a.findtext("atom:name", default="", namespaces=ATOM_NS)
                for a in entry.findall("atom:author", ATOM_NS)
            ]

            arxiv_url = id_el.text.strip() if id_el is not None and id_el.text else ""
            arxiv_id = arxiv_url.rsplit("/", 1)[-1] if arxiv_url else f"unknown_{i}"

            results.append(
                PaperReference(
                    source_id=f"paper_{i+1}",
                    title=(title_el.text or "").strip().replace("\n", " ") if title_el is not None else "Untitled",
                    url=arxiv_url,
                    arxiv_id=arxiv_id,
                    authors=[a.strip() for a in authors if a.strip()],
                    published=published_el.text.strip() if published_el is not None and published_el.text else None,
                    abstract=(summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else "",
                )
            )
        return results


class MockArxivProvider(PaperSearchProvider):
    """Deterministic fake papers. A query containing 'nopapertest' returns
    an empty list to exercise the no-paper-coverage degradation path."""

    def search(self, query: str, max_results: int = 5) -> list[PaperReference]:
        if "nopapertest" in query.lower():
            return []

        results = []
        for i in range(min(max_results, 2)):
            h = hashlib.sha256(f"{query}-paper-{i}".encode()).hexdigest()[:6]
            results.append(
                PaperReference(
                    source_id=f"paper_{i+1}",
                    title=f"[mock] A Study Related to '{query}' ({h})",
                    url=f"https://arxiv.org/abs/2500.{10000+i}",
                    arxiv_id=f"2500.{10000+i}",
                    authors=["A. Researcher", "B. Researcher"],
                    published="2025-01-01",
                    abstract=(
                        f"[mock abstract] This paper investigates aspects of '{query}'. "
                        "It stands in for a real arXiv abstract in offline/test mode."
                    ),
                )
            )
        return results


def build_paper_search_provider() -> PaperSearchProvider:
    provider_name = os.environ.get("PAPER_SEARCH_PROVIDER", "mock").lower()
    if os.environ.get("MOCK_MODE", "false").lower() == "true":
        provider_name = "mock"

    if provider_name == "arxiv":
        return ArxivProvider()
    if provider_name == "mock":
        return MockArxivProvider()
    raise ValueError(f"Unknown PAPER_SEARCH_PROVIDER: {provider_name}")
