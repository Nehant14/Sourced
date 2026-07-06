import pytest

from app.providers.arxiv import MockArxivProvider
from app.providers.llm import MockLLMProvider
from app.providers.search import MockWebSearchProvider


@pytest.fixture
def llm():
    return MockLLMProvider()


@pytest.fixture
def web_provider():
    return MockWebSearchProvider()


@pytest.fixture
def paper_provider():
    return MockArxivProvider()
