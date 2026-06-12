"""
Tests for build_chain() — no live Databricks connection required.

Strategy:
  - Mock load_prompt to return a stub with .template set to a minimal template.
  - Mock ChatDatabricks to return a fake LLM that yields an AIMessage.
  - Mock get_retriever_tool to return a callable that produces Document objects.
  - StrOutputParser extracts AIMessage.content, so the chain returns a plain str.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage


def _make_mocks():
    """Return (mock_load_prompt, MockChatDatabricks, mock_get_retriever_tool)."""
    mock_prompt_obj = MagicMock()
    mock_prompt_obj.template = "{context}\n{question}"

    mock_load = MagicMock(return_value=mock_prompt_obj)

    # The mock LLM must be a Runnable; patch at the class level so that
    # ChatDatabricks(endpoint=...) returns our fake instance.
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="mocked answer")
    # Make the mock behave as a LangChain Runnable by delegating __or__ properly.
    # We patch at import-time so the LCEL pipeline sees our fake instance.
    MockLLMClass = MagicMock(return_value=fake_llm)

    fake_tool = MagicMock()
    fake_tool.invoke.return_value = [Document(page_content="relevant chunk")]
    mock_retriever = MagicMock(return_value=fake_tool)

    return mock_load, MockLLMClass, mock_retriever


def test_build_chain_returns_runnable():
    mock_load, MockLLMClass, mock_retriever = _make_mocks()
    with (
        patch("src.chain.load_prompt", mock_load),
        patch("src.chain.ChatDatabricks", MockLLMClass),
        patch("src.chain.get_retriever_tool", mock_retriever),
    ):
        from src.chain import build_chain  # noqa: PLC0415

        chain = build_chain(alias="production")
        assert chain is not None
        assert hasattr(chain, "invoke"), "chain must be a LangChain Runnable"


def test_chain_invoke_returns_string():
    mock_load, MockLLMClass, mock_retriever = _make_mocks()

    # Make the fake LLM participate in the LCEL pipeline by making it a real
    # Runnable that wraps our desired output.
    from langchain_core.runnables import RunnableLambda

    real_fake_llm = RunnableLambda(lambda _: AIMessage(content="mocked answer"))

    with (
        patch("src.chain.load_prompt", mock_load),
        patch("src.chain.ChatDatabricks", MagicMock(return_value=real_fake_llm)),
        patch("src.chain.get_retriever_tool", mock_retriever),
    ):
        from src.chain import build_chain  # noqa: PLC0415

        chain = build_chain(alias="production")
        result = chain.invoke("what is RAG?")

        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert result == "mocked answer"
