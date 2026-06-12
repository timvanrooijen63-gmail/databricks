"""
Tests for the retriever factory.

VectorSearchRetrieverTool is mocked so these tests run without a live
Databricks cluster or Vector Search endpoint.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def test_retriever_tool_is_created(monkeypatch):
    """get_retriever_tool() should instantiate VectorSearchRetrieverTool once."""
    with patch("src.retriever.VectorSearchRetrieverTool") as MockTool:
        mock_instance = MagicMock()
        MockTool.return_value = mock_instance

        from src.retriever import get_retriever_tool  # noqa: PLC0415 — import inside patch

        tool = get_retriever_tool(num_results=3)

        MockTool.assert_called_once()
        call_kwargs = MockTool.call_args.kwargs
        assert call_kwargs.get("num_results") == 3
        assert tool is mock_instance


def test_retriever_invoke_returns_documents(monkeypatch):
    """Invoking the retriever tool should return a list of Document objects."""
    fake_docs = [Document(page_content="hello world"), Document(page_content="foo bar")]

    with patch("src.retriever.VectorSearchRetrieverTool") as MockTool:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = fake_docs
        MockTool.return_value = mock_instance

        from src.retriever import get_retriever_tool  # noqa: PLC0415

        tool = get_retriever_tool()
        results = tool.invoke("test query")

        assert len(results) == 2
        assert results[0].page_content == "hello world"
        assert results[1].page_content == "foo bar"
