"""
Retriever backed by Mosaic AI Vector Search.

VectorSearchRetrieverTool is preferred over calling the index client directly
because it parses raw data_array responses into LangChain Document objects
automatically and integrates cleanly with both LCEL chains and agents.
"""
from databricks_langchain import VectorSearchRetrieverTool

from . import config


def get_retriever_tool(num_results: int = 5) -> VectorSearchRetrieverTool:
    # Auth is automatic when code runs inside a Databricks notebook or job —
    # never pass PAT tokens or host URLs here.
    #
    # Note: VectorSearchRetrieverTool resolves the Vector Search endpoint from
    # the index metadata.  If your installed version also accepts an explicit
    # `endpoint_name` kwarg, you can pass config.VS_ENDPOINT_NAME here; omit
    # it if the constructor raises an unexpected-keyword-argument error.
    return VectorSearchRetrieverTool(
        index_name=config.vs_index_full_name(),
        num_results=num_results,
        columns=["content"],
        tool_name="document_search",
        tool_description="Search the knowledge base for relevant document chunks.",
    )
