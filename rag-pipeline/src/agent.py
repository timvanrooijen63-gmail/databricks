"""
Deployable Databricks agent (model-from-code).

Answers questions grounded in the ``workspace.default.docs_index`` AI Search
index, reached through the Databricks **managed MCP** server. A LangGraph ReAct
agent drives a ChatDatabricks LLM ("claude"); the MCP server exposes the
vector-search tool(s), which are discovered at runtime (never hardcoded).

The agent is wrapped in MLflow's ``ResponsesAgent`` interface so it can be logged
with ``mlflow.pyfunc.log_model(python_model="src/agent.py", ...)`` and served from
a Databricks model-serving endpoint. ``mlflow.models.set_model(...)`` at the bottom
tells MLflow which object to serve.

Wrapper shape follows the official Databricks examples:
  - https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp
  - https://docs.databricks.com/aws/en/generative-ai/agent-framework/unstructured-retrieval-tools
"""
import asyncio
from typing import Any, Generator

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langgraph.prebuilt import create_react_agent
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

# --- Fixed config (provided by the project; do not change) ------------------
# LLM is a Databricks serving endpoint named "claude" that proxies Claude.
LLM_ENDPOINT_NAME = "claude"
# Path of the managed MCP server that fronts the AI Search index
# workspace.default.docs_index. Joined to the workspace host at runtime so the
# same code works across workspaces without hardcoding a host.
MCP_SERVER_PATH = "/api/2.0/mcp/ai-search/workspace/default"

# Optional grounding instruction (an addition beyond the bare API spec): keeps
# answers tied to retrieved documents rather than the model's prior knowledge.
SYSTEM_PROMPT = (
    "You are a assistent that answers questions on house sales based on purchase agreements." \
    "Do not make stuff up. Verify all questions. Only answer questions related to house sales"
)


class DocsAgent(ResponsesAgent):
    """ResponsesAgent wrapping a LangGraph ReAct agent over the managed MCP server."""

    def __init__(self) -> None:
        # WorkspaceClient picks up auth automatically: from the request inside a
        # served model (resource passthrough), or from the local Databricks CLI
        # config when run interactively.
        self.workspace_client = WorkspaceClient()
        host = self.workspace_client.config.host
        self.mcp_server_url = f"{host}{MCP_SERVER_PATH}"
        self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    def _build_mcp_client(self) -> DatabricksMultiServerMCPClient:
        # Built fresh for each request. NOTE: as of langchain-mcp-adapters 0.1.0
        # (pulled in by databricks-langchain 0.20.0) MultiServerMCPClient is NO
        # LONGER a context manager — `async with client` raises NotImplementedError.
        # Instead you call `await client.get_tools()` directly, and the returned
        # tools open a fresh, short-lived MCP session per invocation. That is
        # precisely the per-request behavior we want under model serving (no
        # module-level/long-lived open session), so we don't manage one ourselves.
        return DatabricksMultiServerMCPClient(
            [
                DatabricksMCPServer(
                    name="vector-search",
                    url=self.mcp_server_url,
                    workspace_client=self.workspace_client,
                ),
            ]
        )

    async def discover_tools(self) -> list[Any]:
        """List the tools exposed by the MCP server (used by the smoke test)."""
        client = self._build_mcp_client()
        return await client.get_tools()

    async def _arun(self, messages: list[dict]) -> list[Any]:
        # Discover tools at runtime, build the ReAct agent, and invoke it. Each
        # tool call manages its own MCP session internally (see _build_mcp_client).
        client = self._build_mcp_client()
        tools = await client.get_tools()
        agent = create_react_agent(self.llm, tools=tools, prompt=SYSTEM_PROMPT)
        result = await agent.ainvoke({"messages": messages})
        # LangGraph returns the full message history; we hand it all back so
        # predict() can pick the final assistant turn.
        return result["messages"]

    @staticmethod
    def _to_chat_messages(request: ResponsesAgentRequest) -> list[dict]:
        """Convert Responses-format input items into chat-completions dicts.

        LangChain/LangGraph accept ``{"role": ..., "content": ...}`` dicts. The
        Responses ``content`` field may be a plain string or a list of typed
        content parts; flatten the latter to text for this text-only agent.
        """
        messages: list[dict] = []
        for item in request.input:
            data = item.model_dump()
            role = data.get("role")
            content = data.get("content")
            if not role or content is None:
                continue
            if isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
            else:
                text = content
            messages.append({"role": role, "content": text})
        return messages

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # Model Serving invokes predict() synchronously (no running event loop),
        # so asyncio.run() is safe here to bridge into the async MCP session.
        out_messages = asyncio.run(self._arun(self._to_chat_messages(request)))
        final = out_messages[-1]
        text = getattr(final, "content", None)
        if text is None:
            text = str(final)
        item = self.create_text_output_item(
            text=text,
            id=getattr(final, "id", None) or "msg-0",
        )
        return ResponsesAgentResponse(output=[item])

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # Minimal streaming implementation: emit the final answer as a single
        # completed output item. predict() above reconstructs from these
        # "response.output_item.done" events, matching the Databricks convention.
        for item in self.predict(request).output:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=item,
            )


# Tell MLflow which object to serve when this file is logged as model-from-code.
mlflow.models.set_model(DocsAgent())
