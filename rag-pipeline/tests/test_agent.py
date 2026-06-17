"""
Smoke test for the docs agent (src/agent.py).

Unlike the unit tests in this repo, this test exercises the agent end to end and
therefore needs a live Databricks workspace: the "claude" serving endpoint, the
ai-search managed MCP server, and workspace.default.docs_index must be reachable
via the local Databricks CLI / SDK auth.

To keep offline CI green (the repo's convention is that the default test run needs
no live cluster), the whole module is skipped automatically when no workspace is
configured.
"""
import asyncio
import os
import sys

import pytest

# Allow `from src.agent import ...` when running from the repo root or tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _databricks_available() -> bool:
    """True only if a Databricks workspace host is resolvable from local auth."""
    try:
        from databricks.sdk import WorkspaceClient

        return bool(WorkspaceClient().config.host)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _databricks_available(),
    reason="No Databricks workspace configured; skipping live agent smoke test.",
)


def _answer_text(response) -> str:
    """Flatten a ResponsesAgentResponse's output items into plain text."""
    text = ""
    for item in response.output:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text += part.get("text", "")
        elif isinstance(content, str):
            text += content
    return text


def test_agent_discovers_tools_and_answers():
    from mlflow.types.responses import ResponsesAgentRequest

    from src.agent import DocsAgent

    agent = DocsAgent()

    # Tools are discovered from the MCP server at runtime (never hardcoded).
    tools = asyncio.run(agent.discover_tools())
    assert tools, "MCP server returned no tools"
    print("Discovered tools:", [getattr(t, "name", str(t)) for t in tools])

    # The agent produces a non-empty answer for a grounded question.
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "Who was the buyer of the house?"}]
    )
    response = agent.predict(request)
    answer = _answer_text(response)
    print("Answer:", answer)
    assert answer.strip(), "Agent returned an empty answer"
