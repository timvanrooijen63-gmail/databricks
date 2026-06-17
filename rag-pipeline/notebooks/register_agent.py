# Databricks notebook source
# Log, register, and deploy the docs agent (src/agent.py) to Unity Catalog.
#
# This notebook:
#   1. installs the agent dependencies and restarts Python,
#   2. points MLflow at Unity Catalog,
#   3. builds the resources list so the served endpoint is granted access to
#      BOTH the "claude" serving endpoint AND the docs_index behind MCP,
#   4. logs the model-from-code agent, registers it to UC, and deploys it.
#
# Run this in Databricks (it uses %pip, dbutils, and live workspace auth).

# COMMAND ----------

# MAGIC %pip install --upgrade databricks-langchain==0.20.0 databricks-mcp==0.9.0 langgraph==1.2.5 langgraph-prebuilt==1.1.0 langgraph-checkpoint==4.1.1 mlflow==3.12.0 databricks-agents==1.11.0
# MAGIC
# MAGIC # Versions pinned to a set verified end-to-end (Jun 2026). The langgraph
# MAGIC # sub-packages (langgraph-prebuilt, langgraph-checkpoint) MUST be pinned
# MAGIC # alongside langgraph core: a bare `--upgrade langgraph` leaves the
# MAGIC # cluster's preinstalled langgraph-prebuilt out of sync, and
# MAGIC # create_react_agent then fails with
# MAGIC #   "cannot import name 'ExecutionInfo' from 'langgraph.runtime'".

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821  (dbutils is provided by Databricks)

# COMMAND ----------

import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

# Register models into Unity Catalog (not the workspace registry).
mlflow.set_registry_uri("databricks-uc")

# --- Fixed config (do not change) -------------------------------------------
LLM_ENDPOINT_NAME = "claude"
INDEX_NAME = "workspace.default.docs_index"
UC_MODEL_NAME = "workspace.default.docs_agent"

ws = WorkspaceClient()
host = ws.config.host
# Managed MCP server fronting the AI Search index workspace.default.docs_index.
# This is the URL the served agent uses at runtime (src/agent.py).
MCP_SERVER_URL = f"{host}/api/2.0/mcp/ai-search/workspace/default"

# COMMAND ----------

# GOVERNANCE (critical): the resources list MUST include the "claude" serving
# endpoint AND the docs_index, or the deployed endpoint can't call the LLM or
# query the index.
#
# Subtlety verified locally (Jun 2026): DatabricksMCPClient.get_databricks_resources()
# only parses the canonical ".../mcp/vector-search/<catalog>/<schema>" URL form.
# For the fixed-config ".../mcp/ai-search/..." URL above it returns [] (and logs
# an error) — which would silently drop the index from the resources list. The
# ai-search and vector-search URLs resolve to the SAME index, so we introspect
# with the vector-search form to obtain the docs_index resource, and fall back to
# declaring it explicitly if introspection ever comes back empty.
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]

VS_MCP_URL = f"{host}/api/2.0/mcp/vector-search/workspace/default"
index_resources = DatabricksMCPClient(
    server_url=VS_MCP_URL,
    workspace_client=ws,
).get_databricks_resources()
if not index_resources:  # safety net — never deploy without the index resource
    index_resources = [DatabricksVectorSearchIndex(index_name=INDEX_NAME)]
resources += index_resources

print("Logging with resources:")
for r in resources:
    print(f"  - {type(r).__name__}: {r.to_dict()}")

# Fail fast if governance is not satisfied.
_dumps = [r.to_dict() for r in resources]
assert any("serving_endpoint" in d for d in _dumps), "claude serving endpoint missing"
assert any("vector_search_index" in d for d in _dumps), "docs_index missing"

# COMMAND ----------

# Resolve the path to the model-from-code file. This notebook lives in
# notebooks/, so the agent module is one directory up under src/. (The agent is
# logged by file path, i.e. "src/agent.py" relative to the repo root.)
AGENT_SCRIPT = os.path.abspath(os.path.join(os.getcwd(), "..", "src", "agent.py"))
assert os.path.exists(AGENT_SCRIPT), f"agent.py not found at {AGENT_SCRIPT}"

# A Responses-format request used to infer/validate the model signature. MLflow
# calls predict() on this example at log time, so we use a benign question: the
# "claude" endpoint's PII/privacy output guardrail blocks answers that name
# people (e.g. "Who was the buyer?"), which would fail logging. This neutral
# question exercises the same code path without tripping the guardrail.
input_example = {
    "input": [{"role": "user", "content": "What topics can you help me with?"}]
}

# Pin the serving environment to EXACTLY what is installed in this notebook
# (post-%pip). This is what prevents the served container from drifting — in
# particular it pins the langgraph sub-packages together, avoiding the
# "cannot import name 'ExecutionInfo' from 'langgraph.runtime'" skew.
from importlib.metadata import version

pip_requirements = [
    f"{pkg}=={version(pkg)}"
    for pkg in [
        "databricks-langchain",
        "databricks-mcp",
        "langgraph",
        "langgraph-prebuilt",
        "langgraph-checkpoint",
        "mlflow",
    ]
]
print("pip_requirements:", pip_requirements)

with mlflow.start_run():
    logged = mlflow.pyfunc.log_model(
        name="agent",
        python_model=AGENT_SCRIPT,
        resources=resources,
        input_example=input_example,
        pip_requirements=pip_requirements,
    )

print(f"Logged model URI: {logged.model_uri}")

# COMMAND ----------

# Register the logged model into Unity Catalog.
registered = mlflow.register_model(logged.model_uri, UC_MODEL_NAME)
print(f"Registered {UC_MODEL_NAME} version {registered.version}")

# COMMAND ----------

# Deploy the registered version to a model-serving endpoint. Resource auth
# passthrough (configured via resources= above) lets the endpoint query the
# index and call the "claude" LLM.
from databricks import agents

deployment = agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=registered.version,
)
print(deployment)
