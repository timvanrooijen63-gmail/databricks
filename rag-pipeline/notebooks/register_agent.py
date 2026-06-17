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

# MAGIC %pip install --upgrade databricks-langchain databricks-mcp langgraph mlflow databricks-agents

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821  (dbutils is provided by Databricks)

# COMMAND ----------

import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from mlflow.models.resources import DatabricksServingEndpoint

# Register models into Unity Catalog (not the workspace registry).
mlflow.set_registry_uri("databricks-uc")

# --- Fixed config (do not change) -------------------------------------------
LLM_ENDPOINT_NAME = "claude"
UC_MODEL_NAME = "workspace.default.docs_agent"

ws = WorkspaceClient()
host = ws.config.host
# Managed MCP server fronting the AI Search index workspace.default.docs_index.
MCP_SERVER_URL = f"{host}/api/2.0/mcp/ai-search/workspace/default"

# COMMAND ----------

# GOVERNANCE (critical): the resources list must include the "claude" serving
# endpoint AND the docs_index. get_databricks_resources() introspects the MCP
# server and returns the underlying index resource(s); without both, the deployed
# endpoint will lack permission to call the LLM or query the index.
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
resources += DatabricksMCPClient(
    server_url=MCP_SERVER_URL,
    workspace_client=ws,
).get_databricks_resources()

print("Logging with resources:")
for r in resources:
    print(f"  - {r}")

# COMMAND ----------

# Resolve the path to the model-from-code file. This notebook lives in
# notebooks/, so the agent module is one directory up under src/. (The agent is
# logged by file path, i.e. "src/agent.py" relative to the repo root.)
AGENT_SCRIPT = os.path.abspath(os.path.join(os.getcwd(), "..", "src", "agent.py"))
assert os.path.exists(AGENT_SCRIPT), f"agent.py not found at {AGENT_SCRIPT}"

# A Responses-format request used to infer the model signature.
input_example = {
    "input": [{"role": "user", "content": "Who was the buyer of the house?"}]
}

with mlflow.start_run():
    logged = mlflow.pyfunc.log_model(
        name="agent",
        python_model=AGENT_SCRIPT,
        resources=resources,
        input_example=input_example,
        # Keep the serving environment in the same dependency band as
        # requirements.txt / the %pip cell above.
        pip_requirements=[
            "databricks-langchain",
            "databricks-mcp",
            "langgraph",
            "mlflow",
        ],
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
