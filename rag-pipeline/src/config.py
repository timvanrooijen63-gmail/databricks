"""
Central configuration — every catalog/schema/endpoint name lives here.
Set these as environment variables in your Databricks cluster config or
locally when running tests; never hardcode them in source files.
"""
import os

CATALOG = os.getenv("DATABRICKS_CATALOG", "main")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")

# Mosaic AI Vector Search
VS_INDEX_NAME = os.getenv("VS_INDEX_NAME", "my_vs_index")
VS_ENDPOINT_NAME = os.getenv("VS_ENDPOINT_NAME", "vs_endpoint")

# Databricks model-serving endpoint (proxies to an LLM such as Claude)
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "claude_endpoint")

# MLflow Prompt Registry name (registered under CATALOG.SCHEMA)
PROMPT_NAME = os.getenv("PROMPT_NAME", "qa_prompt")


def vs_index_full_name() -> str:
    return f"{CATALOG}.{SCHEMA}.{VS_INDEX_NAME}"


def prompt_full_name() -> str:
    return f"{CATALOG}.{SCHEMA}.{PROMPT_NAME}"
