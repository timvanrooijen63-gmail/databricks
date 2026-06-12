# Databricks notebook source
# Register the RAG prompt in the MLflow Prompt Registry and set the
# "production" alias so the chain can load it by alias on every run.

# COMMAND ----------

# %pip install --upgrade databricks-langchain databricks-vectorsearch mlflow langchain langchain-core

# COMMAND ----------

# dbutils restarts the Python interpreter after %pip so that the freshly
# installed packages are importable.  This is a Databricks-specific pattern
# required because the runtime bundles older library versions.
dbutils.library.restartPython()  # noqa: F821  (dbutils is injected by Databricks)

# COMMAND ----------

import sys
# Add the repo root to sys.path so `from src.xxx import yyy` resolves.
# In Databricks Git Folders, the repo is mounted at /Workspace/Repos/<user>/<repo>.
# Adjust the path below to match your workspace.
sys.path.insert(0, "..")

# COMMAND ----------

from src.prompts.qa_prompt import register_prompt, set_alias

# Register version 1 of the prompt.
prompt_v1 = register_prompt(commit_message="Initial RAG prompt — v1")
print(f"Registered prompt version: {prompt_v1.version}")

# Bind the "production" alias to version 1.
# To promote a new version later: register_prompt(...) → set_alias(version=N, alias="production")
# To roll back: set_alias(version=N-1, alias="production")
set_alias(version=prompt_v1.version, alias="production")
print(f"Alias 'production' → version {prompt_v1.version}")

# COMMAND ----------

# Verify that the alias resolves correctly.
from src.prompts.qa_prompt import load_prompt

loaded = load_prompt(alias="production")
print("Loaded template (first 120 chars):", loaded.template[:120])
