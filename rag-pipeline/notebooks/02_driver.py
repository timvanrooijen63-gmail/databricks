# Databricks notebook source
# Thin driver notebook — imports the chain from src/ and runs a sample query.
# Run 01_setup.py first to ensure the "production" alias exists.

# COMMAND ----------

import sys
sys.path.insert(0, "..")

# COMMAND ----------

from src.chain import build_chain

# build_chain() fetches the prompt from the MLflow Prompt Registry by alias.
# Swapping which version "production" points at is the entire promotion mechanism.
chain = build_chain(alias="production")
print("Chain built successfully.")

# COMMAND ----------

question = "What is Retrieval-Augmented Generation and why is it useful?"

answer = chain.invoke(question)
print(f"Q: {question}\nA: {answer}")
