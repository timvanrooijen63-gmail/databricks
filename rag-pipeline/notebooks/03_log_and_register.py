# Databricks notebook source
# Log the RAG chain to MLflow and register it in Unity Catalog so it can be
# served from a Databricks model-serving endpoint.
#
# Why log_model with a signature?
#   The signature records the expected input/output schema so that downstream
#   consumers (and the serving endpoint) know the contract without reading code.

# COMMAND ----------

import sys
sys.path.insert(0, "..")

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from src.chain import build_chain
from src import config

# Point MLflow at Unity Catalog for model registration.
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

chain = build_chain(alias="production")

input_example = "What is Retrieval-Augmented Generation?"
output_example = chain.invoke(input_example)  # captures a real output for the signature

signature = infer_signature(model_input=input_example, model_output=output_example)

# COMMAND ----------

registered_model_name = f"{config.CATALOG}.{config.SCHEMA}.rag_chain"

with mlflow.start_run():
    model_info = mlflow.langchain.log_model(
        lc_model=chain,
        artifact_path="rag_chain",
        input_example=input_example,
        signature=signature,
        # Registers the model in Unity Catalog under <catalog>.<schema>.rag_chain
        registered_model_name=registered_model_name,
    )

print(f"Model logged: {model_info.model_uri}")
print(f"Registered as: {registered_model_name}")

# COMMAND ----------

# After registration, navigate to:
#   Catalog Explorer → <catalog> → <schema> → rag_chain → Serving
# to deploy the model to a real-time serving endpoint.
