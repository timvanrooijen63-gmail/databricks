"""
Prompt lifecycle helpers for the MLflow Prompt Registry (Unity Catalog).

The key pattern for the exam:
  - A prompt is versioned in Unity Catalog just like a model.
  - Aliases ("staging", "production") are mutable pointers to a version.
  - Promoting a prompt = moving the alias.  No code change, no redeployment.
  - Rolling back = moving the alias to a previous version number.

Requires mlflow >= 2.17 for mlflow.genai.register_prompt / set_prompt_alias.
"""
import mlflow

from .. import config

PROMPT_TEMPLATE = """\
You are a helpful assistant. Use only the provided context to answer the question.
Do not use prior knowledge; if the answer is not in the context, say so.

Context:
{context}

Question:
{question}

Answer:"""


def _set_registry() -> None:
    # Point MLflow at Unity Catalog so prompt names resolve to
    # <catalog>.<schema>.<prompt_name> rather than the workspace model registry.
    mlflow.set_registry_uri("databricks-uc")


def register_prompt(commit_message: str = "Initial version") -> object:
    """Register PROMPT_TEMPLATE as a new version and return the Prompt object."""
    _set_registry()
    prompt = mlflow.genai.register_prompt(
        name=config.prompt_full_name(),
        template=PROMPT_TEMPLATE,
        commit_message=commit_message,
    )
    return prompt


def set_alias(version: int, alias: str = "production") -> None:
    """Move *alias* to point at *version* — this is how you promote or roll back."""
    _set_registry()
    mlflow.genai.set_prompt_alias(
        name=config.prompt_full_name(),
        alias=alias,
        version=version,
    )


def load_prompt(alias: str = "production") -> object:
    """
    Load the prompt version currently bound to *alias*.

    Returns an mlflow Prompt object; access its text via .template.
    Loading by alias (not by version number) means the chain automatically
    picks up whichever version the operator has promoted — no code change needed.
    """
    _set_registry()
    uri = f"prompts:/{config.prompt_full_name()}@{alias}"
    return mlflow.load_prompt(uri)
