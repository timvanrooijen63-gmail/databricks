"""Tests for the prompt template — no Databricks connection required."""
from langchain_core.prompts import PromptTemplate

from src.prompts.qa_prompt import PROMPT_TEMPLATE


def test_template_has_required_variables():
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    assert {"context", "question"}.issubset(set(prompt.input_variables))


def test_template_renders_both_values():
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    rendered = prompt.format(context="some context", question="what?")
    assert "some context" in rendered
    assert "what?" in rendered


def test_template_is_non_empty():
    assert PROMPT_TEMPLATE.strip(), "PROMPT_TEMPLATE must not be blank"
