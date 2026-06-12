"""
LCEL RAG chain: retriever | format | prompt | LLM | parse.

The prompt is always fetched from the MLflow Prompt Registry by alias rather
than from a hardcoded string.  This means operators can promote a new prompt
version (or roll back) by moving the alias — no code change, no redeployment.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from databricks_langchain import ChatDatabricks

from . import config
from .prompts.qa_prompt import load_prompt
from .retriever import get_retriever_tool


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_chain(alias: str = "production"):
    """
    Build and return an LCEL chain that answers questions via RAG.

    Parameters
    ----------
    alias : str
        MLflow Prompt Registry alias to load.  Changing which version
        "production" points at is the entire promotion / rollback mechanism.

    Returns
    -------
    A LangChain Runnable that accepts a question string and returns a string.
    """
    # Load the prompt template from the registry — the alias indirection is the
    # whole point: swapping "production" to v2 here requires zero code changes.
    mlflow_prompt = load_prompt(alias)
    prompt = PromptTemplate.from_template(mlflow_prompt.template)

    # ChatDatabricks uses the workspace token automatically; no credentials needed.
    llm = ChatDatabricks(endpoint=config.LLM_ENDPOINT_NAME)

    # Wrap the retriever tool so it behaves as an LCEL Runnable (str → List[Document]).
    retriever = RunnableLambda(get_retriever_tool().invoke)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
