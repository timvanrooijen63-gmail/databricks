# Databricks RAG Pipeline

A hands-on Retrieval-Augmented Generation (RAG) pipeline for the
**Databricks Certified Generative AI Engineer Associate** exam.

Runs entirely inside Databricks (Unity Catalog · Mosaic AI Vector Search ·
model-serving endpoints). Clone the repo into your workspace via
**Repos (Git Folders)** and run the notebooks in order.

---

## Repository layout

```
rag-pipeline/
├── src/
│   ├── config.py          # all names/endpoints in one place (env-var driven)
│   ├── retriever.py       # VectorSearchRetrieverTool wrapper
│   ├── prompts/
│   │   └── qa_prompt.py   # MLflow Prompt Registry helpers
│   └── chain.py           # LCEL chain: retriever | prompt | LLM | parse
├── notebooks/
│   ├── 01_setup.py            # pip install, register prompt v1, set alias
│   ├── 02_driver.py           # run a sample query end-to-end
│   └── 03_log_and_register.py # mlflow.langchain.log_model + UC registration
├── tests/                 # unit tests — no Databricks connection needed
└── .github/workflows/ci.yml
```

---

## Quick start (local tests)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Set environment variables to override defaults (optional for local tests):

```bash
export DATABRICKS_CATALOG=my_catalog
export DATABRICKS_SCHEMA=my_schema
export VS_INDEX_NAME=my_vs_index
export VS_ENDPOINT_NAME=my_vs_endpoint
export LLM_ENDPOINT_NAME=my_claude_endpoint
export PROMPT_NAME=qa_prompt
```

---

## Running in Databricks

1. **Clone this repo** into your workspace via Repos (Git Folders).
2. Open `notebooks/01_setup.py` on a cluster with Unity Catalog enabled.
3. Run `01_setup.py` — installs libraries, registers prompt v1, sets the
   `"production"` alias.
4. Run `02_driver.py` to test an end-to-end query.
5. Run `03_log_and_register.py` to log and register the chain in Unity Catalog.

---

## Prompt promotion workflow

This project uses the **MLflow Prompt Registry** (Unity Catalog) as the
centrepiece pattern. The chain always loads its prompt by *alias*, not by
version number:

```python
mlflow.load_prompt("prompts:/catalog.schema.qa_prompt@production")
```

This means:

```
Developer                   Operator
─────────                   ────────
register_prompt(v2)         # new version exists but nothing uses it yet
set_alias(v2, "staging")    # point "staging" at v2; chain in staging env picks it up
                            # ... smoke test passes ...
set_alias(v2, "production") # promote: production chain now uses v2 — zero redeploy
```

### Rollback

If v2 causes problems, roll back in one call:

```python
from src.prompts.qa_prompt import set_alias
set_alias(version=1, alias="production")   # production chain reverts instantly
```

No code changes, no PR, no redeployment — the alias is the only thing that moves.

### Version history

```
v1  ← production (initial)
v2  ← staging (candidate)
```

After promotion:

```
v1
v2  ← production  ← staging
```

After rollback:

```
v1  ← production
v2  ← staging
```

---

## Key design decisions

| Decision | Why |
|---|---|
| `VectorSearchRetrieverTool` over raw `index.similarity_search()` | Parses `data_array` results into `List[Document]` automatically; integrates with LCEL and agents without manual field extraction. |
| Prompt loaded by alias, never hardcoded | Decouples deployment from prompt iteration. Exam topic: alias = mutable pointer, version = immutable snapshot. |
| `%pip install --upgrade` in notebooks | Databricks runtimes bundle specific library versions that may be stale; `--upgrade` ensures the pinned versions from `requirements.txt` are used. |
| `dbutils.library.restartPython()` after pip | Required for newly installed packages to become importable in the same notebook session. |
| Notebooks in py source format | Diffs cleanly in Git; renders as a notebook in Databricks with no conversion needed. |
| Unity Catalog for model + prompt registry | Single governance layer for data, models, and prompts; access control via UC grants. |
