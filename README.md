# langchain-bigquery-hybridsearch

BigQuery Hybrid Search extension for [langchain-google-community](https://github.com/langchain-ai/langchain-google).

Combines BigQuery `VECTOR_SEARCH()` (semantic similarity) with `SEARCH()` (full-text keyword matching) into a single retrieval step.

## Installation

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```python
from langchain_bigquery_hybridsearch import BigQueryHybridSearchVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

store = BigQueryHybridSearchVectorStore(
    project_id="my-project",
    dataset_name="my_dataset",
    table_name="documents",
    location="US",
    embedding=VertexAIEmbeddings(model_name="text-embedding-005"),
    distance_type="COSINE",
    search_analyzer="LOG_ANALYZER",
)

# Pre-filter mode (default): keyword filter → vector ranking
results = store.hybrid_search(
    query="How to optimize BigQuery performance?",
    text_query="BigQuery optimization",
    k=10,
)

# RRF mode: independent keyword + vector search → merged ranking
results = store.hybrid_search_with_score(
    query="How to optimize BigQuery performance?",
    text_query="BigQuery optimization",
    k=10,
    fetch_k=50,
    hybrid_search_mode="rrf",
)
```

## Search Modes

### Pre-filter

Uses `SEARCH()` to narrow candidates, then `VECTOR_SEARCH()` to rank by embedding distance. Results **must** contain the keyword tokens.

```
Query → SEARCH(content, keywords) → filtered rows → VECTOR_SEARCH() → top-k
```

### RRF (Reciprocal Rank Fusion)

Runs both searches independently and combines results:

```
Query → VECTOR_SEARCH() → vector_rank ─┐
                                       ├→ RRF score → top-k
Query → SEARCH()         → text_rank  ─┘
```

RRF score: `1/(k + vector_rank) + 1/(k + text_rank)` where `k` defaults to 60.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_fields` | `List[str]` | `[content_field]` | Columns for `SEARCH()` |
| `search_analyzer` | `str` | `LOG_ANALYZER` | Text analyzer |
| `search_analyzer_options` | `str` | `None` | Analyzer options (JSON) |
| `hybrid_search_mode` | `str` | `pre_filter` | Default mode |
| `rrf_k` | `int` | `60` | RRF constant |
| `query_task_type` | `Optional[str]` | `None` | Task type for query embeddings |
| `document_task_type` | `Optional[str]` | `None` | Task type for document embeddings |

All parameters from `BigQueryVectorStore` (`distance_type`, `extra_fields`, etc.) are also supported.

## Embedding Task Types

Vertex AI / Google Generative AI embedding models accept a [task type](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types) hint that tunes the embedding for a specific downstream use case. The same text produces different vectors depending on the task type, so matching the type on both the indexing and query sides usually improves retrieval quality.

| Task type | Typical use |
|-----------|-------------|
| `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` | Asymmetric retrieval (default for Google embeddings) |
| `QUESTION_ANSWERING` | Q&A retrieval — set on **both** sides |
| `FACT_VERIFICATION` | Claim verification — set on **both** sides |
| `SEMANTIC_SIMILARITY` | Symmetric similarity — set on **both** sides |
| `CODE_RETRIEVAL_QUERY` | Code search — set on the **query** side only |

When `query_task_type` / `document_task_type` is left as `None`, the embedding model's own default is used (`RETRIEVAL_QUERY` for queries, `RETRIEVAL_DOCUMENT` for documents in the Google integrations). In particular, if you configure the task type on the embedding instance itself — e.g. `GoogleGenerativeAIEmbeddings(model=..., task_type="QUESTION_ANSWERING")` — leaving the store fields as `None` preserves that setting; the store does not override it.

The store works with both `langchain-google-genai` (which uses the `task_type` kwarg) and the deprecated `langchain-google-vertexai` (which uses `embeddings_task_type`); the right kwarg is detected automatically. Embeddings without any task-type kwarg log a warning once and continue without it.

### Instance-level configuration

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_bigquery_hybridsearch import BigQueryHybridSearchVectorStore

# Q&A retrieval — same task type on both sides
store = BigQueryHybridSearchVectorStore(
    project_id="my-project",
    dataset_name="my_dataset",
    table_name="docs",
    embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    query_task_type="QUESTION_ANSWERING",
    document_task_type="QUESTION_ANSWERING",
    hybrid_search_mode="rrf",
)
```

```python
# Code search — only the query side differs
store = BigQueryHybridSearchVectorStore(
    ...,
    query_task_type="CODE_RETRIEVAL_QUERY",
    # document_task_type=None → falls back to RETRIEVAL_DOCUMENT
)
```

### Per-call override

The kwargs on `hybrid_search`, `hybrid_search_with_score`, and `add_texts` win over the instance-level fields:

```python
results = store.hybrid_search(
    query="How does VECTOR_SEARCH work?",
    text_query="VECTOR_SEARCH",
    k=10,
    query_task_type="QUESTION_ANSWERING",
)

ids = store.add_texts(
    texts=["def foo(): ...", "class Bar: ..."],
    document_task_type="CODE_RETRIEVAL_QUERY",
)
```

### Retriever path

```python
retriever = store.as_retriever(
    search_type="hybrid",
    search_kwargs={"k": 4, "query_task_type": "FACT_VERIFICATION"},
)
docs = retriever.invoke("Earth orbits the Sun.")
```

## Score Semantics

The numeric meaning of `score` returned by `*_with_score` methods depends on which mode is used. The two cases below are intentionally opposite, so always check the mode before comparing scores across calls.

### Pre-filter mode and inherited similarity search — smaller is more similar

Both `BigQueryHybridSearchVectorStore.hybrid_search_with_score()` (in **pre-filter** mode) and the inherited `BigQueryVectorStore.similarity_search_with_score()` map BigQuery's `distance` column directly to `score`:

```sql
SELECT base.*, distance AS score
FROM VECTOR_SEARCH(..., distance_type => "COSINE", ...)
ORDER BY score          -- ascending; smaller distance first
```

For all three `distance_type` values BigQuery normalizes the result so that *smaller is closer*:

| `distance_type` | distance returned | interpretation |
|---|---|---|
| `EUCLIDEAN` (default of base store) | `‖a − b‖` (L2) | smaller = more similar |
| `COSINE` | `1 − cos(θ)` | smaller = more similar |
| `DOT_PRODUCT` | `−(a·b)` (negated) | smaller = more similar |

Note that this differs from the typical LangChain convention (where larger score = more similar, e.g. Chroma, Pinecone). The behavior comes from `langchain-google-community`'s `BigQueryVectorStore` and is preserved by this hybrid extension for consistency.

### RRF mode — larger is more relevant

In `hybrid_search_with_score(..., hybrid_search_mode="rrf")` the score is the Reciprocal Rank Fusion sum:

```
rrf_score = 1/(k + vector_rank) + 1/(k + text_rank)
```

A document that ranks high in either retriever contributes a larger reciprocal, so **higher RRF score = stronger combined evidence**. Results are returned in descending order of this score.

## Testing

### Unit Tests

No GCP credentials needed. Unit tests use mocks to verify SQL generation logic.

#### 1. Install dev dependencies

```bash
uv venv  # if .venv does not exist yet
uv pip install -e ".[dev]"
```

The `dev` extra installs `pytest`, `pytest-asyncio`, `pytest-mock`, and `python-dotenv`.

#### 2. Run unit tests

```bash
# All unit tests
pytest tests/unit_tests/ -v

# A single file
pytest tests/unit_tests/test_sql_generation.py -v

# A single test function
pytest tests/unit_tests/test_sql_generation.py::<test_function_name> -v

# Filter by keyword
pytest tests/unit_tests/ -v -k "rrf"

# Exclude integration tests explicitly
pytest tests/unit_tests/ -v -m "not integration"
```

Notes:
- `tests/conftest.py` auto-loads `tests/.env` via `python-dotenv`, but unit tests run without any env vars.
- `pytest-asyncio` is configured with `asyncio_mode = "auto"`, so async tests need no decorator.

### Integration Tests

#### 1. Prerequisites

- Google Cloud SDK (`gcloud`) installed and authenticated
- A GCP project with billing enabled
- The following APIs enabled:
  - BigQuery API
  - Vertex AI API (for embedding model)

```bash
gcloud auth application-default login
gcloud config set project your-gcp-project-id

# Enable required APIs
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

#### 2. IAM Permissions

The authenticated account needs these roles (or equivalent permissions):

| Role | Purpose |
|------|---------|
| `roles/bigquery.dataEditor` | Create/delete tables, insert data |
| `roles/bigquery.jobUser` | Run queries (VECTOR_SEARCH, SEARCH) |
| `roles/bigquery.dataViewer` | Read table data and metadata |
| `roles/aiplatform.user` | Access Vertex AI embedding models |

```bash
# Example: grant roles to your account
PROJECT_ID=your-gcp-project-id
ACCOUNT=$(gcloud config get account)

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:$ACCOUNT" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:$ACCOUNT" \
  --role="roles/bigquery.jobUser"
```

#### 3. BigQuery Dataset Setup

The test fixture creates tables automatically, but the dataset must exist or the authenticated account must have `bigquery.datasets.create` permission.

```bash
# Option A: Let the test create the dataset automatically (requires bigquery.datasets.create)

# Option B: Create the dataset manually
bq --location=US mk --dataset your-gcp-project-id:test_hybridsearch
```

#### 4. Install Additional Dependencies

```bash
uv pip install langchain-google-vertexai
```

#### 5. Configure Environment Variables

Copy the template and fill in your values:

```bash
cp tests/.env.example tests/.env
```

```bash
# tests/.env
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
GOOGLE_CLOUD_LOCATION="us-central1"

BIGQUERY_LOCATION="us-central1"
BIGQUERY_DATASET="test_hybridsearch"
BIGQUERY_TABLE="hybrid_test"   # optional, auto-generated if not set
```

#### 6. Run Integration Tests

```bash
pytest tests/integration_tests/ -v -m integration
```

#### 7. Cleanup

The test fixture deletes the test table automatically after each run.
To remove the dataset entirely:

```bash
bq rm --dataset --force your-gcp-project-id:test_hybridsearch
```

## References

- [BigQuery SEARCH function](https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions#search)
- [BigQuery VECTOR_SEARCH function](https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions#vector_search)
- [BigQuery Search Indexes](https://cloud.google.com/bigquery/docs/search-index)
- [langchain-google-community](https://github.com/langchain-ai/langchain-google/tree/master/libs/community)
