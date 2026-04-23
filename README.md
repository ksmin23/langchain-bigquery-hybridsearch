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

All parameters from `BigQueryVectorStore` (`distance_type`, `extra_fields`, etc.) are also supported.

## Testing

### Unit Tests

No GCP credentials needed:

```bash
pytest tests/unit_tests/ -v
```

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
