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

```bash
# Unit tests (no GCP credentials needed)
pytest tests/unit_tests/ -v

# Integration tests (requires GCP credentials)
PROJECT_ID=your-gcp-project-id pytest tests/integration_tests/ -v -m integration
```

## References

- [BigQuery SEARCH function](https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions#search)
- [BigQuery VECTOR_SEARCH function](https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions#vector_search)
- [BigQuery Search Indexes](https://cloud.google.com/bigquery/docs/search-index)
- [langchain-google-community](https://github.com/langchain-ai/langchain-google/tree/master/libs/community)
