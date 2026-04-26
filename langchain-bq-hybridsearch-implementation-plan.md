# LangChain BigQuery Hybrid Search Implementation Plan

A design document for adding hybrid search to `langchain-google-community`'s `BigQueryVectorStore`.

## 1. Background

### 1.1 Current architecture

```
BaseBigQueryVectorStore (_base.py)
  └── BigQueryVectorStore (bigquery.py)        ← VECTOR_SEARCH() only
  └── VertexFSVectorStore (featurestore.py)    ← Vertex AI Feature Store
```

**Existing search flow:**

```
similarity_search(query)
  → embedding.embed_query(query)
  → similarity_search_by_vectors([embedding])
  → _similarity_search_by_vectors_with_scores_and_embeddings()
  → _search_embeddings()
  → _create_search_query()  ← builds the VECTOR_SEARCH() SQL
```

**The SQL produced by `_create_search_query()`:**

```sql
WITH embeddings AS (
  SELECT 0 AS row_num, @emb_0 AS embedding
  UNION ALL
  SELECT 1 AS row_num, @emb_1 AS embedding
)
SELECT base.*, query.row_num, distance AS score
FROM VECTOR_SEARCH(
  (SELECT * FROM `project.dataset.table` WHERE {filter}),
  "embedding",
  (SELECT row_num, embedding FROM embeddings),
  distance_type => "COSINE",
  top_k => 5,
  options => '{}'
)
ORDER BY row_num, score
```

### 1.2 BigQuery search functions

> Reference: https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions

| Function | Purpose | Return type |
|----------|---------|-------------|
| `SEARCH()` | Token-based full-text search | `BOOL` |
| `VECTOR_SEARCH()` | Embedding-based semantic similarity | Table (`base`, `query`, `distance`) |

**`SEARCH()` signature:**

```sql
SEARCH(
  data_to_search,
  search_query
  [, json_scope   => { 'JSON_VALUES' | 'JSON_KEYS' | 'JSON_KEYS_AND_VALUES' } ]
  [, analyzer     => { 'LOG_ANALYZER' | 'NO_OP_ANALYZER' | 'PATTERN_ANALYZER' } ]
  [, analyzer_options => analyzer_options_values ]
)
```

**Key characteristics:**

- `SEARCH()` returns a boolean with no score — `TRUE` when every search token is present in the data.
- A search index dramatically improves performance.
- `LOG_ANALYZER` (default): splits on delimiters and normalizes the resulting tokens.
- `NO_OP_ANALYZER`: treats the entire input as a single token (exact match).
- `PATTERN_ANALYZER`: splits on a regex pattern.

**`SEARCH()` query syntax:**

```sql
-- Implicit AND (whitespace)
SEARCH(content, 'foo bar')          -- foo AND bar

-- Explicit boolean operators
SEARCH(content, 'foo AND bar')
SEARCH(content, 'foo OR bar')
SEARCH(content, 'foo AND (bar OR baz)')

-- Phrase search (adjacent + ordered)
SEARCH(content, '"foo bar"')

-- Exact match (backtick)
SEARCH(content, '`foobar@example.com`')
```

## 2. Hybrid search strategy

### 2.1 Supported modes

| Mode | Description | SQL shape | Best for |
|------|-------------|-----------|----------|
| **`pre_filter`** | Narrow the candidate set with `SEARCH()`, then rank with `VECTOR_SEARCH()` | Single query: `VECTOR_SEARCH((... WHERE SEARCH(...)), ...)` | Results that *must* contain specific keywords |
| **`rrf`** | Run vector and text search independently, then merge with Reciprocal Rank Fusion | Single query: 2 CTEs + `FULL OUTER JOIN` + RRF score | Balancing keyword relevance with semantic similarity |

### 2.2 Pre-filter mode in detail

```
[User Query] ──→ embed_query() ──→ query_embedding
     │
     └──→ SEARCH(content, query) ──→ candidate filtering
                                         │
                                         ▼
                                   VECTOR_SEARCH(filtered_candidates, query_embedding)
                                         │
                                         ▼
                                   Top-K results (ranked by vector distance)
```

**Pros:** simple implementation, single SQL query, can take advantage of the search index.
**Cons:** keywords must match, which may yield very few results.

### 2.3 RRF (Reciprocal Rank Fusion) mode in detail

```
[User Query] ──→ embed_query() ──→ VECTOR_SEARCH() ──→ vector_rank
     │
     └──→ SEARCH(content, query) ──→ text_rank
                                         │
                          ┌──────────────┘
                          ▼
              FULL OUTER JOIN on doc_id
                          │
                          ▼
              RRF Score = 1/(k + vector_rank) + 1/(k + text_rank)
                          │
                          ▼
              Top-K results (ranked by RRF score)
```

**Pros:** balances both signals, includes documents that match only one side.
**Cons:** more complex SQL, and because `SEARCH()` doesn't return a score, `text_rank` is purely ordinal.

**RRF formula:**

```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```

- `k`: a constant (default 60) that tunes how much top ranks dominate.
- `rank_i(doc)`: the document's rank in source `i` (1-based).

## 3. Class design

### 3.1 Class diagram

```
BaseBigQueryVectorStore (_base.py)
  └── BigQueryVectorStore (bigquery.py)
        └── BigQueryHybridSearchVectorStore (bigquery.py)   ← NEW
```

### 3.2 New class fields

```python
class BigQueryHybridSearchVectorStore(BigQueryVectorStore):

    search_fields: List[str] = []
    """Columns to apply SEARCH() on. Defaults to [content_field] when empty."""

    search_analyzer: Literal["LOG_ANALYZER", "NO_OP_ANALYZER", "PATTERN_ANALYZER"] = "LOG_ANALYZER"
    """Text analyzer for SEARCH()."""

    search_analyzer_options: Optional[str] = None
    """Analyzer options for SEARCH() (JSON string)."""

    hybrid_search_mode: Literal["pre_filter", "rrf"] = "pre_filter"
    """Default hybrid search mode."""

    rrf_k: int = 60
    """RRF constant. Higher values flatten the gap between top and lower ranks."""
```

### 3.3 Public API

```python
def hybrid_search(
    self,
    query: str,
    k: int = 5,
    fetch_k: int = 25,
    text_query: Optional[str] = None,
    filter: Optional[Union[Dict[str, Any], str]] = None,
    hybrid_search_mode: Optional[Literal["pre_filter", "rrf"]] = None,
    **kwargs: Any,
) -> List[Document]:
    """Hybrid search combining keyword and vector similarity.

    Args:
        query: The search query, used for both embedding generation and
            keyword search (unless text_query is supplied separately).
        k: Number of documents to return.
        fetch_k: Number of candidates per source in RRF mode.
        text_query: Separate keyword query. Falls back to query when None.
        filter: Metadata filter (dict or SQL WHERE clause).
        hybrid_search_mode: Override the instance default.

    Returns:
        Documents ranked by the chosen hybrid strategy.
    """
    ...


def hybrid_search_with_score(
    self,
    query: str,
    k: int = 5,
    fetch_k: int = 25,
    text_query: Optional[str] = None,
    filter: Optional[Union[Dict[str, Any], str]] = None,
    hybrid_search_mode: Optional[Literal["pre_filter", "rrf"]] = None,
    **kwargs: Any,
) -> List[Tuple[Document, float]]:
    """Same as hybrid_search but also returns each document's score."""
    ...
```

## 4. SQL generation details

### 4.1 Pre-filter mode SQL

```sql
WITH embeddings AS (
  SELECT 0 AS row_num, @emb_0 AS embedding
)
SELECT base.*, query.row_num, distance AS score
FROM VECTOR_SEARCH(
  (SELECT *
   FROM `{full_table_id}`
   WHERE SEARCH(
       ({search_fields}),
       @text_query,
       analyzer => '{search_analyzer}'
       {', analyzer_options => @analyzer_options' if analyzer_options else ''}
     )
     AND {metadata_filter}
  ),
  "{embedding_field}",
  (SELECT row_num, {embedding_field} FROM embeddings),
  distance_type => "{distance_type}",
  top_k => {k},
  options => '{options_json}'
)
ORDER BY row_num, score
```

**Parameterization:**

```python
query_parameters = [
    bigquery.ArrayQueryParameter("emb_0", "FLOAT64", embedding),
    bigquery.ScalarQueryParameter("text_query", "STRING", text_query),
]
```

### 4.2 RRF mode SQL

```sql
WITH vector_results AS (
  SELECT
    base.{doc_id_field},
    base.{content_field},
    base.{embedding_field},
    {extra_field_selects},
    distance,
    ROW_NUMBER() OVER (ORDER BY distance ASC) AS vector_rank
  FROM VECTOR_SEARCH(
    (SELECT * FROM `{full_table_id}` WHERE {metadata_filter}),
    "{embedding_field}",
    query_value => @query_embedding,
    top_k => {fetch_k},
    distance_type => "{distance_type}"
  )
),
text_results AS (
  SELECT
    {doc_id_field},
    {content_field},
    {embedding_field},
    {extra_field_selects},
    ROW_NUMBER() OVER () AS text_rank
  FROM `{full_table_id}`
  WHERE SEARCH(
      ({search_fields}),
      @text_query,
      analyzer => '{search_analyzer}'
    )
    AND {metadata_filter}
  LIMIT {fetch_k}
),
combined AS (
  SELECT
    COALESCE(v.{doc_id_field}, t.{doc_id_field}) AS {doc_id_field},
    COALESCE(v.{content_field}, t.{content_field}) AS {content_field},
    COALESCE(v.{embedding_field}, t.{embedding_field}) AS {embedding_field},
    {coalesce_extra_fields},
    v.vector_rank,
    t.text_rank,
    v.distance,
    IFNULL(1.0 / ({rrf_k} + v.vector_rank), 0)
      + IFNULL(1.0 / ({rrf_k} + t.text_rank), 0) AS rrf_score
  FROM vector_results v
  FULL OUTER JOIN text_results t
    ON v.{doc_id_field} = t.{doc_id_field}
)
SELECT *
FROM combined
ORDER BY rrf_score DESC
LIMIT {k}
```

### 4.3 SEARCH() target columns

```python
# Single column
search_fields_sql = f"{search_fields[0]}"

# Multiple columns (passed as a tuple)
search_fields_sql = f"({', '.join(search_fields)})"

# Example: SEARCH((title, content, summary), @text_query)
```

## 5. Search index management

### 5.1 Creating the search index

```sql
CREATE SEARCH INDEX IF NOT EXISTS `{table_name}_search_index`
ON `{full_table_id}` ({search_fields})
OPTIONS (analyzer = '{search_analyzer}')
```

### 5.2 Checking the search index

```sql
SELECT index_name, index_status
FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.SEARCH_INDEXES`
WHERE table_name = '{table_name}'
```

### 5.3 Implementation pattern

Mirrors the existing `initialize_bq_vector_index` pattern:

```python
@model_validator(mode="after")
def _initialize_search_index(self) -> Self:
    """Create a search index on the content field if one does not exist."""
    # 1. Query INFORMATION_SCHEMA.SEARCH_INDEXES
    # 2. If the index is missing, run CREATE SEARCH INDEX in a background thread
    # 3. Verify the analyzer settings match the existing index
    ...
```

## 6. File layout

### 6.1 Files touched

```
langchain_google_community/bq_storage_vectorstores/
├── __init__.py              # export BigQueryHybridSearchVectorStore
├── _base.py                 # unchanged
├── bigquery.py              # add BigQueryHybridSearchVectorStore class
├── featurestore.py          # unchanged
└── utils.py                 # unchanged
```

### 6.2 New elements

| Location | Element | Description |
|----------|---------|-------------|
| `bigquery.py` | `BigQueryHybridSearchVectorStore` | The new class |
| `bigquery.py` | `_create_search_index()` | Helper for creating the search index |
| `bigquery.py` | `_create_hybrid_search_query()` | Builds the hybrid SQL (branches on pre_filter/rrf) |

## 7. Implementation priorities

| Priority | Item | Description | Difficulty |
|----------|------|-------------|------------|
| **P0** | Pre-filter mode | Add `SEARCH()` to `VECTOR_SEARCH`'s `base_table_query` | Low |
| **P1** | Auto search index creation | `INFORMATION_SCHEMA` query + `CREATE SEARCH INDEX` | Low |
| **P2** | RRF mode | CTE + FULL OUTER JOIN + RRF score SQL | Medium |
| **P3** | `batch_hybrid_search()` | Hybrid version of the existing `batch_search()` (uses a temp table) | Medium |
| **P4** | Weighted RRF | Tune the vector/text balance (`alpha` parameter) | Low |

## 8. Usage examples

### 8.1 Pre-filter mode

```python
from langchain_google_community.bq_storage_vectorstores.bigquery import (
    BigQueryHybridSearchVectorStore,
)

store = BigQueryHybridSearchVectorStore(
    project_id="my-project",
    dataset_name="my_dataset",
    table_name="documents",
    location="US",
    embedding=embedding_model,
    search_analyzer="LOG_ANALYZER",
    hybrid_search_mode="pre_filter",
)

# Use the same query for both keyword and embedding search
results = store.hybrid_search(
    query="BigQuery vector search performance tuning",
    k=10,
)

# Or specify a separate keyword query
results = store.hybrid_search(
    query="How to optimize search performance?",     # used for the embedding
    text_query="BigQuery VECTOR_SEARCH index",        # used for keyword search
    k=10,
)
```

### 8.2 RRF mode

```python
store = BigQueryHybridSearchVectorStore(
    project_id="my-project",
    dataset_name="my_dataset",
    table_name="documents",
    location="US",
    embedding=embedding_model,
    hybrid_search_mode="rrf",
    rrf_k=60,
)

results = store.hybrid_search_with_score(
    query="machine learning model deployment",
    k=10,
    fetch_k=50,  # 50 candidates per source
)

for doc, score in results:
    print(f"[{score:.4f}] {doc.page_content[:80]}...")
```

### 8.3 Combined with metadata filters

```python
results = store.hybrid_search(
    query="authentication security best practices",
    text_query="OAuth2 JWT token",
    k=5,
    filter={"category": "security", "language": "en"},
)
```

### 8.4 Backwards compatibility with similarity_search

```python
# The inherited vector search keeps working
results = store.similarity_search("What is BigQuery?", k=5)

# hybrid_search is offered as an additional method
results = store.hybrid_search("What is BigQuery?", k=5)
```

## 9. Test plan

### 9.1 Unit tests

- [ ] `_create_hybrid_search_query()` — verify SQL generation in pre_filter mode
- [ ] `_create_hybrid_search_query()` — verify SQL generation in rrf mode
- [ ] `_create_filters()` — combine existing filters with `SEARCH()`
- [ ] `search_fields` — verify the default (`content_field`) behavior
- [ ] `search_fields` — verify SQL shape with multiple columns

### 9.2 Integration tests

- [ ] Insert data into a BigQuery table and run pre_filter mode search
- [ ] Insert data into a BigQuery table and run rrf mode search
- [ ] Verify auto search index creation
- [ ] Confirm performance when both vector and search indexes exist
- [ ] Combine metadata filters with hybrid search

### 9.3 Edge cases

- [ ] Pre-filter mode with zero `SEARCH()` matches
- [ ] RRF mode with zero `VECTOR_SEARCH()` matches
- [ ] Special characters / reserved words in `text_query` (escaping)
- [ ] Search against an empty table
- [ ] `k` greater than the actual result count

---

## 10. Embedding task type support

> Reference: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types

### 10.1 Background

Vertex AI / Google Generative AI embedding models accept one of eight task type hints, and the same input text produces different embedding vectors depending on which type is used. In retrieval scenarios, picking the right task type on each side noticeably improves search quality.

| Task Type | Query side | Document side |
|-----------|------------|---------------|
| `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` | query | document (default) |
| `QUESTION_ANSWERING` | same on both | same on both |
| `FACT_VERIFICATION` | same on both | same on both |
| `SEMANTIC_SIMILARITY` | same on both | same on both |
| `CODE_RETRIEVAL_QUERY` | query | keep `RETRIEVAL_DOCUMENT` |
| `CLASSIFICATION` / `CLUSTERING` | not for retrieval | — |

→ The query and document task types must be **independently configurable**.

### 10.2 Embedding library compatibility

| Library | Parameter name | Default (query / docs) | Status |
|---------|----------------|------------------------|--------|
| `langchain_google_genai.GoogleGenerativeAIEmbeddings` | `task_type` | `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` | **recommended** |
| `langchain_google_vertexai.VertexAIEmbeddings` | `embeddings_task_type` | `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` | deprecated |
| Other embeddings (OpenAI, etc.) | none | — | no task type concept |

→ Passing the kwarg blindly raises `TypeError` on non-Google embeddings.
→ The method signature has to be inspected so the right kwarg name can be chosen automatically.

### 10.3 Affected embedding call sites

| Location | Call | Change required? |
|----------|------|------------------|
| `vectorstore.py:227` (`_hybrid_search_with_scores`) | `embed_query(query)` | ✅ Yes — apply `query_task_type` |
| `_base.py:272` (parent `add_texts`) | `embed_documents(texts)` | ✅ Yes — override `add_texts` |
| `_base.py:153` (dimension probe `embed_query("test")`) | `embed_query("test")` | ❌ No — dimensionality is independent of task type |

### 10.4 New fields on `BigQueryHybridSearchVectorStore`

```python
query_task_type: Optional[str] = None
"""Task type for query embeddings. When None, the embedding model's own
   default applies ('RETRIEVAL_QUERY' for Google embeddings)."""

document_task_type: Optional[str] = None
"""Task type for document embeddings. When None, the embedding model's own
   default applies ('RETRIEVAL_DOCUMENT' for Google embeddings)."""
```

**Design principle:** when both are `None`, behavior is identical to today — fully backwards compatible.

### 10.5 Task type injection helper

A module-level function that auto-detects the kwarg name each library expects:

```python
import inspect
from functools import lru_cache

@lru_cache(maxsize=64)
def _resolve_task_type_kwarg(qualname: str, params: tuple[str, ...]) -> Optional[str]:
    """Return the kwarg name a method accepts for task type, or None."""
    if "task_type" in params:
        return "task_type"
    if "embeddings_task_type" in params:
        return "embeddings_task_type"
    return None


def _call_with_task_type(
    method: Callable,            # embed_query or embed_documents
    arg: Union[str, List[str]],  # text or texts
    task_type: Optional[str],
) -> Any:
    """When task_type is None, call the method untouched. Otherwise,
       inspect the method's signature and forward the value via either
       'task_type' or 'embeddings_task_type'. Embeddings without either
       kwarg get a single warning and run without the task type."""
    if task_type is None:
        return method(arg)
    try:
        sig = inspect.signature(method)
        kwarg = _resolve_task_type_kwarg(
            method.__qualname__, tuple(sig.parameters.keys())
        )
    except (TypeError, ValueError):
        kwarg = None

    if kwarg is None:
        logger.warning(
            "Embedding %s does not accept a task_type kwarg; ignoring %r",
            type(getattr(method, "__self__", None)).__name__, task_type,
        )
        return method(arg)
    return method(arg, **{kwarg: task_type})
```

**Design choices:**
- `lru_cache` memoizes the signature lookup so each method is introspected only once.
- Non-Google embeddings with a task type set: warn and continue, never raise.

### 10.6 Code change locations

#### Query side: modify `_hybrid_search_with_scores`

```python
# Before (vectorstore.py:227)
embedding = self.embedding.embed_query(query)

# After
effective_query_task_type = (
    kwargs.pop("query_task_type", None) or self.query_task_type
)
embedding = _call_with_task_type(
    self.embedding.embed_query, query, effective_query_task_type
)
```

#### Document side: override `add_texts`

The parent `BaseBigQueryVectorStore.add_texts` calls `embed_documents()` directly, leaving no place to inject a parameter. Override the method in the child class so the embeddings can be computed first and then handed to `add_texts_with_embeddings()`:

```python
@override
def add_texts(
    self,
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    *,
    document_task_type: Optional[str] = None,  # NEW per-call override
    **kwargs: Any,
) -> List[str]:
    effective = document_task_type or self.document_task_type
    embs = _call_with_task_type(
        self.embedding.embed_documents, list(texts), effective
    )
    return self.add_texts_with_embeddings(texts, embs, metadatas, **kwargs)
```

#### Add per-call override to the public API

```python
def hybrid_search(
    self,
    query: str,
    k: int = 5,
    fetch_k: int = 25,
    text_query: Optional[str] = None,
    filter: Optional[Union[Dict[str, Any], str]] = None,
    hybrid_search_mode: Optional[Literal["pre_filter", "rrf"]] = None,
    query_task_type: Optional[str] = None,  # NEW
    **kwargs: Any,
) -> List[Document]:
    ...

# hybrid_search_with_score() gets the same addition
```

#### Retriever path

`BigQueryHybridSearchRetriever` needs no changes. Putting `query_task_type` in `search_kwargs` flows through automatically via `**merged`:

```python
retriever = store.as_retriever(
    search_type="hybrid",
    search_kwargs={"k": 4, "query_task_type": "CODE_RETRIEVAL_QUERY"},
)
```

### 10.7 Edge cases

| Situation | Handling |
|-----------|----------|
| Query/document task types set differently (e.g. `CODE_RETRIEVAL_QUERY` + `RETRIEVAL_DOCUMENT`) | Works as expected — this is the officially recommended pairing for code search |
| Dimension probe call (`embed_query("test")`) | Left unchanged. Dimensionality depends only on the model's `dimensions`, not the task type |
| Non-Google embedding with a task type set | Logs a warning and proceeds without the task type |
| Concern about violating the `LangChain Embeddings` standard interface | `add_texts()` is part of the `VectorStore` interface; `document_task_type` is keyword-only, so positional-call compatibility is preserved |
| Conflict between instance field and per-call kwarg | The per-call kwarg wins |
| Switching embedding models (e.g. `text-embedding-004` → `gemini-embedding-001`) | Users only need to update the task type string |

### 10.8 Usage examples

#### A. Instance-level configuration (applies to every search)

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_bigquery_hybridsearch import BigQueryHybridSearchVectorStore

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

#### B. Code search (only the query side differs)

```python
store = BigQueryHybridSearchVectorStore(
    ...,
    query_task_type="CODE_RETRIEVAL_QUERY",
    # document_task_type=None → falls back to RETRIEVAL_DOCUMENT
)
```

#### C. Per-call override (temporarily override the instance default)

```python
results = store.hybrid_search(
    query="How does VECTOR_SEARCH work?",
    query_task_type="QUESTION_ANSWERING",
    k=10,
)
```

#### D. Retriever path

```python
retriever = store.as_retriever(
    search_type="hybrid",
    search_kwargs={"k": 4, "query_task_type": "FACT_VERIFICATION"},
)
```

#### E. Per-call override during ingestion

```python
store.add_texts(
    texts=["def foo(): ...", "class Bar: ..."],
    document_task_type="CODE_RETRIEVAL_QUERY",  # indexing a code base
)
```

### 10.9 Files changed

| File | Change |
|------|--------|
| `src/langchain_bigquery_hybridsearch/vectorstore.py` | 2 fields, 1 helper, `add_texts` override, modified `_hybrid_search_with_scores`, per-call kwarg on the public API |
| `tests/unit_tests/test_sql_generation.py` | Unaffected (SQL generation is unchanged) |
| `tests/unit_tests/test_task_type.py` (new) | Unit tests for the `_call_with_task_type` helper |
| `tests/integration_tests/test_hybridsearch.py` | New integration tests for `hybrid_search` with task types |

### 10.10 Test plan

#### Unit tests (mock-based for `_call_with_task_type`)

- [ ] `task_type=None` → call the original method untouched
- [ ] Method signature contains `task_type` → forward as `task_type=...`
- [ ] Method signature contains `embeddings_task_type` → forward as `embeddings_task_type=...`
- [ ] Neither present → log a warning and drop the task type
- [ ] Verify the `add_texts` and `hybrid_search` flows against two fake embeddings (one with `task_type`, one with `embeddings_task_type`)
- [ ] Per-call kwarg wins over the instance field on conflict

#### Integration tests

- [ ] Default `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` behavior with `GoogleGenerativeAIEmbeddings`
- [ ] Indexing and searching the same data with `SEMANTIC_SIMILARITY` produces a different result distribution
- [ ] `CODE_RETRIEVAL_QUERY` query + `RETRIEVAL_DOCUMENT` document combination

### 10.11 Implementation priorities

| Priority | Item | Difficulty |
|----------|------|------------|
| **P0** | `_call_with_task_type` helper + unit tests | Low |
| **P0** | Two new fields + `_hybrid_search_with_scores` change | Low |
| **P0** | `add_texts` override | Low |
| **P1** | Per-call kwarg on the public API (`hybrid_search` / `_with_score`) | Low |
| **P1** | Add usage examples to README/docs | Low |
| **P2** | Add integration tests | Medium |
| **P3** (out of scope) | Per-document `title` parameter for `RETRIEVAL_DOCUMENT` | Medium |
