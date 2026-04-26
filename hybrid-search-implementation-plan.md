# BigQuery Hybrid Search 구현 계획

`langchain-google-community`의 `BigQueryVectorStore`에 Hybrid Search 기능을 추가하기 위한 구현 계획서.

## 1. 배경 분석

### 1.1 현재 아키텍처

```
BaseBigQueryVectorStore (_base.py)
  └── BigQueryVectorStore (bigquery.py)        ← VECTOR_SEARCH() only
  └── VertexFSVectorStore (featurestore.py)    ← Vertex AI Feature Store
```

**현재 검색 흐름:**

```
similarity_search(query)
  → embedding.embed_query(query)
  → similarity_search_by_vectors([embedding])
  → _similarity_search_by_vectors_with_scores_and_embeddings()
  → _search_embeddings()
  → _create_search_query()  ← VECTOR_SEARCH() SQL 생성
```

**핵심 SQL (`_create_search_query()` 생성 결과):**

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

### 1.2 BigQuery Search Functions

> 참고: https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions

| 함수 | 용도 | 반환 타입 |
|------|------|----------|
| `SEARCH()` | 토큰 기반 Full-text 검색 | `BOOL` |
| `VECTOR_SEARCH()` | 임베딩 기반 시맨틱 유사도 검색 | Table (`base`, `query`, `distance`) |

**`SEARCH()` 함수 시그니처:**

```sql
SEARCH(
  data_to_search,
  search_query
  [, json_scope   => { 'JSON_VALUES' | 'JSON_KEYS' | 'JSON_KEYS_AND_VALUES' } ]
  [, analyzer     => { 'LOG_ANALYZER' | 'NO_OP_ANALYZER' | 'PATTERN_ANALYZER' } ]
  [, analyzer_options => analyzer_options_values ]
)
```

**주요 특성:**

- `SEARCH()`는 Boolean 반환 (점수 없음) — 모든 검색 토큰이 데이터에 포함되면 `TRUE`
- Search Index가 있으면 성능이 크게 향상됨
- `LOG_ANALYZER` (기본값): 구분자 기준으로 토큰 분리 후 정규화
- `NO_OP_ANALYZER`: 전체 텍스트를 단일 토큰으로 처리 (정확 매칭)
- `PATTERN_ANALYZER`: 정규식 패턴으로 토큰 분리

**`SEARCH()` 쿼리 문법:**

```sql
-- 묵시적 AND (공백)
SEARCH(content, 'foo bar')          -- foo AND bar

-- 명시적 논리 연산자
SEARCH(content, 'foo AND bar')
SEARCH(content, 'foo OR bar')
SEARCH(content, 'foo AND (bar OR baz)')

-- 구문 검색 (인접 + 순서 보장)
SEARCH(content, '"foo bar"')

-- 정확 매칭 (backtick)
SEARCH(content, '`foobar@example.com`')
```

## 2. Hybrid Search 전략

### 2.1 지원할 모드

| 모드 | 설명 | SQL 구조 | 적합 케이스 |
|------|------|----------|-------------|
| **`pre_filter`** | SEARCH()로 후보군 축소 후 VECTOR_SEARCH()로 랭킹 | 단일 쿼리: `VECTOR_SEARCH((... WHERE SEARCH(...)), ...)` | 특정 키워드를 반드시 포함하는 결과가 필요할 때 |
| **`rrf`** | Vector Search + Text Search 독립 실행 후 RRF로 통합 랭킹 | 단일 쿼리: CTE 2개 + FULL OUTER JOIN + RRF score | 키워드 관련성과 의미적 유사도 간 균형이 필요할 때 |

### 2.2 Pre-filter 모드 상세

```
[User Query] ──→ embed_query() ──→ query_embedding
     │
     └──→ SEARCH(content, query) ──→ 후보 필터링
                                         │
                                         ▼
                                   VECTOR_SEARCH(filtered_candidates, query_embedding)
                                         │
                                         ▼
                                   Top-K results (ranked by vector distance)
```

**장점:** 단순한 구현, 단일 SQL 쿼리, Search Index 활용 가능
**단점:** 키워드가 반드시 매칭되어야 해서 결과가 제한적일 수 있음

### 2.3 RRF (Reciprocal Rank Fusion) 모드 상세

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

**장점:** 두 신호를 균형 있게 결합, 한쪽에만 매칭되는 문서도 포함 가능
**단점:** 복잡한 SQL, `SEARCH()`가 점수를 반환하지 않아 text_rank는 순서 기반

**RRF 공식:**

```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```

- `k`: 상수 (기본 60), 상위 랭킹의 영향력 조절
- `rank_i(doc)`: i번째 소스에서의 문서 순위 (1-based)

## 3. 클래스 설계

### 3.1 클래스 다이어그램

```
BaseBigQueryVectorStore (_base.py)
  └── BigQueryVectorStore (bigquery.py)
        └── BigQueryHybridSearchVectorStore (bigquery.py)   ← NEW
```

### 3.2 새 클래스 필드

```python
class BigQueryHybridSearchVectorStore(BigQueryVectorStore):

    search_fields: List[str] = []
    """SEARCH() 적용 대상 컬럼 목록. 비어있으면 [content_field] 사용."""

    search_analyzer: Literal["LOG_ANALYZER", "NO_OP_ANALYZER", "PATTERN_ANALYZER"] = "LOG_ANALYZER"
    """SEARCH() 텍스트 분석기."""

    search_analyzer_options: Optional[str] = None
    """SEARCH() 분석기 옵션 (JSON string)."""

    hybrid_search_mode: Literal["pre_filter", "rrf"] = "pre_filter"
    """기본 하이브리드 검색 모드."""

    rrf_k: int = 60
    """RRF 상수. 값이 클수록 상위 순위와 하위 순위의 점수 차이가 줄어듦."""
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
    """키워드 검색과 벡터 검색을 결합한 하이브리드 검색.

    Args:
        query: 검색 쿼리 (임베딩 생성 + 키워드 검색에 모두 사용).
        k: 반환할 문서 수.
        fetch_k: RRF 모드에서 각 소스별 후보 수.
        text_query: 별도 키워드 쿼리. None이면 query를 사용.
        filter: 메타데이터 필터 (dict 또는 SQL WHERE 절).
        hybrid_search_mode: 검색 모드 오버라이드. None이면 인스턴스 기본값 사용.

    Returns:
        관련성 순으로 정렬된 Document 리스트.
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
    """hybrid_search와 동일하되, 각 문서의 점수를 함께 반환."""
    ...
```

## 4. SQL 생성 상세

### 4.1 Pre-filter 모드 SQL

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

**파라미터화:**

```python
query_parameters = [
    bigquery.ArrayQueryParameter("emb_0", "FLOAT64", embedding),
    bigquery.ScalarQueryParameter("text_query", "STRING", text_query),
]
```

### 4.2 RRF 모드 SQL

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

### 4.3 SEARCH() 대상 컬럼 지정

```python
# search_fields가 1개일 때
search_fields_sql = f"{search_fields[0]}"

# search_fields가 여러 개일 때 (tuple로 전달)
search_fields_sql = f"({', '.join(search_fields)})"

# 예시: SEARCH((title, content, summary), @text_query)
```

## 5. Search Index 관리

### 5.1 Search Index 생성

```sql
CREATE SEARCH INDEX IF NOT EXISTS `{table_name}_search_index`
ON `{full_table_id}` ({search_fields})
OPTIONS (analyzer = '{search_analyzer}')
```

### 5.2 Search Index 확인

```sql
SELECT index_name, index_status
FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.SEARCH_INDEXES`
WHERE table_name = '{table_name}'
```

### 5.3 구현 패턴

기존 `initialize_bq_vector_index`와 동일한 패턴 적용:

```python
@model_validator(mode="after")
def _initialize_search_index(self) -> Self:
    """content 필드에 Search Index 생성 (존재하지 않는 경우)."""
    # 1. INFORMATION_SCHEMA.SEARCH_INDEXES 조회
    # 2. 인덱스 없으면 백그라운드 스레드에서 CREATE SEARCH INDEX 실행
    # 3. analyzer 설정이 기존 인덱스와 일치하는지 확인
    ...
```

## 6. 파일 구조

### 6.1 변경 파일

```
langchain_google_community/bq_storage_vectorstores/
├── __init__.py              # BigQueryHybridSearchVectorStore export 추가
├── _base.py                 # 변경 없음
├── bigquery.py              # BigQueryHybridSearchVectorStore 클래스 추가
├── featurestore.py          # 변경 없음
└── utils.py                 # 변경 없음
```

### 6.2 새로 추가되는 요소

| 위치 | 요소 | 설명 |
|------|------|------|
| `bigquery.py` | `BigQueryHybridSearchVectorStore` | 새 클래스 |
| `bigquery.py` | `_create_search_index()` | Search Index 생성 헬퍼 함수 |
| `bigquery.py` | `_create_hybrid_search_query()` | Hybrid SQL 생성 (pre_filter/rrf 분기) |

## 7. 구현 우선순위

| 우선순위 | 항목 | 설명 | 난이도 |
|----------|------|------|--------|
| **P0** | Pre-filter 모드 | `VECTOR_SEARCH`의 `base_table_query`에 `SEARCH()` 추가 | 낮음 |
| **P1** | Search Index 자동 생성 | `INFORMATION_SCHEMA` 조회 + `CREATE SEARCH INDEX` | 낮음 |
| **P2** | RRF 모드 | CTE + FULL OUTER JOIN + RRF score 계산 SQL | 중간 |
| **P3** | `batch_hybrid_search()` | 기존 `batch_search()`의 hybrid 버전 (임시 테이블 활용) | 중간 |
| **P4** | Weighted RRF | vector/text 가중치 조절 (`alpha` 파라미터) | 낮음 |

## 8. 사용 예시

### 8.1 Pre-filter 모드

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

# query를 키워드 + 임베딩 양쪽에 사용
results = store.hybrid_search(
    query="BigQuery vector search performance tuning",
    k=10,
)

# 별도의 키워드 쿼리 지정
results = store.hybrid_search(
    query="How to optimize search performance?",     # 임베딩용
    text_query="BigQuery VECTOR_SEARCH index",        # 키워드용
    k=10,
)
```

### 8.2 RRF 모드

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
    fetch_k=50,  # 각 소스에서 50개 후보 추출
)

for doc, score in results:
    print(f"[{score:.4f}] {doc.page_content[:80]}...")
```

### 8.3 메타데이터 필터 결합

```python
results = store.hybrid_search(
    query="authentication security best practices",
    text_query="OAuth2 JWT token",
    k=5,
    filter={"category": "security", "language": "en"},
)
```

### 8.4 기존 similarity_search와 호환

```python
# 기존 벡터 검색은 그대로 동작
results = store.similarity_search("What is BigQuery?", k=5)

# hybrid_search는 추가 메서드로 제공
results = store.hybrid_search("What is BigQuery?", k=5)
```

## 9. 테스트 계획

### 9.1 단위 테스트

- [ ] `_create_hybrid_search_query()` — pre_filter 모드 SQL 생성 검증
- [ ] `_create_hybrid_search_query()` — rrf 모드 SQL 생성 검증
- [ ] `_create_filters()` — 기존 필터 + SEARCH() 조합 검증
- [ ] `search_fields` 기본값 (content_field) 동작 확인
- [ ] `search_fields` 복수 컬럼 지정 시 SQL 구조 확인

### 9.2 통합 테스트

- [ ] BigQuery 테이블에 데이터 삽입 후 pre_filter 모드 검색
- [ ] BigQuery 테이블에 데이터 삽입 후 rrf 모드 검색
- [ ] Search Index 자동 생성 확인
- [ ] Vector Index + Search Index 동시 존재 시 성능 확인
- [ ] 메타데이터 필터와 하이브리드 검색 결합 테스트

### 9.3 엣지 케이스

- [ ] SEARCH()에 매칭되는 문서가 0건인 경우 (pre_filter)
- [ ] VECTOR_SEARCH()에 매칭되는 문서가 0건인 경우 (rrf)
- [ ] `text_query`에 특수문자/예약어 포함 시 이스케이프 처리
- [ ] 빈 테이블에 대한 검색
- [ ] `k` > 실제 결과 수인 경우

---

## 10. Embedding Task Types 지원

> 참고: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types

### 10.1 배경

Vertex AI / Google Generative AI 임베딩 모델은 8가지 task type을 지원하며, 동일한 텍스트라도 task type에 따라 다른 임베딩 벡터가 생성된다. 검색 시나리오에서는 query/document 측의 task type을 적절히 지정해야 검색 품질이 향상된다.

| Task Type | Query 측 | Document 측 |
|-----------|----------|-------------|
| `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` | 쿼리 | 문서 (기본값) |
| `QUESTION_ANSWERING` | 양쪽 동일 | 양쪽 동일 |
| `FACT_VERIFICATION` | 양쪽 동일 | 양쪽 동일 |
| `SEMANTIC_SIMILARITY` | 양쪽 동일 | 양쪽 동일 |
| `CODE_RETRIEVAL_QUERY` | 쿼리 | `RETRIEVAL_DOCUMENT` 유지 |
| `CLASSIFICATION` / `CLUSTERING` | 검색 시나리오 외 | — |

→ **query 측과 document 측 task type을 독립적으로 설정**할 수 있어야 함.

### 10.2 임베딩 라이브러리 호환성 이슈

| 라이브러리 | 파라미터명 | 기본값 (query / docs) | 상태 |
|------------|-----------|-----------------------|------|
| `langchain_google_genai.GoogleGenerativeAIEmbeddings` | `task_type` | `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` | **권장** |
| `langchain_google_vertexai.VertexAIEmbeddings` | `embeddings_task_type` | `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` | deprecated |
| 기타 임베딩 (OpenAI 등) | 없음 | — | task_type 개념 없음 |

→ 단순히 kwarg를 전달하면 비-Google 임베딩에서 `TypeError` 발생.
→ 메서드 시그니처를 검사하여 적절한 kwarg 이름으로 자동 라우팅 필요.

### 10.3 영향 받는 임베딩 호출 지점

| 위치 | 호출 | 변경 필요 |
|------|------|-----------|
| `vectorstore.py:227` (`_hybrid_search_with_scores`) | `embed_query(query)` | ✅ Yes — query_task_type 적용 |
| `_base.py:272` (부모 `add_texts`) | `embed_documents(texts)` | ✅ Yes — `add_texts` 오버라이드 필요 |
| `_base.py:153` (차원 추정 `embed_query("test")`) | `embed_query("test")` | ❌ No — 차원은 task_type 무관 |

### 10.4 새 필드 (`BigQueryHybridSearchVectorStore`)

```python
query_task_type: Optional[str] = None
"""쿼리 임베딩에 적용할 task type. None이면 임베딩 모델의 기본값 사용
   (Google 임베딩의 경우 'RETRIEVAL_QUERY')."""

document_task_type: Optional[str] = None
"""문서 임베딩에 적용할 task type. None이면 임베딩 모델의 기본값 사용
   (Google 임베딩의 경우 'RETRIEVAL_DOCUMENT')."""
```

**설계 원칙:** 둘 다 `None`이면 기존 동작과 100% 동일 → backward compatible.

### 10.5 Task Type 주입 헬퍼

라이브러리별 kwarg 명을 자동 탐지하는 모듈 함수:

```python
import inspect
from functools import lru_cache

@lru_cache(maxsize=64)
def _resolve_task_type_kwarg(qualname: str, params: tuple[str, ...]) -> Optional[str]:
    """method 시그니처에서 task_type 관련 kwarg 이름을 찾아 반환."""
    if "task_type" in params:
        return "task_type"
    if "embeddings_task_type" in params:
        return "embeddings_task_type"
    return None


def _call_with_task_type(
    method: Callable,            # embed_query 또는 embed_documents
    arg: Union[str, List[str]],  # text 또는 texts
    task_type: Optional[str],
) -> Any:
    """task_type이 None이면 그대로 호출. 아니면 method 시그니처를 검사하여
       'task_type' 또는 'embeddings_task_type' kwarg로 전달.
       어느 쪽도 받지 않으면 한 번만 경고 로그 후 task_type 없이 호출."""
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

**설계 결정:**
- `lru_cache`로 시그니처 검사 결과 캐싱 (메서드당 1회만 introspection)
- 비-Google 임베딩 + task_type 설정 시 경고 후 무시 (예외 발생 안 함)

### 10.6 코드 변경 위치

#### Query 측: `_hybrid_search_with_scores` 수정

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

#### Document 측: `add_texts` 오버라이드

부모 `BaseBigQueryVectorStore.add_texts`는 `embed_documents()`를 직접 호출하므로 파라미터 주입 지점이 없음. 자식 클래스에서 오버라이드하여 임베딩을 먼저 계산한 뒤 `add_texts_with_embeddings()`로 위임:

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

#### Public API에 per-call override 추가

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

# hybrid_search_with_score()도 동일하게 추가
```

#### Retriever 경로

`BigQueryHybridSearchRetriever`는 변경 불필요. `search_kwargs`에 `query_task_type`을 넣으면 `**merged`로 자동 전달됨:

```python
retriever = store.as_retriever(
    search_type="hybrid",
    search_kwargs={"k": 4, "query_task_type": "CODE_RETRIEVAL_QUERY"},
)
```

### 10.7 엣지 케이스

| 상황 | 처리 방안 |
|------|-----------|
| Query/Document task type을 다르게 설정 (예: `CODE_RETRIEVAL_QUERY` + `RETRIEVAL_DOCUMENT`) | 정상 동작 (Vertex 공식 권장 조합) |
| 차원 추정 호출 (`embed_query("test")`) | 변경하지 않음. 차원은 task type과 무관 (모델 dimensions만 영향) |
| 비-Google 임베딩 + task_type 설정 | 경고 로그 후 task_type 무시하고 정상 진행 |
| `LangChain Embeddings` 표준 인터페이스 위배 우려 | `add_texts()`는 `VectorStore` 인터페이스에 정의된 메서드. `document_task_type`은 keyword-only로만 추가하여 표준 호출 호환 유지 |
| 인스턴스 필드와 per-call kwarg 충돌 | per-call kwarg가 우선 |
| 임베딩 모델 변경 (예: `text-embedding-004` → `gemini-embedding-001`) | 사용자가 task_type 문자열만 변경하면 됨 |

### 10.8 사용 예시

#### A. 인스턴스 레벨 설정 (모든 검색에 동일 적용)

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

#### B. 코드 검색 시나리오 (query만 다르게)

```python
store = BigQueryHybridSearchVectorStore(
    ...,
    query_task_type="CODE_RETRIEVAL_QUERY",
    # document_task_type=None → RETRIEVAL_DOCUMENT 기본값 유지
)
```

#### C. Per-call override (인스턴스 기본값 일시 덮어쓰기)

```python
results = store.hybrid_search(
    query="How does VECTOR_SEARCH work?",
    query_task_type="QUESTION_ANSWERING",
    k=10,
)
```

#### D. Retriever 경로

```python
retriever = store.as_retriever(
    search_type="hybrid",
    search_kwargs={"k": 4, "query_task_type": "FACT_VERIFICATION"},
)
```

#### E. 문서 인덱싱 시 per-call override

```python
store.add_texts(
    texts=["def foo(): ...", "class Bar: ..."],
    document_task_type="CODE_RETRIEVAL_QUERY",  # 코드 베이스 인덱싱
)
```

### 10.9 변경 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/langchain_bigquery_hybridsearch/vectorstore.py` | 필드 2개, 헬퍼 1개, `add_texts` 오버라이드, `_hybrid_search_with_scores` 수정, public API에 per-call kwarg 추가 |
| `tests/unit_tests/test_sql_generation.py` | 영향 없음 (SQL 생성 로직 미변경) |
| `tests/unit_tests/test_task_type.py` (신규) | `_call_with_task_type` 헬퍼 단위 테스트 |
| `tests/integration_tests/test_hybridsearch.py` | task_type 사용한 hybrid_search 통합 테스트 추가 |

### 10.10 테스트 계획

#### 단위 테스트 (`_call_with_task_type` mock 기반)

- [ ] `task_type=None` → 원본 메서드를 그대로 호출
- [ ] 메서드 시그니처에 `task_type` 있음 → `task_type=...`로 전달
- [ ] 메서드 시그니처에 `embeddings_task_type` 있음 → `embeddings_task_type=...`로 전달
- [ ] 둘 다 없음 → 경고 로그 + task_type 무시
- [ ] 두 가지 fake 임베딩 클래스(`task_type` kwarg / `embeddings_task_type` kwarg)에 대해 `add_texts`와 `hybrid_search` 흐름 검증
- [ ] 인스턴스 필드와 per-call kwarg 충돌 시 per-call이 우선

#### 통합 테스트

- [ ] `GoogleGenerativeAIEmbeddings`로 `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT` 기본 동작
- [ ] 동일 데이터를 `SEMANTIC_SIMILARITY`로 인덱싱·검색 시 결과 분포 확인
- [ ] `CODE_RETRIEVAL_QUERY` 쿼리 + `RETRIEVAL_DOCUMENT` 문서 조합 검증

### 10.11 구현 우선순위

| 우선순위 | 항목 | 난이도 |
|----------|------|--------|
| **P0** | `_call_with_task_type` 헬퍼 + 단위 테스트 | 낮음 |
| **P0** | 필드 2개 추가 + `_hybrid_search_with_scores` 변경 | 낮음 |
| **P0** | `add_texts` 오버라이드 | 낮음 |
| **P1** | Public API per-call kwarg 추가 (`hybrid_search` / `_with_score`) | 낮음 |
| **P1** | README/문서에 사용 예시 추가 | 낮음 |
| **P2** | 통합 테스트 추가 | 중간 |
| **P3** (out of scope) | `RETRIEVAL_DOCUMENT`용 `title` 파라미터 per-document 전달 | 중간 |
