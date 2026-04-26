from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from threading import Lock, Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

from google.api_core.exceptions import ClientError
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_community.bq_storage_vectorstores.bigquery import (
    BigQueryVectorStore,
)
from pydantic import model_validator
from typing_extensions import Self, override

if TYPE_CHECKING:
    from google.cloud import bigquery as bq

logger = logging.getLogger(__name__)

_search_index_lock = Lock()
_SEARCH_INDEX_CHECK_INTERVAL = timedelta(seconds=60)

_TASK_TYPE_KWARG_CANDIDATES: Tuple[str, ...] = ("task_type", "embeddings_task_type")


@lru_cache(maxsize=128)
def _resolve_task_type_kwarg(
    qualname: str, params: Tuple[str, ...]
) -> Optional[str]:
    """Return the kwarg name an embedding method accepts for task type.

    Different LangChain Google embedding integrations use different parameter
    names: ``GoogleGenerativeAIEmbeddings`` uses ``task_type`` whereas
    ``VertexAIEmbeddings`` uses ``embeddings_task_type``. Cached per
    method qualname so introspection runs only once per embedding method.

    Returns ``None`` if neither kwarg is accepted.
    """
    for candidate in _TASK_TYPE_KWARG_CANDIDATES:
        if candidate in params:
            return candidate
    return None


def _call_with_task_type(
    method: Callable[..., Any],
    arg: Union[str, List[str]],
    task_type: Optional[str],
) -> Any:
    """Invoke an embedding ``embed_query``/``embed_documents`` with task type.

    When ``task_type`` is ``None`` the method is called as-is, preserving the
    embedding's own default. Otherwise, the method's signature is inspected
    to find a compatible kwarg (``task_type`` or ``embeddings_task_type``).
    Embeddings without such a kwarg get a single warning and run without it.
    """
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
        owner = getattr(method, "__self__", None)
        owner_name = type(owner).__name__ if owner is not None else "embedding"
        logger.warning(
            "%s.%s does not accept a task_type kwarg; ignoring task_type=%r",
            owner_name,
            getattr(method, "__name__", "embed"),
            task_type,
        )
        return method(arg)

    return method(arg, **{kwarg: task_type})


class BigQueryHybridSearchRetriever(VectorStoreRetriever):
    """Retriever that adds ``hybrid`` search type on top of the standard ones.

    Returned by :meth:`BigQueryHybridSearchVectorStore.as_retriever`.
    Supports all standard ``search_type`` values (``similarity``, ``mmr``,
    ``similarity_score_threshold``) **plus** ``"hybrid"``.

    When ``search_type="hybrid"``, the following ``search_kwargs`` are
    forwarded to :meth:`~BigQueryHybridSearchVectorStore.hybrid_search`:

    * ``k`` – number of results (default 4)
    * ``text_query`` – separate keyword query; falls back to the retriever
      query if not provided
    * ``fetch_k`` – candidate count per source in RRF mode (default 25)
    * ``filter`` – metadata filter
    * ``hybrid_search_mode`` – ``"pre_filter"`` or ``"rrf"``
    """

    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
        "hybrid",
    )

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        merged = {**self.search_kwargs, **kwargs}

        if self.search_type == "hybrid":
            store: BigQueryHybridSearchVectorStore = self.vectorstore  # type: ignore[assignment]
            return store.hybrid_search(query=query, **merged)

        return super()._get_relevant_documents(
            query, run_manager=run_manager, **kwargs
        )


class BigQueryHybridSearchVectorStore(BigQueryVectorStore):
    """BigQuery Vector Store with hybrid search support.

    Combines BigQuery ``VECTOR_SEARCH()`` (semantic similarity) with
    ``SEARCH()`` (full-text keyword matching) in two modes:

    * **pre_filter** – narrows candidates via ``SEARCH()`` first, then ranks
      by vector distance.  Best when the result *must* contain certain keywords.
    * **rrf** – runs both searches independently and merges results with
      Reciprocal Rank Fusion.  Best when you want to balance keyword relevance
      and semantic similarity.
    """

    search_fields: List[str] = []
    """Columns to apply ``SEARCH()`` on.  Defaults to ``[content_field]``."""

    search_analyzer: Literal["LOG_ANALYZER", "NO_OP_ANALYZER", "PATTERN_ANALYZER"] = (
        "LOG_ANALYZER"
    )
    """Text analyzer for ``SEARCH()``."""

    search_analyzer_options: Optional[str] = None
    """JSON-formatted analyzer options for ``SEARCH()``."""

    hybrid_search_mode: Literal["pre_filter", "rrf"] = "pre_filter"
    """Default hybrid search mode."""

    rrf_k: int = 60
    """RRF constant.  Higher values reduce the gap between top and lower ranks."""

    query_task_type: Optional[str] = None
    """Task type to use when embedding queries.

    When ``None`` (default), the embedding model's own default is used
    (``RETRIEVAL_QUERY`` for Google embeddings).  Set this to override —
    e.g. ``"QUESTION_ANSWERING"``, ``"FACT_VERIFICATION"``,
    ``"SEMANTIC_SIMILARITY"``, or ``"CODE_RETRIEVAL_QUERY"``.

    Only applies when the embedding model accepts a ``task_type`` (or
    ``embeddings_task_type``) kwarg; otherwise a warning is logged and the
    setting is ignored.

    See https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types
    """

    document_task_type: Optional[str] = None
    """Task type to use when embedding documents in :meth:`add_texts`.

    When ``None`` (default), the embedding model's own default is used
    (``RETRIEVAL_DOCUMENT`` for Google embeddings).  Mirrors
    :attr:`query_task_type` and follows the same compatibility rules.
    """

    _have_search_index: bool = False
    _creating_search_index: bool = False
    _last_search_index_check: datetime = datetime.min

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        if not self.search_fields:
            self.search_fields = [self.content_field]
        self._maybe_create_search_index()
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def as_retriever(self, **kwargs: Any) -> BigQueryHybridSearchRetriever:
        """Return a retriever that supports ``search_type="hybrid"``.

        All standard search types (``similarity``, ``mmr``,
        ``similarity_score_threshold``) remain available.

        Example::

            retriever = store.as_retriever(
                search_type="hybrid",
                search_kwargs={"k": 4, "text_query": "BigQuery"},
            )
            docs = retriever.invoke("serverless data warehouse")
        """
        tags = kwargs.pop("tags", None) or [*self._get_retriever_tags()]
        return BigQueryHybridSearchRetriever(
            vectorstore=self, tags=tags, **kwargs
        )

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 25,
        text_query: Optional[str] = None,
        filter: Optional[Union[Dict[str, Any], str]] = None,
        hybrid_search_mode: Optional[Literal["pre_filter", "rrf"]] = None,
        query_task_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Hybrid search combining keyword and vector similarity.

        Args:
            query: Search query used for both embedding generation and keyword
                search (unless *text_query* is provided separately).
            k: Number of documents to return.
            fetch_k: Number of candidates per source in RRF mode.
            text_query: Separate keyword query.  Falls back to *query*.
            filter: Metadata filter (dict or SQL WHERE clause).
            hybrid_search_mode: Override instance default.
            query_task_type: Per-call override for :attr:`query_task_type`.

        Returns:
            Documents ranked by the chosen hybrid strategy.
        """
        results = self._hybrid_search_with_scores(
            query=query,
            k=k,
            fetch_k=fetch_k,
            text_query=text_query,
            filter=filter,
            hybrid_search_mode=hybrid_search_mode,
            query_task_type=query_task_type,
            **kwargs,
        )
        return [doc for doc, _ in results]

    def hybrid_search_with_score(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 25,
        text_query: Optional[str] = None,
        filter: Optional[Union[Dict[str, Any], str]] = None,
        hybrid_search_mode: Optional[Literal["pre_filter", "rrf"]] = None,
        query_task_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Same as :meth:`hybrid_search` but also returns scores."""
        return self._hybrid_search_with_scores(
            query=query,
            k=k,
            fetch_k=fetch_k,
            text_query=text_query,
            filter=filter,
            hybrid_search_mode=hybrid_search_mode,
            query_task_type=query_task_type,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal – orchestration
    # ------------------------------------------------------------------

    def _hybrid_search_with_scores(
        self,
        query: str,
        k: int,
        fetch_k: int,
        text_query: Optional[str],
        filter: Optional[Union[Dict[str, Any], str]],
        hybrid_search_mode: Optional[Literal["pre_filter", "rrf"]],
        query_task_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        from google.cloud import bigquery  # type: ignore[attr-defined]

        mode = hybrid_search_mode or self.hybrid_search_mode
        resolved_text_query = text_query or query
        effective_task_type = query_task_type or self.query_task_type
        embedding = _call_with_task_type(
            self.embedding.embed_query, query, effective_task_type
        )
        options = kwargs.get("options", None)

        if mode == "pre_filter":
            sql, job_config = self._build_prefilter_query(
                embedding=embedding,
                text_query=resolved_text_query,
                filter=filter,
                k=k,
                options=options,
            )
        elif mode == "rrf":
            sql, job_config = self._build_rrf_query(
                embedding=embedding,
                text_query=resolved_text_query,
                filter=filter,
                k=k,
                fetch_k=fetch_k,
                options=options,
            )
        else:
            raise ValueError(f"Unknown hybrid_search_mode: {mode!r}")

        results = self._bq_client.query(
            sql,
            job_config=job_config,
            api_method=bigquery.enums.QueryApiMethod.QUERY,
        )
        return self._rows_to_docs_with_scores(list(results), mode=mode)

    # ------------------------------------------------------------------
    # Internal – SQL builders
    # ------------------------------------------------------------------

    def _search_clause(self, alias: Optional[str] = None) -> str:
        prefix = f"{alias}." if alias else ""
        if len(self.search_fields) == 1:
            col_expr = f"{prefix}{self.search_fields[0]}"
        else:
            col_expr = "({})".format(
                ", ".join(f"{prefix}{f}" for f in self.search_fields)
            )

        parts = [f"SEARCH({col_expr}, @text_query"]
        parts.append(f", analyzer => '{self.search_analyzer}'")
        if self.search_analyzer_options:
            parts.append(", analyzer_options => @analyzer_options")
        parts.append(")")
        return "".join(parts)

    def _build_prefilter_query(
        self,
        embedding: List[float],
        text_query: str,
        filter: Optional[Union[Dict[str, Any], str]],
        k: int,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, "bq.QueryJobConfig"]:
        from google.cloud import bigquery  # type: ignore[attr-defined]

        where_filter = self._create_filters(filter)
        search_cond = self._search_clause()
        options_json = json.dumps(options if options else {})

        sql = f"""
WITH embeddings AS (
  SELECT 0 AS row_num, @emb_0 AS {self.embedding_field}
)
SELECT base.*, query.row_num, distance AS score
FROM VECTOR_SEARCH(
  (SELECT *
   FROM `{self.full_table_id}`
   WHERE {search_cond}
     AND {where_filter}),
  "{self.embedding_field}",
  (SELECT row_num, {self.embedding_field} FROM embeddings),
  distance_type => "{self.distance_type}",
  top_k => {k},
  options => '{options_json}'
)
ORDER BY score
"""

        params: List[Any] = [
            bigquery.ArrayQueryParameter("emb_0", "FLOAT64", embedding),
            bigquery.ScalarQueryParameter("text_query", "STRING", text_query),
        ]
        if self.search_analyzer_options:
            params.append(
                bigquery.ScalarQueryParameter(
                    "analyzer_options", "STRING", self.search_analyzer_options
                )
            )

        job_config = bigquery.QueryJobConfig(
            query_parameters=params,
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE,
        )
        return sql, job_config

    def _build_rrf_query(
        self,
        embedding: List[float],
        text_query: str,
        filter: Optional[Union[Dict[str, Any], str]],
        k: int,
        fetch_k: int,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, "bq.QueryJobConfig"]:
        from google.cloud import bigquery  # type: ignore[attr-defined]

        where_filter = self._create_filters(filter)
        search_cond = self._search_clause(alias="t")
        options_json = json.dumps(options if options else {})

        sql = f"""
WITH vector_results AS (
  SELECT
    base.*,
    distance,
    ROW_NUMBER() OVER (ORDER BY distance ASC) AS vector_rank
  FROM VECTOR_SEARCH(
    (SELECT * FROM `{self.full_table_id}` WHERE {where_filter}),
    "{self.embedding_field}",
    query_value => @query_embedding,
    top_k => {fetch_k},
    distance_type => "{self.distance_type}",
    options => '{options_json}'
  )
),
text_results AS (
  SELECT
    t.*,
    ROW_NUMBER() OVER () AS text_rank
  FROM `{self.full_table_id}` AS t
  WHERE {search_cond}
    AND {where_filter}
  LIMIT {fetch_k}
),
combined AS (
  SELECT
    COALESCE(v.{self.doc_id_field}, t.{self.doc_id_field}) AS {self.doc_id_field},
    COALESCE(v.{self.content_field}, t.{self.content_field}) AS {self.content_field},
    COALESCE(v.{self.embedding_field}, t.{self.embedding_field}) AS {self.embedding_field},
    v.distance,
    v.vector_rank,
    t.text_rank,
    IFNULL(1.0 / ({self.rrf_k} + v.vector_rank), 0)
      + IFNULL(1.0 / ({self.rrf_k} + t.text_rank), 0) AS rrf_score
  FROM vector_results v
  FULL OUTER JOIN text_results t
    ON v.{self.doc_id_field} = t.{self.doc_id_field}
)
SELECT *
FROM combined
ORDER BY rrf_score DESC
LIMIT {k}
"""

        params: List[Any] = [
            bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", embedding),
            bigquery.ScalarQueryParameter("text_query", "STRING", text_query),
        ]
        if self.search_analyzer_options:
            params.append(
                bigquery.ScalarQueryParameter(
                    "analyzer_options", "STRING", self.search_analyzer_options
                )
            )

        job_config = bigquery.QueryJobConfig(
            query_parameters=params,
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE,
        )
        return sql, job_config

    # ------------------------------------------------------------------
    # Internal – result parsing
    # ------------------------------------------------------------------

    def _rows_to_docs_with_scores(
        self,
        rows: List[Any],
        mode: str,
    ) -> List[Tuple[Document, float]]:
        score_field = "rrf_score" if mode == "rrf" else "score"
        skip_fields = {
            self.embedding_field,
            self.content_field,
            "row_num",
            "score",
            "distance",
            "vector_rank",
            "text_rank",
            "rrf_score",
        }

        results: List[Tuple[Document, float]] = []
        for row in rows:
            metadata = {
                field: row[field] for field in row.keys() if field not in skip_fields
            }
            doc = Document(
                page_content=row[self.content_field] or "",
                metadata=metadata,
            )
            results.append((doc, float(row[score_field])))
        return results

    # ------------------------------------------------------------------
    # Search index management
    # ------------------------------------------------------------------

    def _maybe_create_search_index(self) -> None:
        if self._have_search_index or self._creating_search_index:
            return
        if datetime.utcnow() - self._last_search_index_check < _SEARCH_INDEX_CHECK_INTERVAL:
            return

        with _search_index_lock:
            self._last_search_index_check = datetime.utcnow()
            if self._check_search_index_exists():
                self._have_search_index = True
                logger.debug("Search index already exists for %s", self.full_table_id)
            else:
                logger.debug("Creating search index for %s", self.full_table_id)
                self._creating_search_index = True
                Thread(target=self._create_search_index, daemon=True).start()

    def _check_search_index_exists(self) -> bool:
        from google.cloud import bigquery  # type: ignore[attr-defined]

        sql = (
            f"SELECT 1 FROM `{self.project_id}.{self.dataset_name}"
            f".INFORMATION_SCHEMA.SEARCH_INDEXES`"
            f" WHERE table_name = '{self.table_name}'"
        )
        try:
            job = self._bq_client.query(
                sql, api_method=bigquery.enums.QueryApiMethod.QUERY
            )
            return job.result().total_rows > 0
        except Exception:
            logger.debug("Failed to check search index existence", exc_info=True)
            return False

    def _create_search_index(self) -> None:
        index_name = f"{self.table_name}_search_index"
        columns = ", ".join(self.search_fields)
        sql = (
            f"CREATE SEARCH INDEX IF NOT EXISTS `{index_name}`"
            f" ON `{self.full_table_id}` ({columns})"
            f" OPTIONS (analyzer = '{self.search_analyzer}'"
        )
        if self.search_analyzer_options:
            sql += f", analyzer_options = '{self.search_analyzer_options}'"
        sql += ")"

        try:
            self._bq_client.query(sql).result()
            self._have_search_index = True
            logger.info("Search index created: %s", index_name)
        except ClientError as exc:
            logger.warning("Search index creation failed: %s", exc)
        finally:
            self._creating_search_index = False

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    @override
    def add_texts(  # type: ignore[override]
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        *,
        document_task_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed *texts* and store them.

        Mirrors the parent implementation but routes the embedding call
        through :func:`_call_with_task_type` so :attr:`document_task_type`
        (or the per-call ``document_task_type`` kwarg) is honored.
        """
        effective_task_type = document_task_type or self.document_task_type
        embs = _call_with_task_type(
            self.embedding.embed_documents, list(texts), effective_task_type
        )
        return self.add_texts_with_embeddings(
            texts=texts, embs=embs, metadatas=metadatas, **kwargs
        )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls: Type["BigQueryHybridSearchVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "BigQueryHybridSearchVectorStore":
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store
