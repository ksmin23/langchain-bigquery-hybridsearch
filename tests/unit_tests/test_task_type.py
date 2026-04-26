"""Unit tests for embedding task_type routing.

Covers the ``_call_with_task_type`` helper plus the integration points in
``BigQueryHybridSearchVectorStore`` (``_hybrid_search_with_scores`` and
``add_texts``).  No GCP credentials or network access are needed; embeddings
are replaced with hand-written fakes that record their kwargs.
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from langchain_bigquery_hybridsearch import BigQueryHybridSearchVectorStore
from langchain_bigquery_hybridsearch.vectorstore import (
    _call_with_task_type,
    _resolve_task_type_kwarg,
)


# ------------------------------------------------------------------
# Fake embedding classes that mimic the LangChain Google integrations
# ------------------------------------------------------------------


class _GenAIStyleEmbedding:
    """Mimics ``GoogleGenerativeAIEmbeddings`` (uses ``task_type``)."""

    def __init__(self) -> None:
        self.query_calls: List[dict] = []
        self.docs_calls: List[dict] = []

    def embed_query(
        self, text: str, *, task_type: Optional[str] = None
    ) -> List[float]:
        self.query_calls.append({"text": text, "task_type": task_type})
        return [0.1, 0.2, 0.3]

    def embed_documents(
        self, texts: List[str], *, task_type: Optional[str] = None
    ) -> List[List[float]]:
        self.docs_calls.append({"texts": list(texts), "task_type": task_type})
        return [[0.1, 0.2, 0.3] for _ in texts]


class _VertexStyleEmbedding:
    """Mimics ``VertexAIEmbeddings`` (uses ``embeddings_task_type``)."""

    def __init__(self) -> None:
        self.query_calls: List[dict] = []
        self.docs_calls: List[dict] = []

    def embed_query(
        self, text: str, *, embeddings_task_type: Optional[str] = None
    ) -> List[float]:
        self.query_calls.append(
            {"text": text, "embeddings_task_type": embeddings_task_type}
        )
        return [0.4, 0.5, 0.6]

    def embed_documents(
        self,
        texts: List[str],
        *,
        embeddings_task_type: Optional[str] = None,
    ) -> List[List[float]]:
        self.docs_calls.append(
            {"texts": list(texts), "embeddings_task_type": embeddings_task_type}
        )
        return [[0.4, 0.5, 0.6] for _ in texts]


class _NoTaskTypeEmbedding:
    """Mimics a non-Google embedding without any task_type kwarg."""

    def __init__(self) -> None:
        self.query_calls: List[str] = []
        self.docs_calls: List[List[str]] = []

    def embed_query(self, text: str) -> List[float]:
        self.query_calls.append(text)
        return [0.7, 0.8, 0.9]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.docs_calls.append(list(texts))
        return [[0.7, 0.8, 0.9] for _ in texts]


# ------------------------------------------------------------------
# _resolve_task_type_kwarg
# ------------------------------------------------------------------


class TestResolveTaskTypeKwarg:
    def test_prefers_task_type(self) -> None:
        assert (
            _resolve_task_type_kwarg("X.embed_query", ("self", "text", "task_type"))
            == "task_type"
        )

    def test_falls_back_to_embeddings_task_type(self) -> None:
        assert (
            _resolve_task_type_kwarg(
                "X.embed_query", ("self", "text", "embeddings_task_type")
            )
            == "embeddings_task_type"
        )

    def test_returns_none_when_neither_present(self) -> None:
        assert _resolve_task_type_kwarg("X.embed_query", ("self", "text")) is None


# ------------------------------------------------------------------
# _call_with_task_type
# ------------------------------------------------------------------


class TestCallWithTaskType:
    def test_none_task_type_calls_method_unchanged(self) -> None:
        emb = _GenAIStyleEmbedding()
        _call_with_task_type(emb.embed_query, "hello", None)
        assert emb.query_calls == [{"text": "hello", "task_type": None}]

    def test_genai_style_routes_to_task_type(self) -> None:
        emb = _GenAIStyleEmbedding()
        _call_with_task_type(emb.embed_query, "hello", "QUESTION_ANSWERING")
        assert emb.query_calls == [
            {"text": "hello", "task_type": "QUESTION_ANSWERING"}
        ]

    def test_vertex_style_routes_to_embeddings_task_type(self) -> None:
        emb = _VertexStyleEmbedding()
        _call_with_task_type(emb.embed_query, "hello", "QUESTION_ANSWERING")
        assert emb.query_calls == [
            {"text": "hello", "embeddings_task_type": "QUESTION_ANSWERING"}
        ]

    def test_documents_method_routed_too(self) -> None:
        emb = _GenAIStyleEmbedding()
        _call_with_task_type(
            emb.embed_documents, ["a", "b"], "SEMANTIC_SIMILARITY"
        )
        assert emb.docs_calls == [
            {"texts": ["a", "b"], "task_type": "SEMANTIC_SIMILARITY"}
        ]

    def test_unsupported_embedding_warns_and_drops_task_type(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        emb = _NoTaskTypeEmbedding()
        with caplog.at_level("WARNING"):
            _call_with_task_type(emb.embed_query, "hello", "RETRIEVAL_QUERY")
        assert emb.query_calls == ["hello"]
        assert any("does not accept a task_type" in m for m in caplog.messages)


# ------------------------------------------------------------------
# Vectorstore wiring — test helper
# ------------------------------------------------------------------


def _make_store(
    embedding: Any, **overrides: Any
) -> BigQueryHybridSearchVectorStore:
    """Build a store instance without triggering BigQuery validators."""
    defaults = dict(
        project_id="test-project",
        dataset_name="test_dataset",
        table_name="test_table",
        location="US",
        embedding=embedding,
        embedding_dimension=3,
        content_field="content",
        embedding_field="embedding",
        doc_id_field="doc_id",
        distance_type="EUCLIDEAN",
        search_fields=["content"],
        search_analyzer="LOG_ANALYZER",
        search_analyzer_options=None,
        hybrid_search_mode="pre_filter",
        rrf_k=60,
        query_task_type=None,
        document_task_type=None,
    )
    defaults.update(overrides)
    store = BigQueryHybridSearchVectorStore.model_construct(**defaults)
    store._bq_client = MagicMock()
    store._full_table_id = (
        f"{defaults['project_id']}.{defaults['dataset_name']}.{defaults['table_name']}"
    )
    store.table_schema = None
    return store


# ------------------------------------------------------------------
# _hybrid_search_with_scores — query-side task type
# ------------------------------------------------------------------


class TestHybridSearchTaskType:
    def _patch_bq_query(
        self, store: BigQueryHybridSearchVectorStore
    ) -> MagicMock:
        """Stub out BigQuery execution so the SQL is built but not run."""
        mock_iter = MagicMock(return_value=iter([]))
        store._bq_client.query = MagicMock(return_value=mock_iter())
        return store._bq_client.query

    def test_default_uses_no_task_type(self) -> None:
        emb = _GenAIStyleEmbedding()
        store = _make_store(emb)
        self._patch_bq_query(store)
        with patch.object(
            store, "_rows_to_docs_with_scores", return_value=[]
        ):
            store.hybrid_search(query="hello", k=3)
        # Field unset and no per-call override → embed_query called with task_type=None
        assert emb.query_calls == [{"text": "hello", "task_type": None}]

    def test_instance_field_applied(self) -> None:
        emb = _GenAIStyleEmbedding()
        store = _make_store(emb, query_task_type="QUESTION_ANSWERING")
        self._patch_bq_query(store)
        with patch.object(
            store, "_rows_to_docs_with_scores", return_value=[]
        ):
            store.hybrid_search(query="hello", k=3)
        assert emb.query_calls == [
            {"text": "hello", "task_type": "QUESTION_ANSWERING"}
        ]

    def test_per_call_overrides_instance_field(self) -> None:
        emb = _GenAIStyleEmbedding()
        store = _make_store(emb, query_task_type="RETRIEVAL_QUERY")
        self._patch_bq_query(store)
        with patch.object(
            store, "_rows_to_docs_with_scores", return_value=[]
        ):
            store.hybrid_search(
                query="how does it work?",
                k=3,
                query_task_type="QUESTION_ANSWERING",
            )
        assert emb.query_calls == [
            {"text": "how does it work?", "task_type": "QUESTION_ANSWERING"}
        ]

    def test_vertex_style_embedding(self) -> None:
        emb = _VertexStyleEmbedding()
        store = _make_store(emb, query_task_type="CODE_RETRIEVAL_QUERY")
        self._patch_bq_query(store)
        with patch.object(
            store, "_rows_to_docs_with_scores", return_value=[]
        ):
            store.hybrid_search(query="def foo():", k=3)
        assert emb.query_calls == [
            {
                "text": "def foo():",
                "embeddings_task_type": "CODE_RETRIEVAL_QUERY",
            }
        ]

    def test_unsupported_embedding_with_task_type_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        emb = _NoTaskTypeEmbedding()
        store = _make_store(emb, query_task_type="RETRIEVAL_QUERY")
        self._patch_bq_query(store)
        with caplog.at_level("WARNING"), patch.object(
            store, "_rows_to_docs_with_scores", return_value=[]
        ):
            store.hybrid_search(query="hello", k=3)
        assert emb.query_calls == ["hello"]
        assert any("does not accept a task_type" in m for m in caplog.messages)

    def test_with_score_path_also_routes(self) -> None:
        emb = _GenAIStyleEmbedding()
        store = _make_store(emb, query_task_type="FACT_VERIFICATION")
        self._patch_bq_query(store)
        with patch.object(
            store, "_rows_to_docs_with_scores", return_value=[]
        ):
            store.hybrid_search_with_score(query="claim X", k=3)
        assert emb.query_calls == [
            {"text": "claim X", "task_type": "FACT_VERIFICATION"}
        ]


# ------------------------------------------------------------------
# add_texts — document-side task type
# ------------------------------------------------------------------


class TestAddTextsTaskType:
    def test_default_uses_no_task_type(self) -> None:
        emb = _GenAIStyleEmbedding()
        store = _make_store(emb)
        with patch.object(
            type(store), "add_texts_with_embeddings", return_value=["id1"]
        ) as mock_add:
            store.add_texts(["doc"])
        assert emb.docs_calls == [{"texts": ["doc"], "task_type": None}]
        mock_add.assert_called_once()

    def test_instance_field_applied(self) -> None:
        emb = _GenAIStyleEmbedding()
        store = _make_store(emb, document_task_type="SEMANTIC_SIMILARITY")
        with patch.object(
            type(store), "add_texts_with_embeddings", return_value=["id1", "id2"]
        ):
            store.add_texts(["a", "b"])
        assert emb.docs_calls == [
            {"texts": ["a", "b"], "task_type": "SEMANTIC_SIMILARITY"}
        ]

    def test_per_call_overrides_instance_field(self) -> None:
        emb = _GenAIStyleEmbedding()
        store = _make_store(emb, document_task_type="RETRIEVAL_DOCUMENT")
        with patch.object(
            type(store), "add_texts_with_embeddings", return_value=["id1"]
        ):
            store.add_texts(
                ["code snippet"],
                document_task_type="CODE_RETRIEVAL_QUERY",
            )
        assert emb.docs_calls == [
            {"texts": ["code snippet"], "task_type": "CODE_RETRIEVAL_QUERY"}
        ]

    def test_vertex_style_embedding_routed(self) -> None:
        emb = _VertexStyleEmbedding()
        store = _make_store(emb, document_task_type="QUESTION_ANSWERING")
        with patch.object(
            type(store), "add_texts_with_embeddings", return_value=["id1"]
        ):
            store.add_texts(["passage"])
        assert emb.docs_calls == [
            {"texts": ["passage"], "embeddings_task_type": "QUESTION_ANSWERING"}
        ]

    def test_metadatas_propagated(self) -> None:
        emb = _GenAIStyleEmbedding()
        store = _make_store(emb)
        metadatas = [{"k": "v"}]
        with patch.object(
            type(store), "add_texts_with_embeddings", return_value=["id1"]
        ) as mock_add:
            store.add_texts(["doc"], metadatas)
        # Verify metadatas reached add_texts_with_embeddings
        _, kwargs = mock_add.call_args
        assert kwargs["metadatas"] == metadatas
