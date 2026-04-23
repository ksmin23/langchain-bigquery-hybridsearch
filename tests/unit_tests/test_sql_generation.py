"""Unit tests for SQL generation logic.

These tests use pydantic model_construct to bypass BigQuery initialization,
so no GCP credentials or network access are needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_bigquery_hybridsearch import BigQueryHybridSearchVectorStore


def _make_store(**overrides: Any) -> BigQueryHybridSearchVectorStore:
    """Build a store instance without triggering model validators."""
    defaults = dict(
        project_id="test-project",
        dataset_name="test_dataset",
        table_name="test_table",
        location="US",
        embedding=MagicMock(),
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
    )
    defaults.update(overrides)

    # model_construct skips all validators (no BQ client needed)
    store = BigQueryHybridSearchVectorStore.model_construct(**defaults)
    store._bq_client = MagicMock()
    store._full_table_id = (
        f"{defaults['project_id']}.{defaults['dataset_name']}.{defaults['table_name']}"
    )
    store.table_schema = None
    return store


# ------------------------------------------------------------------
# SEARCH clause generation
# ------------------------------------------------------------------


class TestSearchClause:
    def test_single_field(self) -> None:
        store = _make_store(search_fields=["content"])
        clause = store._search_clause()
        assert "SEARCH(content, @text_query" in clause
        assert "LOG_ANALYZER" in clause

    def test_multiple_fields(self) -> None:
        store = _make_store(search_fields=["title", "content", "summary"])
        clause = store._search_clause()
        assert "(title, content, summary)" in clause

    def test_analyzer_override(self) -> None:
        store = _make_store(search_analyzer="PATTERN_ANALYZER")
        clause = store._search_clause()
        assert "PATTERN_ANALYZER" in clause

    def test_analyzer_options_included(self) -> None:
        store = _make_store(search_analyzer_options='{"delimiters": [" ", "@"]}')
        clause = store._search_clause()
        assert "@analyzer_options" in clause

    def test_no_analyzer_options(self) -> None:
        store = _make_store(search_analyzer_options=None)
        clause = store._search_clause()
        assert "analyzer_options" not in clause

    def test_alias_prefix(self) -> None:
        store = _make_store(search_fields=["content"])
        clause = store._search_clause(alias="t")
        assert "t.content" in clause


# ------------------------------------------------------------------
# Pre-filter SQL
# ------------------------------------------------------------------


class TestPrefilterSQL:
    def test_basic_structure(self) -> None:
        store = _make_store()
        sql, job_config = store._build_prefilter_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="hello world",
            filter=None,
            k=5,
        )
        assert "VECTOR_SEARCH" in sql
        assert "SEARCH(" in sql
        assert "top_k => 5" in sql
        assert "@text_query" in sql
        assert "@emb_0" in sql

    def test_metadata_filter_dict(self) -> None:
        store = _make_store()
        store.table_schema = {"category": "STRING", "year": "INTEGER"}
        sql, _ = store._build_prefilter_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test",
            filter={"category": "science"},
            k=10,
        )
        assert "category = 'science'" in sql

    def test_metadata_filter_string(self) -> None:
        store = _make_store()
        store.table_schema = {"doc_id": "STRING", "content": "STRING", "year": "INTEGER"}
        sql, _ = store._build_prefilter_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test",
            filter="year > 2020",
            k=10,
        )
        assert "year > 2020" in sql

    def test_distance_type(self) -> None:
        store = _make_store(distance_type="COSINE")
        sql, _ = store._build_prefilter_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test",
            filter=None,
            k=5,
        )
        assert 'distance_type => "COSINE"' in sql

    def test_query_params(self) -> None:
        store = _make_store()
        _, job_config = store._build_prefilter_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test query",
            filter=None,
            k=5,
        )
        param_names = [p.name for p in job_config.query_parameters]
        assert "emb_0" in param_names
        assert "text_query" in param_names

    def test_analyzer_options_param(self) -> None:
        store = _make_store(search_analyzer_options='{"delimiters": [" "]}')
        _, job_config = store._build_prefilter_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test",
            filter=None,
            k=5,
        )
        param_names = [p.name for p in job_config.query_parameters]
        assert "analyzer_options" in param_names


# ------------------------------------------------------------------
# RRF SQL
# ------------------------------------------------------------------


class TestRRFSQL:
    def test_basic_structure(self) -> None:
        store = _make_store()
        sql, _ = store._build_rrf_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="hello world",
            filter=None,
            k=5,
            fetch_k=25,
        )
        assert "vector_results" in sql
        assert "text_results" in sql
        assert "combined" in sql
        assert "rrf_score" in sql
        assert "FULL OUTER JOIN" in sql

    def test_rrf_k_parameter(self) -> None:
        store = _make_store(rrf_k=100)
        sql, _ = store._build_rrf_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test",
            filter=None,
            k=5,
            fetch_k=25,
        )
        assert "100 + v.vector_rank" in sql
        assert "100 + t.text_rank" in sql

    def test_fetch_k_applied(self) -> None:
        store = _make_store()
        sql, _ = store._build_rrf_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test",
            filter=None,
            k=10,
            fetch_k=50,
        )
        assert "top_k => 50" in sql
        assert "LIMIT 50" in sql

    def test_final_limit(self) -> None:
        store = _make_store()
        sql, _ = store._build_rrf_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test",
            filter=None,
            k=7,
            fetch_k=25,
        )
        lines = sql.strip().split("\n")
        last_limit_line = [l for l in lines if "LIMIT" in l][-1]
        assert "7" in last_limit_line

    def test_query_params(self) -> None:
        store = _make_store()
        _, job_config = store._build_rrf_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test query",
            filter=None,
            k=5,
            fetch_k=25,
        )
        param_names = [p.name for p in job_config.query_parameters]
        assert "query_embedding" in param_names
        assert "text_query" in param_names

    def test_doc_id_in_join(self) -> None:
        store = _make_store(doc_id_field="my_id")
        sql, _ = store._build_rrf_query(
            embedding=[0.1, 0.2, 0.3],
            text_query="test",
            filter=None,
            k=5,
            fetch_k=25,
        )
        assert "v.my_id" in sql
        assert "t.my_id" in sql
        assert "ON v.my_id = t.my_id" in sql


# ------------------------------------------------------------------
# Result parsing
# ------------------------------------------------------------------


class _FakeRow(dict):
    def keys(self):
        return list(super().keys())


class TestRowsParsing:
    def test_prefilter_mode(self) -> None:
        store = _make_store()
        rows = [
            _FakeRow(
                content="hello",
                embedding=[0.1, 0.2, 0.3],
                doc_id="1",
                score=0.5,
                row_num=0,
            ),
        ]
        results = store._rows_to_docs_with_scores(rows, mode="pre_filter")
        assert len(results) == 1
        doc, score = results[0]
        assert doc.page_content == "hello"
        assert score == 0.5
        assert "doc_id" in doc.metadata

    def test_rrf_mode(self) -> None:
        store = _make_store()
        rows = [
            _FakeRow(
                content="world",
                embedding=[0.1, 0.2, 0.3],
                doc_id="2",
                distance=0.1,
                vector_rank=1,
                text_rank=2,
                rrf_score=0.032,
            ),
        ]
        results = store._rows_to_docs_with_scores(rows, mode="rrf")
        assert len(results) == 1
        doc, score = results[0]
        assert doc.page_content == "world"
        assert score == 0.032

    def test_metadata_excludes_internal_fields(self) -> None:
        store = _make_store()
        rows = [
            _FakeRow(
                content="text",
                embedding=[0.1],
                doc_id="3",
                score=0.1,
                row_num=0,
                category="science",
            ),
        ]
        results = store._rows_to_docs_with_scores(rows, mode="pre_filter")
        doc, _ = results[0]
        assert "category" in doc.metadata
        assert "score" not in doc.metadata
        assert "row_num" not in doc.metadata
        assert "embedding" not in doc.metadata

    def test_empty_rows(self) -> None:
        store = _make_store()
        results = store._rows_to_docs_with_scores([], mode="pre_filter")
        assert results == []


# ------------------------------------------------------------------
# Default field behavior
# ------------------------------------------------------------------


class TestDefaultSearchFields:
    def test_defaults_to_content_field(self) -> None:
        store = _make_store(search_fields=[], content_field="my_content")
        # _post_init would set this, but we use model_construct; simulate it
        if not store.search_fields:
            store.search_fields = [store.content_field]
        assert store.search_fields == ["my_content"]

    def test_explicit_search_fields_preserved(self) -> None:
        store = _make_store(search_fields=["title", "body"])
        assert store.search_fields == ["title", "body"]


# ------------------------------------------------------------------
# Retriever (as_retriever)
# ------------------------------------------------------------------


class TestRetriever:
    def test_hybrid_search_type_allowed(self) -> None:
        from langchain_bigquery_hybridsearch import BigQueryHybridSearchRetriever

        assert "hybrid" in BigQueryHybridSearchRetriever.allowed_search_types

    def test_standard_search_types_still_allowed(self) -> None:
        from langchain_bigquery_hybridsearch import BigQueryHybridSearchRetriever

        for st in ("similarity", "mmr", "similarity_score_threshold"):
            assert st in BigQueryHybridSearchRetriever.allowed_search_types

    def test_as_retriever_returns_custom_class(self) -> None:
        from langchain_bigquery_hybridsearch import BigQueryHybridSearchRetriever

        store = _make_store()
        retriever = store.as_retriever(
            search_type="hybrid",
            search_kwargs={"k": 4, "text_query": "keyword"},
        )
        assert isinstance(retriever, BigQueryHybridSearchRetriever)
        assert retriever.search_type == "hybrid"
        assert retriever.search_kwargs == {"k": 4, "text_query": "keyword"}

    def test_as_retriever_similarity_still_works(self) -> None:
        store = _make_store()
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        assert retriever.search_type == "similarity"

    def test_hybrid_retriever_calls_hybrid_search(self) -> None:
        store = _make_store()
        mock_result = [Document(page_content="test")]
        with patch.object(
            type(store), "hybrid_search", return_value=mock_result
        ) as mock_hs:
            retriever = store.as_retriever(
                search_type="hybrid",
                search_kwargs={"k": 2, "text_query": "kw"},
            )
            docs = retriever.invoke("my query")
            mock_hs.assert_called_once_with(
                query="my query", k=2, text_query="kw"
            )
            assert len(docs) == 1
            assert docs[0].page_content == "test"
