"""Integration tests for BigQueryHybridSearchVectorStore.

These tests require a live BigQuery connection and GCP credentials.
Run with: pytest tests/integration_tests/ -m integration

Environment variables are loaded from .env file via conftest.py
(see tests/.env.example for the template).
"""

from __future__ import annotations

import os
import uuid
from typing import Generator

import pytest

GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BIGQUERY_LOCATION = os.environ.get("BIGQUERY_LOCATION", "us-central1")
BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "test_hybridsearch")
BIGQUERY_TABLE = os.environ.get("BIGQUERY_TABLE", f"hybrid_test_{uuid.uuid4().hex[:8]}")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "gemini-embedding-001")


@pytest.fixture(scope="module")
def embedding_model():
    """Use VertexAI embeddings if available, else skip."""
    try:
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
        )
    except ImportError:
        pytest.skip("langchain-google-vertexai not installed")


@pytest.fixture(scope="module")
def store(embedding_model) -> Generator:
    if not GOOGLE_CLOUD_PROJECT:
        pytest.skip("GOOGLE_CLOUD_PROJECT not set")

    from langchain_bigquery_hybridsearch import BigQueryHybridSearchVectorStore

    vs = BigQueryHybridSearchVectorStore(
        project_id=GOOGLE_CLOUD_PROJECT,
        dataset_name=BIGQUERY_DATASET,
        table_name=BIGQUERY_TABLE,
        location=BIGQUERY_LOCATION,
        embedding=embedding_model,
        distance_type="COSINE",
        search_analyzer="LOG_ANALYZER",
    )

    vs.add_texts(
        texts=[
            "BigQuery is a serverless data warehouse by Google Cloud.",
            "VECTOR_SEARCH finds semantically similar embeddings in BigQuery.",
            "The SEARCH function performs full-text keyword matching.",
            "Hybrid search combines keyword and semantic search for better results.",
            "Cloud Spanner is a globally distributed relational database.",
        ],
        metadatas=[
            {"source": "docs", "topic": "bigquery"},
            {"source": "docs", "topic": "vector"},
            {"source": "docs", "topic": "search"},
            {"source": "docs", "topic": "hybrid"},
            {"source": "docs", "topic": "spanner"},
        ],
    )

    yield vs

    # Cleanup
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT, location=BIGQUERY_LOCATION)
        client.delete_table(
            f"{GOOGLE_CLOUD_PROJECT}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}",
            not_found_ok=True,
        )
    except Exception:
        pass


@pytest.mark.integration
class TestPreFilterMode:
    def test_basic_search(self, store) -> None:
        results = store.hybrid_search(
            query="How does BigQuery search work?",
            text_query="BigQuery",
            k=3,
            hybrid_search_mode="pre_filter",
        )
        assert len(results) > 0
        assert all(
            "BigQuery" in doc.page_content or "bigquery" in doc.page_content.lower()
            for doc in results
        )

    def test_with_score(self, store) -> None:
        results = store.hybrid_search_with_score(
            query="serverless data warehouse",
            text_query="serverless",
            k=3,
            hybrid_search_mode="pre_filter",
        )
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(score, float)

    def test_no_match_returns_empty(self, store) -> None:
        results = store.hybrid_search(
            query="quantum computing algorithms",
            text_query="xyznonexistent123",
            k=3,
            hybrid_search_mode="pre_filter",
        )
        assert len(results) == 0


@pytest.mark.integration
class TestRRFMode:
    def test_basic_search(self, store) -> None:
        results = store.hybrid_search(
            query="How does keyword search work?",
            text_query="SEARCH",
            k=3,
            hybrid_search_mode="rrf",
        )
        assert len(results) > 0

    def test_with_score(self, store) -> None:
        results = store.hybrid_search_with_score(
            query="semantic similarity search",
            text_query="VECTOR_SEARCH",
            k=3,
            hybrid_search_mode="rrf",
        )
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(score, float)
            assert score > 0

    def test_rrf_includes_both_sources(self, store) -> None:
        results = store.hybrid_search(
            query="distributed relational database",
            text_query="hybrid search",
            k=5,
            fetch_k=5,
            hybrid_search_mode="rrf",
        )
        contents = [doc.page_content for doc in results]
        has_vector_match = any("Spanner" in c or "distributed" in c for c in contents)
        has_text_match = any("Hybrid" in c or "hybrid" in c for c in contents)
        assert has_vector_match or has_text_match


@pytest.mark.integration
class TestFallbackBehavior:
    def test_similarity_search_still_works(self, store) -> None:
        results = store.similarity_search("BigQuery serverless", k=3)
        assert len(results) > 0

    def test_query_as_text_query_default(self, store) -> None:
        results = store.hybrid_search(
            query="BigQuery VECTOR_SEARCH",
            k=3,
            hybrid_search_mode="pre_filter",
        )
        assert len(results) > 0


# ------------------------------------------------------------------
# Embedding task types
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def store_qa(embedding_model) -> Generator:
    """Store whose query and document embeddings both use QUESTION_ANSWERING.

    Verifies that the task_type is honored end-to-end (ingestion + search)
    and that documents embedded with a non-default task type can still be
    retrieved.
    """
    if not GOOGLE_CLOUD_PROJECT:
        pytest.skip("GOOGLE_CLOUD_PROJECT not set")

    from langchain_bigquery_hybridsearch import BigQueryHybridSearchVectorStore

    qa_table_name = f"hybrid_qa_{uuid.uuid4().hex[:8]}"
    vs = BigQueryHybridSearchVectorStore(
        project_id=GOOGLE_CLOUD_PROJECT,
        dataset_name=BIGQUERY_DATASET,
        table_name=qa_table_name,
        location=BIGQUERY_LOCATION,
        embedding=embedding_model,
        distance_type="COSINE",
        search_analyzer="LOG_ANALYZER",
        query_task_type="QUESTION_ANSWERING",
        document_task_type="QUESTION_ANSWERING",
    )
    vs.add_texts(
        texts=[
            "BigQuery is a serverless data warehouse by Google Cloud.",
            "VECTOR_SEARCH finds semantically similar embeddings in BigQuery.",
            "The SEARCH function performs full-text keyword matching.",
        ],
        metadatas=[
            {"source": "docs", "topic": "bigquery"},
            {"source": "docs", "topic": "vector"},
            {"source": "docs", "topic": "search"},
        ],
    )
    yield vs

    try:
        from google.cloud import bigquery

        client = bigquery.Client(
            project=GOOGLE_CLOUD_PROJECT, location=BIGQUERY_LOCATION
        )
        client.delete_table(
            f"{GOOGLE_CLOUD_PROJECT}.{BIGQUERY_DATASET}.{qa_table_name}",
            not_found_ok=True,
        )
    except Exception:
        pass


@pytest.mark.integration
class TestEmbeddingTaskType:
    """End-to-end checks that ``query_task_type`` / ``document_task_type``
    reach the Vertex / Gemini embeddings API without error and return results.
    """

    def test_per_call_query_task_type_override(self, store) -> None:
        """Per-call override on a default-configured store."""
        results = store.hybrid_search(
            query="What is a serverless data warehouse?",
            text_query="serverless",
            k=3,
            query_task_type="QUESTION_ANSWERING",
        )
        assert len(results) > 0

    def test_per_call_overrides_instance_field(self, store_qa) -> None:
        """Per-call value should win over the store's instance-level field."""
        results = store_qa.hybrid_search(
            query="distributed database",
            text_query="VECTOR_SEARCH",
            k=3,
            query_task_type="SEMANTIC_SIMILARITY",
        )
        assert len(results) > 0

    def test_qa_round_trip(self, store_qa) -> None:
        """Documents embedded with QUESTION_ANSWERING should be retrievable
        with a query embedded using the same task type.
        """
        results = store_qa.hybrid_search(
            query="What does the SEARCH function do?",
            text_query="SEARCH",
            k=3,
        )
        assert len(results) > 0
        assert any(
            "SEARCH" in doc.page_content for doc in results
        )

    def test_with_score_carries_task_type(self, store_qa) -> None:
        """``hybrid_search_with_score`` path also routes the task type."""
        results = store_qa.hybrid_search_with_score(
            query="How are embeddings compared?",
            text_query="VECTOR_SEARCH",
            k=3,
            hybrid_search_mode="rrf",
        )
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(score, float)
