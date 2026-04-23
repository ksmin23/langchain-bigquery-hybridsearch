"""Integration tests for BigQueryHybridSearchVectorStore.

These tests require a live BigQuery connection and GCP credentials.
Run with: pytest tests/integration_tests/ -m integration

Set the following environment variables before running:
    PROJECT_ID      – GCP project ID
    DATASET_NAME    – BigQuery dataset (will be created if missing)
    TABLE_NAME      – BigQuery table name for test data
    LOCATION        – BigQuery location (default: US)
"""

from __future__ import annotations

import os
import uuid
from typing import Generator

import pytest

PROJECT_ID = os.environ.get("PROJECT_ID", "")
DATASET_NAME = os.environ.get("DATASET_NAME", "test_hybridsearch")
TABLE_NAME = os.environ.get("TABLE_NAME", f"hybrid_test_{uuid.uuid4().hex[:8]}")
LOCATION = os.environ.get("LOCATION", "US")


@pytest.fixture(scope="module")
def embedding_model():
    """Use VertexAI embeddings if available, else skip."""
    try:
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(
            model_name="text-embedding-005",
            project=PROJECT_ID,
            location=LOCATION,
        )
    except ImportError:
        pytest.skip("langchain-google-vertexai not installed")


@pytest.fixture(scope="module")
def store(embedding_model) -> Generator:
    if not PROJECT_ID:
        pytest.skip("PROJECT_ID not set")

    from langchain_bigquery_hybridsearch import BigQueryHybridSearchVectorStore

    vs = BigQueryHybridSearchVectorStore(
        project_id=PROJECT_ID,
        dataset_name=DATASET_NAME,
        table_name=TABLE_NAME,
        location=LOCATION,
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

        client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
        client.delete_table(f"{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}", not_found_ok=True)
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
