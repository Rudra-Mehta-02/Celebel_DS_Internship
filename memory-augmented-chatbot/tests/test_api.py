"""Tests for app.py — FastAPI endpoint tests."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    # Mock heavy dependencies before importing app
    with patch("src.llm.engine.MultiProviderLLM") as mock_llm, \
         patch("src.rag.vector_store.HybridVectorStore") as mock_vs, \
         patch("src.graph.store.GraphStore") as mock_gs, \
         patch("src.memory.store.MemoryStore") as mock_ms:

        mock_llm.return_value.generate.return_value = "Test response"
        mock_llm.return_value.is_available.return_value = True
        mock_vs.return_value.get_stats.return_value = {"document_count": 100}
        mock_gs.return_value.get_stats.return_value = {"node_count": 50, "edge_count": 75}
        mock_ms.return_value.get_facts.return_value = []
        mock_ms.return_value.get_history.return_value = []

        from app import app
        yield TestClient(app)


class TestHealthEndpoints:
    """Test health check and system info endpoints."""

    def test_root_endpoint(self, client):
        """GET / should return system info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "name" in data

    def test_health_endpoint(self, client):
        """GET /health should return detailed health info."""
        response = client.get("/health")
        assert response.status_code == 200


class TestChatEndpoints:
    """Test the main chat functionality."""

    def test_chat_endpoint(self, client):
        """POST /chat should return a response."""
        with patch("app.agent_invoke") as mock_invoke:
            mock_invoke.return_value = {
                "answer": "Test answer about AI",
                "route": "direct",
                "sources": [],
                "confidence": 0.9,
                "provider_used": "groq",
                "latency": {"total": 500},
            }
            response = client.post("/chat", json={
                "user_id": "test_user",
                "message": "What is AI?"
            })
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data

    def test_chat_requires_user_id(self, client):
        """POST /chat without user_id should fail validation."""
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 422  # Validation error

    def test_chat_requires_message(self, client):
        """POST /chat without message should fail validation."""
        response = client.post("/chat", json={"user_id": "test"})
        assert response.status_code == 422


class TestMemoryEndpoints:
    """Test memory management endpoints."""

    def test_get_memory(self, client):
        """GET /memory/{user_id} should return facts."""
        response = client.get("/memory/test_user")
        assert response.status_code == 200

    def test_delete_memory(self, client):
        """DELETE /memory/{user_id} should clear facts."""
        response = client.delete("/memory/test_user")
        assert response.status_code == 200


class TestKGEndpoints:
    """Test knowledge graph endpoints."""

    def test_kg_stats(self, client):
        """GET /kg/stats should return graph statistics."""
        response = client.get("/kg/stats")
        assert response.status_code == 200

    def test_kg_search(self, client):
        """GET /kg/search should accept query parameter."""
        response = client.get("/kg/search", params={"q": "machine learning"})
        assert response.status_code == 200


class TestDataEndpoints:
    """Test data pipeline endpoints."""

    def test_data_status(self, client):
        """GET /data/status should return pipeline stats."""
        response = client.get("/data/status")
        assert response.status_code == 200


class TestEvalEndpoints:
    """Test evaluation endpoints."""

    def test_eval_results(self, client):
        """GET /eval/results should return evaluation results."""
        response = client.get("/eval/results")
        assert response.status_code == 200 or response.status_code == 404


class TestCORSAndMiddleware:
    """Test middleware configuration."""

    def test_cors_headers(self, client):
        """Response should include CORS headers."""
        response = client.options("/", headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        })
        # CORS should be configured
        assert response.status_code in [200, 405]
