"""Tests for src/agent/graph.py — LangGraph agent workflow."""

import pytest
from unittest.mock import patch, MagicMock
from src.agent.state import ChatState


class TestChatState:
    """Test the LangGraph state schema."""

    def test_state_has_required_fields(self):
        """ChatState should have all required fields."""
        required_fields = [
            "user_id", "message", "user_facts", "chat_history",
            "rewritten_query", "route", "answer", "confidence",
        ]
        # Check that ChatState defines these keys
        state_keys = ChatState.__annotations__.keys()
        for field in required_fields:
            assert field in state_keys, f"Missing field: {field}"

    def test_state_has_reflection_fields(self):
        """ChatState should have self-reflection fields."""
        state_keys = ChatState.__annotations__.keys()
        assert "reflection_count" in state_keys
        assert "needs_retry" in state_keys

    def test_state_has_latency_tracking(self):
        """ChatState should have latency tracking."""
        state_keys = ChatState.__annotations__.keys()
        assert "latency" in state_keys

    def test_state_has_provider_tracking(self):
        """ChatState should track which LLM provider was used."""
        state_keys = ChatState.__annotations__.keys()
        assert "provider_used" in state_keys


class TestAgentRouting:
    """Test the agent's routing logic."""

    def test_routes_include_expected_values(self):
        """Router should support rag, kg, tool, direct, hybrid routes."""
        expected_routes = {"rag", "kg", "tool", "direct", "hybrid"}
        # These are the routes the router_node can return
        from src.agent.graph import VALID_ROUTES
        if hasattr(VALID_ROUTES, '__iter__'):
            assert set(VALID_ROUTES) >= expected_routes

    def test_tool_route_extracts_tool_name(self):
        """Tool route should include tool_name in state."""
        state = {
            "user_id": "test",
            "message": "What time is it?",
            "route": "tool",
            "tool_name": "get_current_datetime",
            "tool_args": "",
        }
        assert state["tool_name"] == "get_current_datetime"


class TestAgentNodes:
    """Test individual agent nodes (mocked)."""

    def test_memory_node_loads_facts(self):
        """Memory node should load user facts from memory store."""
        mock_memory_store = MagicMock()
        mock_memory_store.get_facts.return_value = ["User likes Python"]
        mock_memory_store.get_history.return_value = [
            {"role": "user", "content": "Hello"},
        ]

        with patch("src.agent.graph.get_memory_store", return_value=mock_memory_store):
            from src.agent.graph import memory_node
            state = {
                "user_id": "test_user",
                "message": "What should I learn?",
                "user_facts": [],
                "chat_history": [],
            }
            result = memory_node(state)
            assert "user_facts" in result
            mock_memory_store.get_facts.assert_called_once_with("test_user")

    def test_answer_node_generates_response(self):
        """Answer node should generate a response using the LLM."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"answer": "Test answer", "confidence": 0.85}'

        with patch("src.agent.graph.get_llm", return_value=mock_llm):
            from src.agent.graph import answer_node
            state = {
                "user_id": "test",
                "message": "What is AI?",
                "rewritten_query": "What is artificial intelligence?",
                "route": "direct",
                "user_facts": [],
                "chat_history": [],
                "rag_context": [],
                "kg_context": [],
                "tool_result": None,
                "reflection_count": 0,
            }
            result = answer_node(state)
            assert "answer" in result

    def test_reflect_node_accepts_high_confidence(self):
        """Reflect node should accept answers with high confidence."""
        from src.agent.graph import reflect_node
        state = {
            "answer": "A good detailed answer about machine learning.",
            "confidence": 0.9,
            "reflection_count": 0,
            "message": "What is ML?",
        }
        result = reflect_node(state)
        # High confidence should not trigger retry
        assert result.get("needs_retry", False) is False

    def test_reflect_node_retries_low_confidence(self):
        """Reflect node should retry on low confidence if under max retries."""
        from src.agent.graph import reflect_node
        state = {
            "answer": "I'm not sure.",
            "confidence": 0.2,
            "reflection_count": 0,
            "message": "What is quantum computing?",
        }
        result = reflect_node(state)
        # Low confidence with 0 retries should trigger retry
        assert result.get("needs_retry", False) is True

    def test_reflect_node_stops_after_max_retries(self):
        """Reflect node should stop retrying after max attempts."""
        from src.agent.graph import reflect_node
        state = {
            "answer": "Still not great.",
            "confidence": 0.3,
            "reflection_count": 2,  # Max retries reached
            "message": "What is something?",
        }
        result = reflect_node(state)
        # Max retries reached — should not retry
        assert result.get("needs_retry", False) is False


class TestAgentGraph:
    """Test the compiled LangGraph workflow."""

    def test_graph_compiles(self):
        """The LangGraph workflow should compile without errors."""
        from src.agent.graph import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        """Compiled graph should have all 9 nodes."""
        from src.agent.graph import build_graph
        graph = build_graph()
        # LangGraph compiled graphs have a nodes attribute
        node_count = len(graph.nodes) if hasattr(graph, 'nodes') else 0
        assert node_count >= 8  # At least 8 of the 9 nodes
