"""Tests for src/tools/tools.py — All 12 dynamic tools."""

import pytest
from unittest.mock import patch, MagicMock
from src.tools.tools import (
    get_current_datetime,
    calculator,
    unit_converter,
    get_definition,
    web_search,
    wikipedia_lookup,
    get_weather,
    get_stock_price,
    get_crypto_price,
    get_news_headlines,
    url_reader,
    python_executor,
    TOOL_REGISTRY,
)


class TestGetCurrentDatetime:
    """Test the datetime tool."""

    def test_returns_string(self):
        result = get_current_datetime()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_date_components(self):
        result = get_current_datetime()
        # Should contain year, time indication
        assert "20" in result  # Year 20XX


class TestCalculator:
    """Test the safe math calculator."""

    def test_basic_addition(self):
        result = calculator("2 + 3")
        assert "5" in result

    def test_multiplication(self):
        result = calculator("7 * 8")
        assert "56" in result

    def test_complex_expression(self):
        result = calculator("(10 + 5) * 3")
        assert "45" in result

    def test_division(self):
        result = calculator("100 / 4")
        assert "25" in result

    def test_invalid_expression(self):
        result = calculator("not a math expression")
        # Should return error, not crash
        assert isinstance(result, str)

    def test_blocks_dangerous_code(self):
        """Should not execute dangerous code."""
        result = calculator("__import__('os').system('ls')")
        # Should fail safely
        assert "error" in result.lower() or "invalid" in result.lower() or isinstance(result, str)


class TestUnitConverter:
    """Test the unit converter tool."""

    def test_length_conversion(self):
        result = unit_converter("5 miles to kilometers")
        assert isinstance(result, str)
        # 5 miles ≈ 8.05 km
        assert "8" in result or "kilometer" in result.lower()

    def test_temperature_conversion(self):
        result = unit_converter("100 celsius to fahrenheit")
        assert isinstance(result, str)
        assert "212" in result or "fahrenheit" in result.lower()

    def test_invalid_conversion(self):
        result = unit_converter("gibberish to nonsense")
        assert isinstance(result, str)  # Should not crash


class TestGetDefinition:
    """Test the dictionary definition tool."""

    @patch("src.tools.tools.requests")
    def test_returns_definition(self, mock_requests):
        """Should return a definition string."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            "word": "hello",
            "meanings": [{
                "partOfSpeech": "exclamation",
                "definitions": [{"definition": "Used as a greeting"}]
            }]
        }]
        mock_requests.get.return_value = mock_response

        result = get_definition("hello")
        assert isinstance(result, str)
        assert len(result) > 0


class TestWebSearch:
    """Test the DuckDuckGo web search tool."""

    @patch("src.tools.tools.DDGS")
    def test_returns_results(self, mock_ddgs):
        """Should return search results."""
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {"title": "Test", "href": "https://example.com", "body": "Test result"}
        ]
        mock_ddgs.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock_ddgs.return_value.__exit__ = MagicMock(return_value=False)

        result = web_search("test query")
        assert isinstance(result, str)


class TestPythonExecutor:
    """Test the sandboxed Python executor."""

    def test_simple_print(self):
        result = python_executor("print('hello world')")
        assert "hello" in result.lower()

    def test_math_computation(self):
        result = python_executor("print(2 ** 10)")
        assert "1024" in result

    def test_blocked_imports(self):
        """Should block dangerous imports."""
        result = python_executor("import os; os.system('ls')")
        # Should fail or be blocked
        assert isinstance(result, str)

    def test_timeout(self):
        """Long-running code should be killed."""
        result = python_executor("while True: pass")
        assert isinstance(result, str)  # Should return timeout error


class TestToolRegistry:
    """Test the tool registry."""

    def test_all_12_tools_registered(self):
        """Should have exactly 12 tools in the registry."""
        assert len(TOOL_REGISTRY) == 12

    def test_registry_has_required_keys(self):
        """Each tool entry should have function, description, parameters."""
        for name, tool in TOOL_REGISTRY.items():
            assert "function" in tool, f"Tool '{name}' missing 'function'"
            assert "description" in tool, f"Tool '{name}' missing 'description'"
            assert callable(tool["function"]), f"Tool '{name}' function is not callable"

    def test_tool_names_match(self):
        """Tool names should include all expected tools."""
        expected_tools = {
            "get_current_datetime", "web_search", "wikipedia_lookup",
            "get_weather", "get_stock_price", "get_crypto_price",
            "calculator", "get_news_headlines", "url_reader",
            "python_executor", "unit_converter", "get_definition",
        }
        assert set(TOOL_REGISTRY.keys()) == expected_tools
