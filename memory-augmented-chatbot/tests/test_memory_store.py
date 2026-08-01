"""Tests for src/memory/store.py — Memory store CRUD operations."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch


class TestMemoryStore:
    """Test SQLite-based memory store (the fallback that works without PostgreSQL)."""

    @pytest.fixture(autouse=True)
    def setup_store(self, temp_dir):
        """Create a fresh memory store for each test."""
        db_path = temp_dir / "test_memory.db"
        with patch.dict(os.environ, {"POSTGRES_DSN": ""}):
            from src.memory.store import MemoryStore
            self.store = MemoryStore(db_path=str(db_path))

    def test_store_and_retrieve_fact(self):
        """Should store a fact and retrieve it."""
        self.store.add_fact("user1", "User likes Python")
        facts = self.store.get_facts("user1")
        assert any("Python" in f for f in facts)

    def test_store_multiple_facts(self):
        """Should store and retrieve multiple facts."""
        self.store.add_fact("user1", "User likes Python")
        self.store.add_fact("user1", "User works at Google")
        self.store.add_fact("user1", "User is from India")
        facts = self.store.get_facts("user1")
        assert len(facts) >= 3

    def test_facts_isolated_by_user(self):
        """Different users should have separate facts."""
        self.store.add_fact("user1", "User likes Python")
        self.store.add_fact("user2", "User likes JavaScript")
        facts1 = self.store.get_facts("user1")
        facts2 = self.store.get_facts("user2")
        assert any("Python" in f for f in facts1)
        assert not any("JavaScript" in f for f in facts1)
        assert any("JavaScript" in f for f in facts2)

    def test_duplicate_fact_not_added(self):
        """Duplicate facts should be deduplicated."""
        self.store.add_fact("user1", "User likes Python")
        self.store.add_fact("user1", "User likes Python")  # Duplicate
        facts = self.store.get_facts("user1")
        python_facts = [f for f in facts if "Python" in f]
        assert len(python_facts) == 1

    def test_add_chat_history(self):
        """Should store and retrieve chat history."""
        self.store.add_message("user1", "user", "Hello!")
        self.store.add_message("user1", "assistant", "Hi there!")
        history = self.store.get_history("user1")
        assert len(history) >= 2

    def test_chat_history_order(self):
        """Chat history should be in chronological order."""
        self.store.add_message("user1", "user", "First message")
        self.store.add_message("user1", "assistant", "First response")
        self.store.add_message("user1", "user", "Second message")
        history = self.store.get_history("user1")
        assert len(history) >= 3

    def test_clear_memory(self):
        """Should clear all facts for a user."""
        self.store.add_fact("user1", "User likes Python")
        self.store.add_fact("user1", "User works at Google")
        self.store.clear_facts("user1")
        facts = self.store.get_facts("user1")
        assert len(facts) == 0

    def test_clear_memory_doesnt_affect_other_users(self):
        """Clearing one user's memory shouldn't affect others."""
        self.store.add_fact("user1", "User likes Python")
        self.store.add_fact("user2", "User likes JavaScript")
        self.store.clear_facts("user1")
        facts2 = self.store.get_facts("user2")
        assert len(facts2) >= 1

    def test_empty_user_returns_empty(self):
        """Non-existent user should return empty lists."""
        facts = self.store.get_facts("nonexistent_user")
        assert facts == [] or len(facts) == 0
        history = self.store.get_history("nonexistent_user")
        assert history == [] or len(history) == 0
