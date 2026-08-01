"""Tests for src/graph/store.py — Knowledge graph operations."""

import pytest
from src.graph.store import GraphStore


class TestNetworkXGraphStore:
    """Test the NetworkX fallback graph store."""

    @pytest.fixture(autouse=True)
    def setup_store(self, temp_dir):
        """Create a fresh graph store for each test."""
        self.store = GraphStore(persist_path=str(temp_dir / "test_graph.json"))

    def test_add_entity(self):
        """Should add an entity node to the graph."""
        self.store.add_entity("Machine Learning", "CONCEPT", "A subset of AI")
        stats = self.store.get_stats()
        assert stats["node_count"] >= 1

    def test_add_relation(self):
        """Should add a relationship between entities."""
        self.store.add_entity("Machine Learning", "CONCEPT", "")
        self.store.add_entity("Neural Network", "TECHNOLOGY", "")
        self.store.add_relation("Machine Learning", "Neural Network", "USES", "ML uses NNs")
        stats = self.store.get_stats()
        assert stats["edge_count"] >= 1

    def test_query_entity(self):
        """Should retrieve entity and its relationships."""
        self.store.add_entity("Python", "TECHNOLOGY", "Programming language")
        self.store.add_entity("Machine Learning", "CONCEPT", "")
        self.store.add_relation("Machine Learning", "Python", "USES", "")
        result = self.store.query_entity("Python")
        assert result is not None

    def test_fuzzy_search(self):
        """Should find entities with partial name matches."""
        self.store.add_entity("Machine Learning", "CONCEPT", "")
        self.store.add_entity("Deep Learning", "CONCEPT", "")
        results = self.store.search_entities("Learning")
        assert len(results) >= 2

    def test_get_neighbors(self):
        """Should return 1-hop neighbors of an entity."""
        self.store.add_entity("Python", "TECHNOLOGY", "")
        self.store.add_entity("Machine Learning", "CONCEPT", "")
        self.store.add_entity("Data Science", "CONCEPT", "")
        self.store.add_relation("Python", "Machine Learning", "USED_IN", "")
        self.store.add_relation("Python", "Data Science", "USED_IN", "")
        neighbors = self.store.get_neighbors("Python")
        assert len(neighbors) >= 2

    def test_graph_stats(self):
        """Should return accurate graph statistics."""
        self.store.add_entity("A", "CONCEPT", "")
        self.store.add_entity("B", "CONCEPT", "")
        self.store.add_entity("C", "CONCEPT", "")
        self.store.add_relation("A", "B", "RELATED_TO", "")
        self.store.add_relation("B", "C", "RELATED_TO", "")
        stats = self.store.get_stats()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2

    def test_clear_graph(self):
        """Should clear all nodes and edges."""
        self.store.add_entity("A", "CONCEPT", "")
        self.store.add_entity("B", "CONCEPT", "")
        self.store.add_relation("A", "B", "RELATED_TO", "")
        self.store.clear()
        stats = self.store.get_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0

    def test_duplicate_entity_merge(self):
        """Adding same entity twice should merge, not duplicate."""
        self.store.add_entity("Python", "TECHNOLOGY", "Language v1")
        self.store.add_entity("Python", "TECHNOLOGY", "Language v2")
        stats = self.store.get_stats()
        assert stats["node_count"] == 1

    def test_visualization_data(self):
        """Should export data for visualization."""
        self.store.add_entity("A", "CONCEPT", "Test A")
        self.store.add_entity("B", "TECHNOLOGY", "Test B")
        self.store.add_relation("A", "B", "USES", "")
        vis_data = self.store.get_visualization_data()
        assert "nodes" in vis_data
        assert "edges" in vis_data
        assert len(vis_data["nodes"]) >= 2
        assert len(vis_data["edges"]) >= 1

    def test_persistence(self, temp_dir):
        """Graph should persist to disk and reload."""
        path = str(temp_dir / "persist_test.json")
        store1 = GraphStore(persist_path=path)
        store1.add_entity("Persistent", "CONCEPT", "")
        store1.save()

        store2 = GraphStore(persist_path=path)
        stats = store2.get_stats()
        assert stats["node_count"] >= 1
