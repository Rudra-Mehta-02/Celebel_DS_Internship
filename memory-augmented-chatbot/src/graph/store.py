"""
Graph store — dual-backend: Neo4j (production) + NetworkX (fallback).

Features:
  - MERGE-based upsert (deduplication)
  - Fuzzy entity search
  - 1-hop and 2-hop relationship queries
  - Graph statistics and analytics
  - JSON export for visualisation
"""

from __future__ import annotations

import abc
import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


class BaseGraphStore(abc.ABC):
    """Abstract graph store interface."""

    @abc.abstractmethod
    def add_entities(self, entities: list[dict]) -> int: ...

    @abc.abstractmethod
    def add_relations(self, relations: list[dict]) -> int: ...

    @abc.abstractmethod
    def search_entity(self, name: str, limit: int = 5) -> list[dict]: ...

    @abc.abstractmethod
    def get_entity_relations(self, entity_name: str, hops: int = 1) -> list[str]: ...

    @abc.abstractmethod
    def get_stats(self) -> dict: ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    def add_extraction(self, extraction: dict) -> None:
        """Add entities and relations from an extraction result."""
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])
        self.add_entities(entities)
        self.add_relations(relations)


# ── Neo4j Backend ────────────────────────────────────────────

class Neo4jGraphStore(BaseGraphStore):
    """Neo4j graph database backend with Cypher queries."""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        # Test connection
        with self._driver.session() as session:
            session.run("RETURN 1")
        logger.info("✅ Neo4j connected: %s", uri)

    def add_entities(self, entities: list[dict]) -> int:
        count = 0
        with self._driver.session() as session:
            for e in entities:
                name = e.get("name", "").strip()
                etype = re.sub(r"[^A-Za-z0-9_]", "_", e.get("type", "CONCEPT"))
                desc = e.get("description", "")
                if not name:
                    continue
                session.run(
                    f"MERGE (n:{etype} {{name: $name}}) "
                    "SET n.description = COALESCE(n.description, $desc)",
                    name=name, desc=desc,
                )
                count += 1
        return count

    def add_relations(self, relations: list[dict]) -> int:
        count = 0
        with self._driver.session() as session:
            for r in relations:
                src = r.get("source", "").strip()
                tgt = r.get("target", "").strip()
                rtype = re.sub(r"[^A-Z0-9_]", "_", r.get("type", "RELATED_TO").upper())
                desc = r.get("description", "")
                if not src or not tgt:
                    continue
                # Use parameterised query — safe from injection
                session.run(
                    f"MERGE (a {{name: $src}}) "
                    f"MERGE (b {{name: $tgt}}) "
                    f"MERGE (a)-[r:{rtype}]->(b) "
                    "SET r.description = COALESCE(r.description, $desc)",
                    src=src, tgt=tgt, desc=desc,
                )
                count += 1
        return count

    def search_entity(self, name: str, limit: int = 5) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) "
                "RETURN n.name AS name, labels(n) AS types, n.description AS description "
                "ORDER BY CASE WHEN toLower(n.name) = toLower($name) THEN 0 ELSE 1 END "
                "LIMIT $limit",
                name=name, limit=limit,
            )
            return [dict(r) for r in result]

    def get_entity_relations(self, entity_name: str, hops: int = 1) -> list[str]:
        with self._driver.session() as session:
            if hops == 1:
                result = session.run(
                    "MATCH (a)-[r]->(b) "
                    "WHERE toLower(a.name) CONTAINS toLower($name) "
                    "RETURN a.name + ' -[' + type(r) + ']-> ' + b.name AS fact "
                    "LIMIT 20",
                    name=entity_name,
                )
            else:
                result = session.run(
                    "MATCH (a)-[r1]->(b)-[r2]->(c) "
                    "WHERE toLower(a.name) CONTAINS toLower($name) "
                    "RETURN a.name + ' -[' + type(r1) + ']-> ' + b.name + ' -[' + type(r2) + ']-> ' + c.name AS fact "
                    "LIMIT 20",
                    name=entity_name,
                )
            return [r["fact"] for r in result]

    def get_stats(self) -> dict:
        with self._driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            return {"nodes": nodes, "edges": edges, "backend": "neo4j"}

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared Neo4j graph")

    def close(self) -> None:
        self._driver.close()


# ── NetworkX Backend ─────────────────────────────────────────

class NetworkXGraphStore(BaseGraphStore):
    """Local NetworkX graph with JSON persistence."""

    def __init__(self, filepath: Optional[str] = None):
        import networkx as nx
        self.filepath = filepath or get_settings().networkx_path
        self.graph = nx.DiGraph()
        self._load()
        logger.info("✅ NetworkX graph loaded: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())

    def _load(self) -> None:
        path = Path(self.filepath)
        if path.exists():
            try:
                import networkx as nx
                data = json.loads(path.read_text(encoding="utf-8"))
                self.graph = nx.node_link_graph(data)
            except Exception as e:
                logger.warning("Failed to load graph: %s", e)

    def _save(self) -> None:
        import networkx as nx
        path = Path(self.filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_entities(self, entities: list[dict]) -> int:
        count = 0
        for e in entities:
            name = e.get("name", "").strip()
            if not name:
                continue
            self.graph.add_node(
                name,
                type=e.get("type", "CONCEPT"),
                description=e.get("description", ""),
            )
            count += 1
        self._save()
        return count

    def add_relations(self, relations: list[dict]) -> int:
        count = 0
        for r in relations:
            src = r.get("source", "").strip()
            tgt = r.get("target", "").strip()
            if not src or not tgt:
                continue
            # Ensure nodes exist
            if not self.graph.has_node(src):
                self.graph.add_node(src, type="CONCEPT", description="")
            if not self.graph.has_node(tgt):
                self.graph.add_node(tgt, type="CONCEPT", description="")
            self.graph.add_edge(
                src, tgt,
                type=r.get("type", "RELATED_TO"),
                description=r.get("description", ""),
            )
            count += 1
        self._save()
        return count

    def search_entity(self, name: str, limit: int = 5) -> list[dict]:
        name_lower = name.lower()
        results = []
        for node, data in self.graph.nodes(data=True):
            if name_lower in node.lower():
                results.append({
                    "name": node,
                    "types": [data.get("type", "CONCEPT")],
                    "description": data.get("description", ""),
                })
        # Sort: exact matches first
        results.sort(key=lambda x: 0 if x["name"].lower() == name_lower else 1)
        return results[:limit]

    def get_entity_relations(self, entity_name: str, hops: int = 1) -> list[str]:
        name_lower = entity_name.lower()
        # Find matching node(s)
        matching_nodes = [
            n for n in self.graph.nodes()
            if name_lower in n.lower()
        ]

        facts = []
        for node in matching_nodes:
            for _, target, data in self.graph.out_edges(node, data=True):
                rel_type = data.get("type", "RELATED_TO")
                facts.append(f"{node} -[{rel_type}]-> {target}")

                if hops >= 2:
                    for _, target2, data2 in self.graph.out_edges(target, data=True):
                        rel_type2 = data2.get("type", "RELATED_TO")
                        facts.append(f"{node} -[{rel_type}]-> {target} -[{rel_type2}]-> {target2}")

        return facts[:20]

    def get_stats(self) -> dict:
        import networkx as nx
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "backend": "networkx",
        }
        if self.graph.number_of_nodes() > 0:
            stats["density"] = round(nx.density(self.graph), 4)
            # Top entities by degree
            degrees = sorted(
                self.graph.degree(), key=lambda x: x[1], reverse=True
            )[:10]
            stats["top_entities"] = [{"name": n, "connections": d} for n, d in degrees]
        return stats

    def clear(self) -> None:
        self.graph.clear()
        self._save()
        logger.info("Cleared NetworkX graph")

    def get_visualization_data(self) -> dict:
        """Export graph data for pyvis visualisation."""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                "group": data.get("type", "CONCEPT"),
                "title": data.get("description", node),
            })

        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append({
                "from": src,
                "to": tgt,
                "label": data.get("type", ""),
                "title": data.get("description", ""),
            })

        return {"nodes": nodes, "edges": edges}


# ── Factory ──────────────────────────────────────────────────
_store: Optional[BaseGraphStore] = None


def get_graph_store() -> BaseGraphStore:
    """Return the graph store — Neo4j if configured, else NetworkX."""
    global _store
    if _store is None:
        settings = get_settings()
        if settings.has_neo4j:
            try:
                _store = Neo4jGraphStore(
                    uri=settings.neo4j_uri,
                    user=settings.neo4j_user,
                    password=settings.neo4j_password,
                )
            except Exception as e:
                logger.warning("Neo4j connection failed: %s — using NetworkX", e)
                _store = NetworkXGraphStore()
        else:
            _store = NetworkXGraphStore()
    return _store
