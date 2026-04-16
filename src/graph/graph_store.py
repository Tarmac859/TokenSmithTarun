"""
graph_store.py

SQLite-backed knowledge graph store.

Schema
------
    entities : id, name
    edges    : id, head_id, relation, tail_id, chunk_id

Each edge carries a foreign key back to its source chunk so that
graph traversal returns chunk IDs directly to the retrieval pipeline,
mirroring how IndexKeywordRetriever maps page numbers to chunk IDs.
"""

import sqlite3
from pathlib import Path
from typing import Set


class GraphStore:
    def __init__(self, db_path: str = "index/graph/knowledge_graph.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL COLLATE NOCASE,
                    UNIQUE(name)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    head_id  INTEGER NOT NULL,
                    relation TEXT    NOT NULL,
                    tail_id  INTEGER NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    FOREIGN KEY(head_id) REFERENCES entities(id),
                    FOREIGN KEY(tail_id) REFERENCES entities(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_head ON edges(head_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_tail ON edges(tail_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add_triple(self, head: str, relation: str, tail: str, chunk_id: int):
        """Insert a (head, relation, tail) triple linked to chunk_id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO entities(name) VALUES (?)", (head.strip(),)
            )
            head_id = conn.execute(
                "SELECT id FROM entities WHERE name=?", (head.strip(),)
            ).fetchone()[0]

            conn.execute(
                "INSERT OR IGNORE INTO entities(name) VALUES (?)", (tail.strip(),)
            )
            tail_id = conn.execute(
                "SELECT id FROM entities WHERE name=?", (tail.strip(),)
            ).fetchone()[0]

            conn.execute(
                "INSERT INTO edges(head_id, relation, tail_id, chunk_id) VALUES (?,?,?,?)",
                (head_id, relation.strip(), tail_id, chunk_id),
            )

    def get_chunk_ids_for_entity(self, entity_name: str, hops: int = 2) -> Set[int]:
        """
        BFS traversal up to `hops` from any entity whose name contains
        entity_name (case-insensitive), collecting all linked chunk IDs.
        """
        chunk_ids: Set[int] = set()
        with sqlite3.connect(self.db_path) as conn:
            seed_rows = conn.execute(
                "SELECT id FROM entities WHERE name LIKE ?",
                (f"%{entity_name}%",),
            ).fetchall()
            if not seed_rows:
                return chunk_ids

            visited: Set[int] = set()
            frontier: Set[int] = {row[0] for row in seed_rows}

            for _ in range(hops):
                if not frontier:
                    break
                visited |= frontier
                placeholders = ",".join("?" * len(frontier))
                frontier_list = list(frontier)
                edges = conn.execute(
                    f"""SELECT head_id, tail_id, chunk_id FROM edges
                        WHERE head_id IN ({placeholders})
                           OR tail_id IN ({placeholders})""",
                    frontier_list + frontier_list,
                ).fetchall()

                new_frontier: Set[int] = set()
                for head_id, tail_id, chunk_id in edges:
                    chunk_ids.add(chunk_id)
                    if head_id not in visited:
                        new_frontier.add(head_id)
                    if tail_id not in visited:
                        new_frontier.add(tail_id)
                frontier = new_frontier

        return chunk_ids

    def entity_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def edge_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
