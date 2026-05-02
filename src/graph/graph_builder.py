"""
graph_builder.py

Builds the knowledge graph index during the ingestion phase.

For each text chunk, prompts the local LLM to extract
(entity, relation, entity) triples and stores them in GraphStore.

Usage (standalone):
    python -m src.graph.graph_builder

Or call build_graph_index() from a custom script.
"""

import json
import pickle
import re
from typing import List, Tuple

from tqdm import tqdm
from llama_cpp import Llama

from src.graph.graph_store import GraphStore



_TRIPLE_PROMPT = """\
Extract knowledge graph triples from the database systems textbook passage below.
Output ONLY a JSON array of triples: [["entity1", "relation", "entity2"], ...]
Rules:
- Focus on database concepts, algorithms, properties, and their causal/structural relationships.
- Each entity should be a concise noun phrase (e.g. "ARIES", "write-ahead log", "steal policy").
- Each relation should be a short verb phrase (e.g. "requires", "enables", "is part of").
- Extract 3 to 7 triples. If no clear triples exist return [].
- Output ONLY the JSON array, nothing else.

Passage:
{text}

JSON:"""



def extract_triples(
    text: str,
    model: Llama,
    max_tokens: int = 256,
) -> List[Tuple[str, str, str]]:

    prompt = _TRIPLE_PROMPT.format(text=text[:900])
    try:
        result = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["```", "Note:"],
        )
        raw = result["choices"][0]["text"].strip()

        # The model sometimes prepends "[] " before the real array, or uses
        # dict format {"entity1":..., "relation":..., "entity2":...} instead
        # of arrays. Scan every '[' position and take the first non-empty parse.
        parsed = None
        for m in re.finditer(r"\[", raw):
            try:
                candidate = json.loads(raw[m.start():])
                if candidate:
                    parsed = candidate
                    break
            except json.JSONDecodeError:
                continue

        if not parsed:
            return []

        triples = []
        for item in parsed:
            if isinstance(item, list) and len(item) == 3:
                h, r, t = str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()
                if h and r and t:
                    triples.append((h, r, t))
            elif isinstance(item, dict):
                h = str(item.get("entity1", item.get("head", ""))).strip()
                r = str(item.get("relation", "")).strip()
                t = str(item.get("entity2", item.get("tail", ""))).strip()
                if h and r and t:
                    triples.append((h, r, t))
        return triples
    except Exception:
        return []


def build_graph_index(
    chunks: List[str],
    gen_model_path: str,
    graph_store: GraphStore,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
) -> None:
    import sys
    model = Llama(
        model_path=gen_model_path,
        n_ctx=n_ctx,
        verbose=False,
        n_gpu_layers=n_gpu_layers,
    )

    # Single query to get all already-processed chunk IDs
    done_ids = graph_store.get_processed_chunk_ids()
    already_done = len(done_ids)
    print(f"Building graph index for {len(chunks):,} chunks "
          f"({already_done:,} already processed, resuming)...", flush=True)
    total_triples = 0
    skipped = 0

    for chunk_id, chunk_text in enumerate(tqdm(chunks, desc="Extracting triples", file=sys.stdout)):
        if chunk_id in done_ids:
            skipped += 1
            continue
        triples = extract_triples(chunk_text, model)
        graph_store.add_triples_batch(triples, chunk_id)
        graph_store.mark_processed(chunk_id)
        total_triples += len(triples)

    print(
        f"\nGraph index complete: "
        f"{graph_store.entity_count():,} entities, "
        f"{graph_store.edge_count():,} edges "
        f"({total_triples:,} new triples, {skipped:,} chunks skipped)",
        flush=True,
    )


def main():
    import pathlib

    chunks_path = pathlib.Path("index/sections/textbook_index_chunks.pkl")
    db_path     = "index/graph/knowledge_graph.db"
    gen_model   = "models/qwen2.5-3b-instruct-q8_0.gguf"

    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run 'make run-index' first.")
        return

    chunks = pickle.load(open(chunks_path, "rb"))
    store  = GraphStore(db_path=db_path)
    # n_gpu_layers=0 forces CPU-only — avoids Vulkan driver crashes on long runs
    build_graph_index(chunks, gen_model_path=gen_model, graph_store=store, n_gpu_layers=0)


if __name__ == "__main__":
    main()
