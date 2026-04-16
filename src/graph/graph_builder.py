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
from pathlib import Path
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

    prompt = _TRIPLE_PROMPT.format(text=text[:900])   # cap input length
    try:
        result = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["\n\n", "```", "Note"],
        )
        raw = result["choices"][0]["text"].strip()

        # Pull out the first JSON array in the response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []

        parsed = json.loads(match.group())
        triples = []
        for item in parsed:
            if isinstance(item, list) and len(item) == 3:
                h, r, t = str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()
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
    model = Llama(
        model_path=gen_model_path,
        n_ctx=n_ctx,
        verbose=False,
        n_gpu_layers=n_gpu_layers,
    )

    print(f"Building graph index for {len(chunks):,} chunks...")
    total_triples = 0

    for chunk_id, chunk_text in enumerate(tqdm(chunks, desc="Extracting triples")):
        triples = extract_triples(chunk_text, model)
        for head, relation, tail in triples:
            graph_store.add_triple(head, relation, tail, chunk_id)
        total_triples += len(triples)

    print(
        f"\nGraph index complete: "
        f"{graph_store.entity_count():,} entities, "
        f"{graph_store.edge_count():,} edges "
        f"({total_triples:,} triples extracted)"
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
    build_graph_index(chunks, gen_model_path=gen_model, graph_store=store)


if __name__ == "__main__":
    main()
