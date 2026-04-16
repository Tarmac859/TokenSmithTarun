"""
graph_retriever.py

Query-time graph traversal retriever.

Plugs into the existing EnsembleRanker exactly like IndexKeywordRetriever:
  - Inherits from Retriever ABC
  - name = "graph"  (matches ranker_weights key)
  - get_scores() returns Dict[chunk_id, float]

Entity extraction follows the same stopword + lemmatization pattern
used in IndexKeywordRetriever._extract_keywords().
"""

import re
from typing import Dict, List, Set

import nltk
from nltk.stem import WordNetLemmatizer

from src.retriever import Retriever
from src.graph.graph_store import GraphStore

_STOPWORDS = {
    "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
    "to", "of", "by", "with", "that", "this", "it", "as", "are", "was",
    "what", "how", "why", "when", "where", "who", "does", "do", "be",
    "between", "difference", "explain", "describe", "give", "show",
}


def _extract_entities(query: str) -> List[str]:
    nltk.download("wordnet", quiet=True)
    lemmatizer = WordNetLemmatizer()

    tokens = re.findall(r"[a-zA-Z0-9_+\-']+", query.lower())

    def lemmatize(word: str) -> str:
        lemma = lemmatizer.lemmatize(word, pos="n")
        return lemma if lemma != word else lemmatizer.lemmatize(word, pos="v")

    clean = [lemmatize(t) for t in tokens if t not in _STOPWORDS and len(t) > 2]

    # Also build bigrams for compound terms ("write ahead", "two phase")
    bigrams = [f"{clean[i]} {clean[i+1]}" for i in range(len(clean) - 1)]

    return clean + bigrams


class GraphRetriever(Retriever):

    name = "graph"

    def __init__(self, graph_store: GraphStore, hops: int = 2):
        self.store = graph_store
        self.hops = hops

    def get_scores(
        self, query: str, pool_size: int, chunks: List[str]
    ) -> Dict[int, float]:
        entities = _extract_entities(query)
        chunk_ids: Set[int] = set()
        for entity in entities:
            chunk_ids |= self.store.get_chunk_ids_for_entity(entity, hops=self.hops)

        return {
            chunk_id: 1.0
            for chunk_id in chunk_ids
            if 0 <= chunk_id < len(chunks)
        }
