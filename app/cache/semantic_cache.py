"""Basic in-memory semantic cache for /query/graph. ... (docstring unchanged)"""
import threading
import numpy as np
from langsmith import traceable

from app.embeddings.embedder import get_embedding_model

SIMILARITY_THRESHOLD = 0.92  # cosine similarity; tune against real traffic

_lock = threading.Lock()
_entries = []  # [{"embedding": np.ndarray, "question": str, "result": dict}]
_embedding_model = None


def _get_embedder():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@traceable(name="semantic_cache_lookup", run_type="tool")
def lookup(question: str):
    """Returns the cached result dict for the closest sufficiently-similar
    past question, or None if there's no hit above SIMILARITY_THRESHOLD."""
    if not _entries:
        return None

    query_vec = np.array(_get_embedder().embed_query(question))

    with _lock:
        best_score, best_entry = -1.0, None
        for entry in _entries:
            score = _cosine_sim(query_vec, entry["embedding"])
            if score > best_score:
                best_score, best_entry = score, entry

    if best_entry is not None and best_score >= SIMILARITY_THRESHOLD:
        return {**best_entry["result"], "cache_similarity": best_score}
    return None


@traceable(name="semantic_cache_store", run_type="tool")
def store(question: str, result: dict):
    query_vec = np.array(_get_embedder().embed_query(question))
    with _lock:
        _entries.append({"embedding": query_vec, "question": question, "result": result})