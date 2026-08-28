from app.graph.build_graph import build_crag_graph
from app.graph.nodes import _top_docs_for
from app.cache import semantic_cache
from langsmith import traceable

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_crag_graph()
    return _graph


@traceable(name="query_graph_request", run_type="chain")
def run_graph_query(question: str, session_id: str) -> dict:
    cached = semantic_cache.lookup(question)
    if cached is not None:
        return {**cached, "cache_hit": True}

    graph = get_graph()

    initial_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": None,
        "best_generation": None,
        "retrieval_mode": "vector",
        "grade": None,
        "retry_count": 0,
        "route": None,
        "sub_questions": [],
        "uncovered_sub_questions": [],
        "blocked": False,
        "block_reason": None,
        "original_sub_questions": [],
        "output_blocked": False,
        "unsupported_claims": [],
    }

    config = {
        "configurable": {"thread_id": session_id},
        "tags": ["crag-graph"],
        "metadata": {"pipeline": "crag"},
    }
    final_state = graph.invoke(initial_state, config=config)

    # Re-derive the EXACT context generate() used, via the same deterministic
    # rerank _top_docs_for that generate() itself calls — no new state field
    # needed, since documents is never mutated again after generate() runs.
    n = 8 if final_state.get("route") == "complex" else 5
    sub_qs = final_state.get("sub_questions") or [question]
    top_docs = _top_docs_for(sub_qs, final_state.get("documents") or [], n)

    result = {
        "answer": final_state.get("best_generation") or final_state.get("generation") or "No answer generated.",
        "retrieval_mode": final_state.get("retrieval_mode", "vector"),
        "retry_count": final_state.get("retry_count", 0),
        "route": final_state.get("route"),
        "blocked": final_state.get("blocked", False),
        "output_blocked": final_state.get("output_blocked", False),
        "documents": [
            {"page_content": d.page_content, "metadata": d.metadata} for d in top_docs
        ],
        "cache_hit": False,
    }

    if not result["blocked"] and result["answer"] != "No answer generated.":
        semantic_cache.store(question, result)

    return result