from app.graph.build_graph import build_crag_graph
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

    result = {
        "answer": final_state.get("best_generation") or final_state.get("generation") or "No answer generated.",
        "retrieval_mode": final_state.get("retrieval_mode", "vector"),
        "retry_count": final_state.get("retry_count", 0),
        "route": final_state.get("route"),
        "blocked": final_state.get("blocked", False),
        "output_blocked": final_state.get("output_blocked", False),
        "cache_hit": False,
    }

    # Don't cache blocked requests or failed generations -- a guardrail
    # verdict is question-specific, and "No answer generated" isn't a
    # result worth serving to the next similar question.
    if not result["blocked"] and result["answer"] != "No answer generated.":
        semantic_cache.store(question, result)

    return result