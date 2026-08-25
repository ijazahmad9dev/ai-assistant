from app.graph.build_graph import build_crag_graph

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_crag_graph()
    return _graph


def run_graph_query(question: str, session_id: str) -> dict:
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
    }

    config = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(initial_state, config=config)

    return {
        "answer": final_state.get("generation") or "No answer generated.",
        "retrieval_mode": final_state.get("retrieval_mode", "vector"),
        "retry_count": final_state.get("retry_count", 0),
        "route": final_state.get("route"),
    }