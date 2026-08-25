# test_crag_graph.py
from app.ingestion.run_ingestion import get_or_build_retriever
from app.api import state

# Populate state.retriever before the graph runs — normally main_api.py does this at startup
state.retriever = get_or_build_retriever()

from app.graph.build_graph import build_crag_graph

graph = build_crag_graph()

initial_state = {
    "question": "who is the CEO of NextBridge",
    "original_question": "who is the CEO of NextBridge",
    "documents": [],
    "generation": None,
    "retrieval_mode": "vector",
    "grade": None,
    "retry_count": 0,
    "route": None,
}

config = {"configurable": {"thread_id": "test-1"}}

for step in graph.stream(initial_state, config=config):
    node_name = list(step.keys())[0]
    print(f"\n=== NODE: {node_name} ===")
    print(step[node_name])