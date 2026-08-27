# test_langsmith.py
from app.ingestion.run_ingestion import get_or_build_retriever
from app.api import state
state.retriever = get_or_build_retriever()

from app.graph.build_graph import build_crag_graph

graph = build_crag_graph()
result = graph.invoke(
    {
        "question": "who is the CEO of NextBridge",
        "original_question": "who is the CEO of NextBridge",
        "documents": [], "generation": None, "best_generation": None,
        "retrieval_mode": "vector", "grade": None, "retry_count": 0, "route": None,
    },
    config={"configurable": {"thread_id": "langsmith-test"}},
)
print(result["generation"])