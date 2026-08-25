from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import GraphState
from app.graph.nodes import (
    route_question, retrieve, grade_documents, transform_query,
    web_search_node, generate, grade_generation, finalize, fast_generate,
)
from app.graph.edges import (
    route_after_classification, route_after_grading, route_after_generation_grade, route_after_fast_generate,
)

_checkpointer = MemorySaver()


def build_crag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)
    workflow.add_node("finalize", finalize)
    workflow.add_node("fast_generate", fast_generate)

    workflow.set_entry_point("route_question")

    # TRUE adaptive routing — branches to a different node path based on
    # complexity classification, not just different parameters within nodes.
    workflow.add_conditional_edges(
        "route_question",
        route_after_classification,
        {"fast_path": "fast_generate", "full_path": "retrieve"},
    )

    # Fast path: simple questions skip grading/correction entirely.
    workflow.add_conditional_edges(
        "fast_generate",
        route_after_fast_generate,
        {"end": END, "retrieve": "retrieve"},
    )

    # Full CRAG + Self-RAG path: complex questions.
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "transform_query": "transform_query", "web_search": "web_search"},
    )
    workflow.add_edge("transform_query", "retrieve")   # CRAG loop
    workflow.add_edge("web_search", "generate")

    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_generation_grade,
        {"end": END, "transform_query": "transform_query", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=_checkpointer)