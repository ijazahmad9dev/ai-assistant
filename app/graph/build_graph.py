from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import GraphState
from app.graph.nodes import (
    input_guardrail, blocked_response, route_question, decompose_question,
    retrieve, grade_documents, transform_query, web_search_node, generate,
    grade_generation, finalize, fast_generate, output_guardrail,
)
from app.graph.edges import (
    route_after_input_guardrail, route_after_classification,
    route_after_grading, route_after_generation_grade, route_after_fast_generate,
)

_checkpointer = MemorySaver()


def build_crag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("input_guardrail", input_guardrail)
    workflow.add_node("blocked_response", blocked_response)
    workflow.add_node("route_question", route_question)
    workflow.add_node("decompose_question", decompose_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)
    workflow.add_node("finalize", finalize)
    workflow.add_node("fast_generate", fast_generate)
    workflow.add_node("output_guardrail", output_guardrail)

    # Every request enters through the input guardrail first — nothing reaches
    # retrieval, tools, or an LLM generation call over real context otherwise.
    workflow.set_entry_point("input_guardrail")
    workflow.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"blocked": "blocked_response", "allowed": "route_question"},
    )
    workflow.add_edge("blocked_response", END)

    workflow.add_conditional_edges(
        "route_question",
        route_after_classification,
        {"fast_path": "fast_generate", "full_path": "decompose_question"},
    )
    workflow.add_edge("decompose_question", "retrieve")

    # Fast path — "end" now routes through the output guardrail instead of
    # straight to END, so simple-question answers get the same final check.
    workflow.add_conditional_edges(
        "fast_generate",
        route_after_fast_generate,
        {"end": "output_guardrail", "retrieve": "retrieve"},
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "transform_query": "transform_query", "web_search": "web_search"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("web_search", "generate")

    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_generation_grade,
        # "end" now routes through the output guardrail too — this is the
        # single convergence point every path passes through before END.
        {"end": "output_guardrail", "transform_query": "transform_query", "finalize": "finalize"},
    )
    workflow.add_edge("finalize", "output_guardrail")
    workflow.add_edge("output_guardrail", END)

    return workflow.compile(checkpointer=_checkpointer, name="CRAG-Graph")