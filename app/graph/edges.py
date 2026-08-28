from app.graph.state import GraphState

MAX_RETRIES = 3


def route_after_input_guardrail(state: GraphState) -> str:
    if state.get("blocked"):
        return "blocked"
    return "allowed"


def route_after_classification(state: GraphState) -> str:
    if state.get("route") == "simple":
        return "fast_path"
    return "full_path"


def route_after_grading(state: GraphState) -> str:
    if state["grade"] == "relevant":
        return "generate"
    if state["retry_count"] >= MAX_RETRIES:
        return "web_search"
    return "transform_query"


def route_after_generation_grade(state: GraphState) -> str:
    if state["grade"] == "relevant":
        return "end"
    if state["retry_count"] >= MAX_RETRIES:
        return "finalize"
    return "transform_query"


def route_after_fast_generate(state: GraphState) -> str:
    if state["grade"] == "relevant":
        return "end"
    return "retrieve"