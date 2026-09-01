import re

from app.graph.state import GraphState

MAX_RETRIES = 3

WEB_SEARCH_INTENT_PATTERNS = [
    r"\bsearch (the )?web\b",
    r"\bweb search\b",
    r"\bsearch online\b",
    r"\blook (it )?up online\b",
    r"\bgoogle (it|this)\b",
    r"\bsearch the internet\b",
    r"\bcheck online\b",
    r"\bbrowse the web\b",
]


def _wants_explicit_web_search(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in WEB_SEARCH_INTENT_PATTERNS)


def route_after_input_guardrail(state: GraphState) -> str:
    """Explicit 'search the web' intent skips route_question/decompose_question/
    retrieve entirely and goes straight to web_search — checked AFTER the
    blocked check, so a blocked+injection combo still can't slip through."""
    if state.get("blocked"):
        return "blocked"
    if _wants_explicit_web_search(state["original_question"]):
        return "explicit_web_search"
    return "allowed"


def route_after_classification(state: GraphState) -> str:
    if state.get("route") == "simple":
        return "fast_path"
    return "full_path"


def route_after_grading(state: GraphState) -> str:
    if state["grade"] == "relevant":
        return "generate"
    if state["retry_count"] >= MAX_RETRIES:
        if not state.get("web_search_attempted"):
            return "web_search"
        return "generate"
    return "transform_query"


def route_after_generation_grade(state: GraphState) -> str:
    if state["grade"] == "relevant":
        return "end"
    if state["retry_count"] >= MAX_RETRIES:
        if not state.get("web_search_attempted"):
            return "web_search"
        return "finalize"
    return "transform_query"


def route_after_fast_generate(state: GraphState) -> str:
    if state["grade"] == "relevant":
        return "end"
    return "retrieve"


def route_after_web_search(state: GraphState) -> str:
    """web_search_node sets web_search_terminal=True when it already produced
    a final canned message (nothing found / off-topic) — in that case, skip
    generate() entirely rather than letting it silently overwrite that
    message with a generic 'no context' refusal."""
    if state.get("web_search_terminal"):
        return "end"
    return "generate"