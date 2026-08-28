from app.ingestion.run_ingestion import get_or_build_retriever
from app.api import state as app_state
from app.agent.nextbridge_agent import build_agent
from app.graph.run_graph import run_graph_query

import re


def _ensure_retriever():
    if app_state.retriever is None:
        app_state.retriever = get_or_build_retriever()


def run_after_pipeline(question: str, session_id: str) -> dict:
    """AFTER: the CRAG/Self-RAG graph — grading, correction loop, adaptive routing."""
    _ensure_retriever()
    result = run_graph_query(question, session_id)

    # Uses the EXACT context the graph used to generate this answer (now
    # returned by run_graph_query) instead of re-running retrieval separately
    # on the raw question — a fresh call would ignore decomposition, query
    # rewrites, and web-search fallback, and score the answer against
    # contexts that have nothing to do with what actually produced it.
    context_texts = [d["page_content"] for d in result.get("documents", [])]
    if not context_texts:
        context_texts = ["(no context retrieved / request was blocked)"]

    return {
        "answer": result["answer"],
        "contexts": context_texts,
        "retrieval_mode": result["retrieval_mode"],
        "retry_count": result["retry_count"],
    }


def run_before_pipeline(question: str, session_id: str) -> dict:
    """BEFORE: the ReAct agent — decides which tool(s) to call, no grading/correction loop."""
    _ensure_retriever()
    agent = build_agent()

    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    final_message = result["messages"][-1]
    answer = final_message.content

    context_texts = []
    for msg in result["messages"]:
        if getattr(msg, "type", None) == "tool" or msg.__class__.__name__ == "ToolMessage":
            content = getattr(msg, "content", "")
            if content:
                chunks = re.split(r"\n\n---\n\n", content)
                context_texts.extend(chunks)

    if not context_texts:
        context_texts = ["(no tool was called / no context retrieved)"]

    return {
        "answer": answer,
        "contexts": context_texts,
        "retrieval_mode": "agent-tool-call",
        "retry_count": 0,
    }