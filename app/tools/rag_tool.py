from langchain_core.tools import tool
from app.api import state
from app.vectorstore.reranker import rerank


def _extract_source_info(doc) -> str:
    filename = doc.metadata.get("source", "Unknown source")
    page = doc.metadata.get("page")
    if page:
        return f"{filename} (Page {page})"
    return filename


@tool
def nextbridge_docs_search(query: str) -> str:
    """Search NextBridge's internal documents (policies, HR forms, ISO compliance docs,
    project files, internal processes) using semantic search, keyword search, and
    relevance reranking. Use this FIRST for any NextBridge-related question before
    searching the web."""
    if state.retriever is None:
        return "Internal document store is not available right now."

    candidates = state.retriever.invoke(query)

    if not candidates:
        return (
            "No relevant information found in internal documents. "
            "You MUST now call nextbridge_web_search with the same or a rephrased "
            "query before responding to the employee."
        )

    docs = rerank(query, candidates, top_k=5)

    results = []
    for d in docs:
        source_info = _extract_source_info(d)
        results.append(f"Content: {d.page_content}\nSource: {source_info}")

    return "\n\n---\n\n".join(results)