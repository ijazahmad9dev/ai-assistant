from langchain_core.tools import tool
from app.api import state


@tool
def nextbridge_docs_search(query: str) -> str:
    """Search NextBridge's internal documents (policies, HR forms, ISO compliance docs,
    project files, internal processes) for specific company information.
    Use this FIRST for any NextBridge-related question before searching the web."""
    if state.vectorstore is None:
        return "Internal document store is not available right now."

    retriever = state.vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in internal documents."

    results = []
    for d in docs:
        heading = (d.metadata.get("dl_meta", {}) or {}).get("headings", [])
        heading_str = f" (Section: {heading[0]})" if heading else ""
        results.append(f"{d.page_content}{heading_str}")

    return "\n\n---\n\n".join(results)