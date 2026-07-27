from langchain_core.tools import tool
from app.api import state


def _extract_source_info(doc) -> str:
    dl_meta = doc.metadata.get("dl_meta", {}) or {}
    filename = dl_meta.get("origin", {}).get("filename", "Unknown source")

    pages = set()
    for item in dl_meta.get("doc_items", []):
        for prov in item.get("prov", []):
            page_no = prov.get("page_no")
            if page_no is not None:
                pages.add(page_no)

    if pages:
        page_str = ", ".join(str(p) for p in sorted(pages))
        page_label = "Page" if len(pages) == 1 else "Pages"
        return f"{filename} ({page_label} {page_str})"
    return filename


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
        source_info = _extract_source_info(d)
        heading = (d.metadata.get("dl_meta", {}) or {}).get("headings", [])
        heading_str = f" | Section: {heading[0]}" if heading else ""

        results.append(
            f"Content: {d.page_content}\n"
            f"Source: {source_info}{heading_str}"
        )

    return "\n\n---\n\n".join(results)