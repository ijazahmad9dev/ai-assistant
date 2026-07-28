from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from app.src.config import TAVILY_API_KEY, NEXTBRIDGE_DOMAIN

_search = TavilySearch(
    max_results=3,
    tavily_api_key=TAVILY_API_KEY,
)


@tool
def nextbridge_web_search(query: str) -> str:
    """Search the web for current, up-to-date information specifically about NextBridge
    (news, announcements, general public facts), restricted to NextBridge's official
    website. Only use this if internal documents don't have the answer. Do NOT use this
    for topics unrelated to NextBridge."""
    result = _search.invoke(query)

    # Tavily returns a dict with a "results" list of {title, url, content, ...}
    if isinstance(result, dict) and "results" in result:
        entries = result["results"]
        if not entries:
            return "No relevant results found on NextBridge's website."

        formatted = []
        for r in entries:
            formatted.append(f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')}")
        return "\n\n---\n\n".join(formatted)

    return str(result)