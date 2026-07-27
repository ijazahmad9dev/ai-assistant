from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

_search = DuckDuckGoSearchRun()


@tool
def nextbridge_web_search(query: str) -> str:
    """Search the web for current, up-to-date information specifically about NextBridge
    (news, announcements, general public facts, projects, services). Only use this if internal documents
    don't have the answer. Do NOT use this for topics unrelated to NextBridge."""
    scoped_query = f"NextBridge {query}"
    return _search.run(scoped_query)