from langchain_groq import ChatGroq
from app.src.config import GROQ_API_KEY, GROQ_MODEL


def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.2,
    )