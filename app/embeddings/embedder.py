from langchain_ollama import OllamaEmbeddings
from app.src.config import EMBED_MODEL_ID, OLLAMA_BASE_URL


def get_embedding_model():
    return OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBED_MODEL_ID,
    )