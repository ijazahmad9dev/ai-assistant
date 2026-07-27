from langchain_huggingface import HuggingFaceEmbeddings
from app.src.config import EMBED_MODEL_ID


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)