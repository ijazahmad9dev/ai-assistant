import os
import pickle
from langchain_community.vectorstores import FAISS
from app.src.config import FAISS_INDEX_PATH

CHUNKS_PATH = os.path.join(FAISS_INDEX_PATH, "chunks.pkl")


def build_and_save_index(docs, embedding):
    vectorstore = FAISS.from_documents(documents=docs, embedding=embedding)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"FAISS index + {len(docs)} chunks saved to {FAISS_INDEX_PATH}")
    return vectorstore


def load_index(embedding):
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding,
        allow_dangerous_deserialization=True,
    )
    print(f"FAISS index loaded from {FAISS_INDEX_PATH}")
    return vectorstore


def load_chunks():
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


def index_exists():
    return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH)