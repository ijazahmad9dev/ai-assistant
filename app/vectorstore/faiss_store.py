import os
import pickle
import time
from langchain_community.vectorstores import FAISS
from app.src.config import FAISS_INDEX_PATH

CHUNKS_PATH = os.path.join(FAISS_INDEX_PATH, "chunks.pkl")
BATCH_SIZE = 20


def build_and_save_index(docs, embedding):
    vectorstore = None
    total_batches = (len(docs) - 1) // BATCH_SIZE + 1

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(documents=batch, embedding=embedding)
        else:
            vectorstore.add_documents(batch)

        time.sleep(1)  # pace requests so we don't overload the Ollama origin server

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