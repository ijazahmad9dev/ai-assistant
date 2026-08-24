from app.ingestion.pdf_loader import load_pdfs_from_directory
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import build_and_save_index, load_index, load_chunks, index_exists
from app.vectorstore.hybrid_retriever import build_hybrid_retriever


def get_or_build_retriever():
    embedding = get_embedding_model()

    if index_exists():
        vectorstore = load_index(embedding)
        docs = load_chunks()
    else:
        docs = load_pdfs_from_directory()
        vectorstore = build_and_save_index(docs, embedding)

    return build_hybrid_retriever(vectorstore, docs, k=5)