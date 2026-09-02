from app.ingestion.pdf_loader import load_pdfs_from_directory
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import build_and_save_index, load_index, load_chunks, index_exists
from app.vectorstore.hybrid_retriever import build_hybrid_retriever, build_semantic_retriever


def _load_vectorstore_and_docs():
    """Builds the FAISS index once so both retrievers below share the exact
    same underlying data — no risk of the hybrid and semantic retrievers
    ever drifting out of sync with each other."""
    embedding = get_embedding_model()

    if index_exists():
        vectorstore = load_index(embedding)
        docs = load_chunks()
    else:
        docs = load_pdfs_from_directory()
        vectorstore = build_and_save_index(docs, embedding)

    return vectorstore, docs


def get_or_build_retriever():
    """CRAG's retriever — hybrid (BM25 + semantic) + multi-query expansion."""
    vectorstore, docs = _load_vectorstore_and_docs()
    return build_hybrid_retriever(vectorstore, docs, k=5)


def get_or_build_semantic_retriever():
    """ReAct agent's retriever — plain semantic/vector search only."""
    vectorstore, _ = _load_vectorstore_and_docs()
    return build_semantic_retriever(vectorstore, k=5)