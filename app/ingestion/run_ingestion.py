from app.ingestion.doclingLoader import load_pdfs_from_directory
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.milvus_store import build_and_save_index, load_index, index_exists


def get_or_build_vectorstore():
    embedding = get_embedding_model()

    if index_exists():
        return load_index(embedding)

    docs = load_pdfs_from_directory()
    return build_and_save_index(docs, embedding)
