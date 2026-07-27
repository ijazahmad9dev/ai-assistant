import os
from langchain_milvus import Milvus
from app.src.config import MILVUS_URI, MILVUS_COLLECTION


def build_and_save_index(docs, embedding):
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=MILVUS_COLLECTION,
        connection_args={"uri": MILVUS_URI},
        index_params={"index_type": "FLAT", "metric_type": "COSINE"},
        drop_old=True,
    )
    print(f"Milvus index built at {MILVUS_URI}")
    return vectorstore


def load_index(embedding):
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name=MILVUS_COLLECTION,
        connection_args={"uri": MILVUS_URI},
    )
    print(f"Milvus index loaded from {MILVUS_URI}")
    return vectorstore


def index_exists():
    return os.path.exists(MILVUS_URI)