from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from app.llm.ollama_llm import get_llm


def build_hybrid_retriever(vectorstore, docs, k=5, semantic_weight=0.6):
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[1 - semantic_weight, semantic_weight],
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=hybrid_retriever,
        llm=get_llm(),
    )

    return multi_query_retriever