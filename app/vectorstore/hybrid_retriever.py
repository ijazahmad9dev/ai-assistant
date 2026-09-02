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

def build_semantic_retriever(vectorstore, k=5):
    """Plain vector similarity search — no BM25, no multi-query expansion.
    Used by the ReAct agent, which calls this tool inside its own reasoning
    loop and can just re-invoke it with a different query itself if the first
    call misses; it doesn't need the hybrid retriever's extra recall/latency
    since it already has its own iterative retry mechanism (multiple tool
    calls), unlike CRAG's single retrieve() call per attempt."""
    return vectorstore.as_retriever(search_kwargs={"k": k})