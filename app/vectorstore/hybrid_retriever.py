from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from app.llm.ollama_llm import get_llm


HYDE_PROMPT_TEMPLATE = (
    "You are writing a hypothetical passage that would appear in NextBridge's "
    "internal HR handbook or company documents, answering the following employee "
    "question. Write it as if it were a real excerpt from the actual policy "
    "document — confident, factual in tone, using formal HR/policy language — "
    "even though you don't know the real answer. This passage is used ONLY to "
    "improve document retrieval; it is never shown to the user.\n\n"
    "Question: {question}\n\n"
    "Hypothetical passage:"
)


class HydeRetriever(BaseRetriever):
    """HyDE (Hypothetical Document Embeddings): instead of embedding the raw
    query, an LLM first generates a hypothetical ANSWER passage, and that
    passage — not the question — is what gets embedded and searched against.
    This replaces MultiQueryRetriever as the "advanced retriever" component;
    it does not replace the BM25 side of hybrid search, which stays as-is."""

    model_config = {"arbitrary_types_allowed": True}

    vectorstore: VectorStore
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        llm = get_llm()
        hyde_prompt = HYDE_PROMPT_TEMPLATE.format(question=query)
        hypothetical_doc = llm.invoke(hyde_prompt).content
        # Embed and search using the hypothetical ANSWER text, not the
        # original question — this is the entire point of HyDE.
        return self.vectorstore.similarity_search(hypothetical_doc, k=self.k)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


def build_hybrid_retriever(vectorstore, docs, k=5, semantic_weight=0.6):
    hyde_retriever = HydeRetriever(vectorstore=vectorstore, k=k)

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # Hybrid search unchanged in spirit — BM25 (keyword) + a dense retriever
    # (semantic_weight) — the dense side is now HyDE-driven instead of a
    # plain vectorstore retriever wrapped in MultiQueryRetriever.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, hyde_retriever],
        weights=[1 - semantic_weight, semantic_weight],
    )

    return hybrid_retriever