from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from app.llm.groq_llm import get_llm
from app.rag.prompt import RAG_PROMPT


def build_rag_chain(vectorstore, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = get_llm()

    question_answer_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def ask(rag_chain, question: str):
    result = rag_chain.invoke({"input": question})
    return result["answer"], result["context"]