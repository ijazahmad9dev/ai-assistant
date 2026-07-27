from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate.from_template(
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "If the answer cannot be found in the context, say you don't know.\n"
    "Query: {input}\n"
    "Answer:\n"
)