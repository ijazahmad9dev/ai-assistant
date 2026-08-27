from app.src import config 

from fastapi import FastAPI
from app.ingestion.run_ingestion import get_or_build_retriever
from app.api.routes import router
from app.api import state

app = FastAPI(title="NextBridge RAG Agent API")

# from app.vectorstore.reranker import get_reranker

# print("Loading cross-encoder reranker...")
# get_reranker()
# print("Reranker ready.")

print("Building hybrid retriever (FAISS + BM25)...")
state.retriever = get_or_build_retriever()
print("Retriever ready.")

app.include_router(router)