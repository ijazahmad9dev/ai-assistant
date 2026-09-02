from app.src import config 

from fastapi import FastAPI
from app.ingestion.run_ingestion import get_or_build_retriever, get_or_build_semantic_retriever
from app.api.routes import router
from app.api import state

app = FastAPI(title="NextBridge RAG Agent API")

print("Building hybrid retriever (FAISS + BM25 + multi-query) for CRAG...")
state.retriever = get_or_build_retriever()
print("Hybrid retriever ready.")

print("Building semantic-only retriever for ReAct agent...")
state.semantic_retriever = get_or_build_semantic_retriever()
print("Semantic retriever ready.")

app.include_router(router)