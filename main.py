from fastapi import FastAPI

from app.ingestion.run_ingestion import get_or_build_vectorstore
from app.agent.nextbridge_agent import build_agent
from app.api.routes import router
from app.api import state

app = FastAPI(title="NextBridge RAG Agent API")

print("Loading vectorstore...")
state.vectorstore = get_or_build_vectorstore()

print("Building agent...")
state.agent = build_agent()
print("Agent ready.")

app.include_router(router)


# from app.ingestion.run_ingestion import get_or_build_vectorstore
# from app.rag.rag_chain import build_rag_chain, ask


# def main():
#     vectorstore = get_or_build_vectorstore()
#     rag_chain = build_rag_chain(vectorstore, k=3)

#     print("RAG system ready. Type 'exit' to quit.\n")
#     while True:
#         query = input("Ask a question: ").strip()
#         if query.lower() in {"exit", "quit"}:
#             break

#         answer, context = ask(rag_chain, query)

#         print(f"\nAnswer:\n{answer}\n")
#         print("Sources:")
#         for i, doc in enumerate(context, 1):
#             print(f"  [{i}] {doc.page_content[:150]}...")
#             meta = doc.metadata.get("dl_meta", {})
#             if meta.get("headings"):
#                 print(f"      Section: {meta['headings']}")
#         print()


# if __name__ == "__main__":
#     main()