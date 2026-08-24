import logging
from app.ingestion.run_ingestion import get_or_build_retriever

# Show the LLM-generated query variants as they're created
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)  # in case it's here instead

retriever = get_or_build_retriever()

query = "who is the CEO of NextBridge"
print(f"\n=== Query: {query} ===\n")

docs = retriever.invoke(query)

print(f"\n=== Retrieved {len(docs)} chunks ===\n")
for i, d in enumerate(docs, 1):
    print(f"--- Result {i} ---")
    print(d.page_content[:300])
    print(f"Source: {d.metadata.get('source')}, Page: {d.metadata.get('page')}")
    print()