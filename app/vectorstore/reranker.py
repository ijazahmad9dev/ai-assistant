from sentence_transformers import CrossEncoder
from langsmith import traceable

_model = None


def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


@traceable(name="cross_encoder_rerank", run_type="tool")
def rerank(query: str, docs: list, top_k: int = 5):
    if not docs:
        return docs

    model = get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]