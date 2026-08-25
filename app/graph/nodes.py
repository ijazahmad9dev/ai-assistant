from langchain_core.documents import Document
from app.api import state as app_state
from app.llm.ollama_llm import get_llm
from app.tools.web_search import _search as tavily_search
from app.vectorstore.reranker import rerank
from app.graph.state import GraphState


def route_question(state: GraphState) -> GraphState:
    llm = get_llm()
    prompt = (
        "Classify this employee question as SIMPLE or COMPLEX.\n\n"
        "SIMPLE = asks for exactly one discrete fact (a name, a date, a single number) "
        "that is very likely stated in one place.\n"
        "COMPLEX = anything else: policy explanations, multi-part questions, "
        "questions with sub-cases or conditions, list/enumeration requests, or "
        "anything where the answer might span more than one section.\n\n"
        "When in doubt, classify as COMPLEX — it is safer to over-classify.\n\n"
        f"Question: {state['question']}\n\n"
        "Respond with only one word: SIMPLE or COMPLEX."
    )
    result = llm.invoke(prompt).content.strip().upper()
    route = "simple" if "SIMPLE" in result else "complex"
    return {**state, "route": route}


def retrieve(state: GraphState) -> GraphState:
    k = 10 if state.get("route") == "complex" else 5
    candidates = app_state.retriever.invoke(state["question"])
    docs = rerank(state["question"], candidates, top_k=k)
    return {**state, "documents": docs, "retrieval_mode": "vector"}


def grade_documents(state: GraphState) -> GraphState:
    """CRAG grading node — scores whether retrieved docs actually answer the question."""
    llm = get_llm()
    question = state["question"]
    docs = state["documents"]

    if not docs:
        return {**state, "grade": "irrelevant"}

    n = 8 if state.get("route") == "complex" else 5
    joined = "\n\n".join(d.page_content[:400] for d in docs[:n])
    prompt = (
        "You are grading retrieved documents for relevance to a question.\n"
        f"Question: {question}\n\nDocuments:\n{joined}\n\n"
        "If this question asks to LIST or ENUMERATE multiple items (e.g. 'list all X', "
        "'who are the Y'), consider the documents sufficient if they collectively "
        "provide SOME relevant items, even if the list might be incomplete. Do not "
        "require a single document to fully answer the question on its own.\n\n"
        "Do these documents contain enough information to answer the question? "
        "Respond with only one word: RELEVANT or IRRELEVANT."
    )
    result = llm.invoke(prompt).content.strip().upper()
    grade = "relevant" if "RELEVANT" in result and "IR" not in result else "irrelevant"
    return {**state, "grade": grade}


def transform_query(state: GraphState) -> GraphState:
    """Rewrites the question to improve retrieval on the next pass."""
    llm = get_llm()
    prompt = (
        "The following question did not retrieve useful documents. Rewrite it to be "
        "more specific or use alternate terminology that might match company documents "
        "(e.g. formal job titles, policy names). Return ONLY the rewritten question.\n\n"
        f"Original question: {state['question']}"
    )
    rewritten = llm.invoke(prompt).content.strip()
    return {
        **state,
        "question": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


def web_search_node(state: GraphState) -> GraphState:
    result = tavily_search.invoke(state["original_question"])
    entries = result.get("results", []) if isinstance(result, dict) else []
    docs = [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", ""), "title": r.get("title", "")},
        )
        for r in entries
    ]
    return {**state, "documents": docs, "retrieval_mode": "web", "grade": "relevant"}


def generate(state: GraphState) -> GraphState:
    llm = get_llm()
    n = 8 if state.get("route") == "complex" else 5
    context = "\n\n---\n\n".join(d.page_content for d in state["documents"][:n])
    prompt = (
        "Answer the employee's question using ONLY the context below. If this is a "
        "list/enumeration question, combine information from ALL relevant context "
        "chunks to produce as complete a list as possible. If the context doesn't "
        "contain the answer, say so honestly.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['original_question']}\n\nAnswer:"
    )
    answer = llm.invoke(prompt).content
    return {**state, "generation": answer}


def grade_generation(state: GraphState) -> GraphState:
    """Self-RAG reflection: checks the answer against retrieved context."""
    llm = get_llm()
    n = 8 if state.get("route") == "complex" else 5
    context = "\n\n".join(d.page_content[:300] for d in state["documents"][:n])
    prompt = (
        "Does the CONTEXT below contain support for this ANSWER? The context may "
        "include irrelevant material — that's fine. Only mark UNGROUNDED if the "
        "answer states something that directly contradicts the context, or is not "
        "supported by ANY part of the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['original_question']}\n\n"
        f"Answer: {state['generation']}\n\nRespond with only one word: GROUNDED or UNGROUNDED."
    )
    result = llm.invoke(prompt).content.strip().upper()
    grade = "relevant" if "GROUNDED" in result and "UN" not in result else "irrelevant"

    updates = {**state, "grade": grade}
    if grade == "relevant":
        updates["best_generation"] = state["generation"]
    return updates


def finalize(state: GraphState) -> GraphState:
    """Called when retries are exhausted and the current generation was never
    graded relevant. Falls back to the best grounded answer seen so far, if any."""
    if state.get("best_generation"):
        return {**state, "generation": state["best_generation"]}
    return state



REFUSAL_MARKERS = ["sorry", "don't have", "doesn't include", "not included", "cannot find", "no information"]

def fast_generate(state: GraphState) -> GraphState:
    llm = get_llm()
    candidates = app_state.retriever.invoke(state["question"])
    docs = rerank(state["question"], candidates, top_k=5)  # widened from 3

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = (
        "Answer the employee's question using ONLY the context below. If the "
        "context doesn't contain the answer, say so honestly.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['original_question']}\n\nAnswer:"
    )
    answer = llm.invoke(prompt).content

    # Escape hatch: if the fast path failed to answer, signal that a full
    # CRAG retry is needed instead of returning the failure directly.
    looks_like_refusal = any(marker in answer.lower() for marker in REFUSAL_MARKERS)

    return {
        **state,
        "documents": docs,
        "generation": answer,
        "best_generation": None if looks_like_refusal else answer,
        "retrieval_mode": "vector",
        "grade": "irrelevant" if looks_like_refusal else "relevant",
    }