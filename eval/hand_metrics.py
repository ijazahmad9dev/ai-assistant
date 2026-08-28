"""
Hand-rolled implementation of Context Precision.

Context Precision measures: of the chunks we retrieved, how many were actually
relevant to answering the question — and were the relevant ones ranked near the top?

Formula (matches RAGAS's own definition, computed manually here instead of calling
the library, so the mechanics are visible):

    Context Precision@K = ( sum over k=1..K of (Precision@k * relevance_k) ) / (total relevant items in top K)

where:
    relevance_k ∈ {0, 1} — is the chunk at rank k relevant to the expected answer?
    Precision@k = (# relevant chunks in positions 1..k) / k

This rewards retrieval systems that put relevant chunks EARLY, not just present
somewhere in the list — a chunk that's relevant but ranked last contributes much
less than one ranked first, even though both are "hits."
"""

from app.llm.ollama_llm import get_llm


def _judge_relevance(question: str, expected_answer: str, chunk_text: str) -> bool:
    """LLM judge: does this chunk actually contain info needed for the expected answer?"""
    llm = get_llm()
    prompt = (
        "You are checking whether a retrieved passage is relevant to answering a "
        "question, given the known correct answer.\n\n"
        f"Question: {question}\n"
        f"Correct answer: {expected_answer}\n\n"
        f"Retrieved passage:\n{chunk_text[:500]}\n\n"
        "Does this passage contain information that helps answer the question "
        "(even partially)? Respond with only one word: YES or NO."
    )
    result = llm.invoke(prompt).content.strip().upper()
    return "YES" in result and "NO" not in result.replace("YES", "")


def context_precision_at_k(question: str, expected_answer: str, retrieved_contexts: list) -> float:
    """Hand-rolled Context Precision@K, computed exactly per the formula above."""
    if not retrieved_contexts:
        return 0.0

    relevance_flags = [
        _judge_relevance(question, expected_answer, chunk)
        for chunk in retrieved_contexts
    ]

    num_relevant_total = sum(relevance_flags)
    if num_relevant_total == 0:
        return 0.0

    weighted_sum = 0.0
    relevant_so_far = 0
    for k, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            relevant_so_far += 1
            precision_at_k = relevant_so_far / k
            weighted_sum += precision_at_k

    return weighted_sum / num_relevant_total


def faithfulness_score(answer: str, contexts: list) -> float:
    """Hand-rolled faithfulness check that doesn't depend on RAGAS's internal
    structured-output parsing (which has been failing against the local
    Ollama model — see NaN faithfulness/context_precision/context_recall in
    results.json). Uses the same plain YES/NO judging pattern as
    context_precision_at_k, which has been reliable in practice.

    Returns 1.0 if the answer is judged fully grounded in the given contexts,
    0.0 otherwise.
    """
    if not answer or not contexts:
        return 0.0

    llm = get_llm()
    context_block = "\n\n---\n\n".join(c[:500] for c in contexts)
    prompt = (
        "You are checking whether an ANSWER is faithfully grounded in the given CONTEXT.\n\n"
        f"CONTEXT:\n{context_block}\n\nANSWER:\n{answer}\n\n"
        "Does the CONTEXT support the claims made in the ANSWER, without the ANSWER "
        "stating anything that contradicts or is absent from the CONTEXT? Minor phrasing "
        "differences and reasonable summarization are fine.\n\n"
        "Respond with only one word: YES or NO."
    )
    result = llm.invoke(prompt).content.strip().upper()
    return 1.0 if ("YES" in result and "NO" not in result.replace("YES", "")) else 0.0