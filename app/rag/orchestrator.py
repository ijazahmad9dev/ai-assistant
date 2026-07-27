from app.rag.guardrail import is_about_nextbridge
from app.rag.rag_chain import ask
from app.tools.web_search import nextbridge_web_search

OFF_TOPIC_MESSAGE = (
    "I'm here to help with questions about NextBridge. "
    "Feel free to ask me anything about NextBridge and I'd be happy to help!"
)


def handle_query(rag_chain, question: str):
    if not is_about_nextbridge(question):
        return OFF_TOPIC_MESSAGE, []

    answer, context = ask(rag_chain, question)

    no_useful_context = not context
    answer_uncertain = any(
        phrase in answer.lower()
        for phrase in ["don't know", "cannot be found", "no information", "not mentioned"]
    )

    if no_useful_context or answer_uncertain:
        web_answer = nextbridge_web_search(question)
        return web_answer, []

    return answer, context