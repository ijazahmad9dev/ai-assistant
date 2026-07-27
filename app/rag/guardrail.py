from langchain_core.prompts import PromptTemplate
from app.llm.groq_llm import get_llm

TOPIC_CHECK_PROMPT = PromptTemplate.from_template(
    "You are a topic classifier for a company chatbot about \"NextBridge\".\n"
    "Decide if the user's question is related to NextBridge (the company, its services, "
    "products, projects, internship program, policies, or anything company-specific).\n"
    "Respond with only one word: YES or NO.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def is_about_nextbridge(question: str) -> bool:
    llm = get_llm()
    chain = TOPIC_CHECK_PROMPT | llm
    result = chain.invoke({"question": question})
    text = result.content if hasattr(result, "content") else str(result)
    return text.strip().upper().startswith("Y")