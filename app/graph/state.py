from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str                  # current question (may get rewritten)
    original_question: str         # kept for reference/logging
    documents: List[Document]      # currently held candidate docs
    generation: Optional[str]      # current draft answer
    best_generation: Optional[str] # last answer graded as grounded
    retrieval_mode: str            # "vector" | "web"
    grade: Optional[str]           # "relevant" | "irrelevant"
    retry_count: int               # rewrite/re-retrieve attempts so far
    route: Optional[str]           # "simple" | "complex"