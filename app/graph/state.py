from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str                          # current question (may get rewritten)
    original_question: str                 # kept for reference/logging
    documents: List[Document]              # currently held candidate docs (accumulates across retries)
    generation: Optional[str]              # current draft answer
    best_generation: Optional[str]         # last answer graded as grounded
    retrieval_mode: str                    # "vector" | "web"
    grade: Optional[str]                   # "relevant" | "irrelevant"
    retry_count: int                       # rewrite/re-retrieve attempts so far
    route: Optional[str]                   # "simple" | "complex"
    sub_questions: List[str]               # decomposed atomic parts of the question
    uncovered_sub_questions: List[str]     # parts not yet answered by current documents/generation
    blocked: Optional[bool]                # input guardrail verdict
    block_reason: Optional[str]            # why the input guardrail blocked it
    output_blocked: Optional[bool]         # output guardrail verdict
    unsupported_claims: List[str]          # claims the output guardrail flagged as ungrounded
    original_sub_questions: List[str]