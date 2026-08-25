from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    session_id: str


class SourceChunk(BaseModel):
    content: str
    section: Optional[str] = None
    source: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []


class GraphQueryResponse(BaseModel):
    answer: str
    retrieval_mode: str
    retry_count: int
    route: Optional[str] = None