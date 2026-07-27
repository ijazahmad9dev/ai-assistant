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