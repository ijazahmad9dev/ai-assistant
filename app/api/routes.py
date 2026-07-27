from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import QueryRequest, QueryResponse
from app.api.dependencies import get_agent
from app.agent.nextbridge_agent import run_agent

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest, agent=Depends(get_agent)):
    try:
        answer = run_agent(agent, request.question, request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(answer=answer, sources=[])


@router.get("/health")
def health_check():
    from app.api import state
    return {"status": "ready" if state.agent is not None else "loading"}