from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest
from app.api.dependencies import get_agent
from app.agent.nextbridge_agent import run_agent_stream

router = APIRouter()


@router.post("/query")
def query_agent(request: QueryRequest, agent=Depends(get_agent)):
    def event_generator():
        try:
            for chunk in run_agent_stream(agent, request.question, request.session_id):
                yield chunk
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(event_generator(), media_type="text/plain")


@router.get("/health")
def health_check():
    from app.api import state
    return {"status": "ready" if state.agent is not None else "loading"}