from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest
from app.agent.nextbridge_agent import build_agent, run_agent_stream

router = APIRouter()


@router.post("/query")
def query_agent(request: QueryRequest):
    agent = build_agent()  # fresh agent per request, no Depends/state needed

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
    return {"status": "ready" if state.retriever is not None else "loading"}