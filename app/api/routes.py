from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest, GraphQueryResponse
from app.agent.nextbridge_agent import build_agent, run_agent_stream
from app.graph.run_graph import run_graph_query

router = APIRouter()


@router.post("/query")
def query_agent(request: QueryRequest):
    agent = build_agent()

    def event_generator():
        try:
            for chunk in run_agent_stream(agent, request.question, request.session_id):
                yield chunk
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(event_generator(), media_type="text/plain")


@router.post("/query/graph", response_model=GraphQueryResponse)
def query_graph(request: QueryRequest):
    try:
        result = run_graph_query(request.question, request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GraphQueryResponse(**result)


@router.get("/health")
def health_check():
    from app.api import state
    return {"status": "ready" if state.retriever is not None else "loading"}