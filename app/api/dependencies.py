from fastapi import HTTPException
from app.api import state


def get_agent():
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready yet.")
    return state.agent