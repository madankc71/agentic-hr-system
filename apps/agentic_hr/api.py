from fastapi import FastAPI
from pydantic import BaseModel

from apps.agentic_hr.graph import build_hr_graph
from apps.agentic_hr.state.hr_state import HRState


app = FastAPI(title="Agentic HR System")

# Build the workflow once at startup
hr_graph = build_hr_graph()


class HRChatRequest(BaseModel):
    message: str


class HRChatResponse(BaseModel):
    intent: str | None
    answer: str | None
    trace: list[str]


@app.post("/hr/chat", response_model=HRChatResponse)
def hr_chat(request: HRChatRequest):
    """
    Entry point for HR agentic workflow.
    """
    print("API call")

    initial_state = HRState(user_query=request.message)

    # LangGraph returns dict-based state
    final_state = hr_graph.invoke(initial_state)

    return HRChatResponse(
        intent=final_state.get("intent"),
        answer=final_state.get("answer"),
        trace=final_state.get("trace", []),
    )