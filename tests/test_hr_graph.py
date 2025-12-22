from apps.agentic_hr.graph import build_hr_graph
from apps.agentic_hr.state.hr_state import HRState


def test_hr_graph_employment_policy_flow():
    graph = build_hr_graph()

    initial_state = HRState(user_query="What is the leave policy?")
    final_state = graph.invoke(initial_state)

    assert final_state["intent"] == "employment_policy"
    assert final_state["answer"] is not None
    assert "leave" in final_state["answer"].lower()
