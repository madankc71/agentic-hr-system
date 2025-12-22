from apps.agentic_hr.nodes.employment_policy_agent import employment_policy_agent
from apps.agentic_hr.state.hr_state import HRState


def test_leave_policy_answer():
    state = HRState(user_query="What is the leave policy?")
    state = employment_policy_agent(state)

    assert state.answer is not None
    assert "leave" in state.answer.lower()


def test_remote_work_answer():
    state = HRState(user_query="What is the remote work policy?")
    state = employment_policy_agent(state)

    assert state.answer is not None
    assert "remote" in state.answer.lower()
