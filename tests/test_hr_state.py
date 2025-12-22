from apps.agentic_hr.state.hr_state import HRState


def test_hr_state_initialization():
    state = HRState(user_query="What is the leave policy?")

    assert state.user_query == "What is the leave policy?"
    assert state.intent is None
    assert state.evidence == []
    assert state.answer is None
    assert state.trace == []