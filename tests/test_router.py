from apps.agentic_hr.nodes.router import route_by_intent
from apps.agentic_hr.state.hr_state import HRState


def test_route_employment_policy():
    state = HRState(user_query="x", intent="employment_policy")
    assert route_by_intent(state) == "employment_policy_agent"


def test_route_benefits():
    state = HRState(user_query="x", intent="benefits_compensation")
    assert route_by_intent(state) == "benefits_compensation_agent"


def test_route_procedures():
    state = HRState(user_query="x", intent="hr_procedures")
    assert route_by_intent(state) == "hr_procedures_agent"


def test_route_handbook():
    state = HRState(user_query="x", intent="employee_handbook")
    assert route_by_intent(state) == "employee_handbook_agent"


def test_route_eligibility():
    state = HRState(user_query="x", intent="eligibility_exceptions")
    assert route_by_intent(state) == "eligibility_exceptions_agent"


def test_route_unknown():
    state = HRState(user_query="x", intent="unknown")
    assert route_by_intent(state) == "fallback_agent"