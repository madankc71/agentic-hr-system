from apps.agentic_hr.state.hr_state import HRState


def route_by_intent(state: HRState) -> str:
    """
    Deterministic router.
    Returns the name of the agent node to execute next.
    """

    intent_to_agent = {
        "employment_policy": "employment_policy_agent",
        "benefits_compensation": "benefits_compensation_agent",
        "hr_procedures": "hr_procedures_agent",
        "employee_handbook": "employee_handbook_agent",
        "eligibility_exceptions": "eligibility_exceptions_agent",
        "unknown": "fallback_agent",
    }

    agent = intent_to_agent.get(state.intent, "fallback_agent")

    state.trace.append(f"Routed to agent: {agent}")
    return agent