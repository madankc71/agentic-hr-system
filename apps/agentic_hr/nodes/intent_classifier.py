from apps.agentic_hr.state.hr_state import HRState


def classify_intent(state: HRState) -> HRState:
    """
    Deterministic intent classifier.
    Updates only the `intent` field of the state.
    """

    query = state.user_query.lower()

    if any(word in query for word in ["policy", "working hours", "remote", "leave policy"]):
        state.intent = "employment_policy"

    elif any(word in query for word in ["benefit", "insurance", "bonus", "compensation"]):
        state.intent = "benefits_compensation"

    elif any(word in query for word in ["how do i", "process", "apply", "request", "procedure"]):
        state.intent = "hr_procedures"

    elif any(word in query for word in ["conduct", "behavior", "values", "culture"]):
        state.intent = "employee_handbook"

    elif any(word in query for word in ["eligible", "eligibility", "exception", "contractor"]):
        state.intent = "eligibility_exceptions"

    else:
        state.intent = "unknown"

    state.trace.append(f"Intent classified as: {state.intent}")
    return state
