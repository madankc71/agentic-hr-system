from apps.agentic_hr.state.hr_state import HRState


def employment_policy_agent(state: HRState) -> HRState:
    """
    Agent responsible for answering employment policy questions.
    This version uses placeholder logic (no RAG yet).
    """

    query = state.user_query.lower()

    if "leave" in query:
        state.answer = (
            "According to company policy, employees are entitled to paid leave "
            "as outlined in the official leave policy document."
        )

    elif "remote" in query:
        state.answer = (
            "The company allows remote work based on role eligibility and "
            "manager approval, as defined in the remote work policy."
        )

    elif "working hours" in query:
        state.answer = (
            "Standard working hours are defined in the employment policy and "
            "may vary depending on role and location."
        )

    else:
        state.answer = (
            "This question relates to employment policy, but no specific policy "
            "information is available at this time."
        )

    state.trace.append("Employment policy agent generated an answer")
    return state