from apps.agentic_hr.state.hr_state import HRState

INTENT_RULES = {
    "employment_policy": [
        "policy", "leave", "pto", "vacation", "remote",
        "work from home", "attendance", "termination", "hours"
    ],
    "benefits_compensation": [
        "benefit", "insurance", "bonus", "salary",
        "401k", "medical", "reimbursement"
    ],
    "hr_procedures": [
        "how do i", "apply", "submit", "request", "process",
        "approve", "workflow"
    ],
    "employee_handbook": [
        "conduct", "behavior", "culture", "values",
        "code of conduct"
    ],
    "eligibility_exceptions": [
        "eligible", "eligibility", "exception", "contractor",
        "intern", "part time"
    ],
}


from openai import OpenAI
client = OpenAI()

CLASSIFIER_PROMPT = """
Classify the user's HR question into one of these categories:

- employment_policy
- benefits_compensation
- hr_procedures
- employee_handbook
- eligibility_exceptions
- unknown

Only output the label. Do not explain.
Question: "{q}"
"""

def classify_intent(state: HRState) -> HRState:
    query = state.user_query.lower()

    # --- Rule pass first ---
    for intent, keywords in INTENT_RULES.items():
        if any(k in query for k in keywords):
            state.intent = intent
            state.trace.append(f"Intent classified as: {state.intent} (rule)")
            return state

    # --- LLM fallback ---
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=CLASSIFIER_PROMPT.format(q=state.user_query)
        )

        state.intent = response.output_text.strip()
        state.trace.append(f"Intent classified by LLM: {state.intent}")

    except Exception as e:
        state.intent = "unknown"
        state.trace.append(f"LLM classification failed: {e}")

    return state