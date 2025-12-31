from apps.agentic_hr.state.hr_state import HRState
from openai import OpenAI

client = OpenAI()

INTENT_RULES = {
    "eligibility_exceptions": [
        "eligible", "eligibility", "exception",
        "contractor", "intern", "part time", "probation"
    ],
    "employment_policy": [
        "policy", "leave", "pto", "vacation",
        "remote", "attendance", "termination", "hours"
    ],
    "benefits_compensation": [
        "benefit", "insurance", "bonus", "salary",
        "401k", "medical", "reimbursement"
    ],
    "hr_procedures": [
        "how do i", "apply", "submit", "request",
        "process", "approve", "workflow"
    ],
    "employee_handbook": [
        "conduct", "behavior", "culture", "values",
        "code of conduct"
    ],
}

# Priority (who wins ties)
INTENT_PRIORITY = [
    "eligibility_exceptions",
    "hr_procedures",
    "employment_policy",
    "benefits_compensation",
    "employee_handbook",
]


CLASSIFIER_PROMPT = """
Classify the user's HR question into one of these categories:

- employment_policy
- benefits_compensation
- hr_procedures
- employee_handbook
- eligibility_exceptions
- unknown

Return ONLY the label. No sentences.
Question: "{q}"
"""

def classify_intent(state: HRState) -> HRState:
    query = state.user_query.lower()

    # Score rule matches instead of first-hit
    scores = {intent: 0 for intent in INTENT_RULES}

    for intent, keywords in INTENT_RULES.items():
        for k in keywords:
            if k in query:
                scores[intent] += 1

    # choose best rule match following priority order
    best_intent = "unknown"
    best_score = 0

    for intent in INTENT_PRIORITY:
        if scores[intent] > best_score:
            best_intent = intent
            best_score = scores[intent]

    # If confidence is good — stop here, we consider 2+ signals as "confident"
    if best_score >= 2:
        state.intent = best_intent
        state.trace.append(f"Intent classified (rules, confident): {best_intent}")
        return state

    # --- LLM fallback ---
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=CLASSIFIER_PROMPT.format(q=state.user_query)
        )
        
        llm_label = response.output_text.strip()

        # sanity check — prevent hallucinated labels
        if llm_label in INTENT_RULES or llm_label == "unknown":
            state.intent = llm_label
            state.trace.append(f"Intent classified (LLM fallback): {llm_label}")
        else:
            state.intent = best_intent
            state.trace.append(f"LLM returned invalid label ({llm_label}), kept rule result: {best_intent}")

    except Exception as e:
        state.intent = "unknown"
        state.trace.append(f"LLM classification failed: {e}")

    return state