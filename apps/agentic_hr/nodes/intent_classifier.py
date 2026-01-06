# from apps.agentic_hr.state.hr_state import HRState
# from openai import OpenAI

# client = OpenAI()

# INTENT_RULES = {
#     "eligibility_exceptions": [
#         "eligible","eligibility","exception","contractor","intern","part time","temporary","probation","new employee","not eligible","who qualifies","who is eligible","am i eligible",
#     ],

#     "employment_policy": [
#         "policy", "leave", "pto", "paid time off", "vacation",
#         "remote", "attendance", "termination", "hours"
#     ],
#     "benefits_compensation": [
#         "benefit", "insurance", "bonus", "salary",
#         "401k", "medical", "reimbursement"
#     ],
#     "hr_procedures": [
#         "how do i", "apply", "submit", "request",
#         "process", "approve", "workflow"
#     ],
#     "employee_handbook": [
#         "conduct", "behavior", "culture", "values",
#         "code of conduct"
#     ],
# }

# # Priority (who wins ties)
# INTENT_PRIORITY = [
#     "eligibility_exceptions",
#     "employment_policy",   
#     "hr_procedures",
#     "employee_handbook",
#     "benefits_compensation", 
# ]


# CLASSIFIER_PROMPT = """
# Classify the user's HR question into one of these categories:

# - employment_policy
# - eligibility_exceptions
# - benefits_compensation
# - hr_procedures
# - employee_handbook
# - unknown

# Return ONLY the label. No sentences.
# Question: "{q}"
# """

# def classify_intent(state: HRState) -> HRState:
#     query = state.user_query.lower()

#     # Score rule matches instead of first-hit
#     scores = {intent: 0 for intent in INTENT_RULES}

#     for intent, keywords in INTENT_RULES.items():
#         for k in keywords:
#             if k.lower() in query:
#                 scores[intent] += 1

#     # choose best rule match following priority order
#     best_intent = "unknown"
#     best_score = 0

#     for intent in INTENT_PRIORITY:
#         if scores[intent] > best_score:
#             best_intent = intent
#             best_score = scores[intent]

#     # If confidence is good — stop here, we consider 1+ signals as "confident"
#     if best_score >= 2:
#         state.intent = best_intent
#         state.trace.append(f"Intent classified (rules, confident): {best_intent}")
#         return state

#     # --- LLM fallback ---
#     try:
#         response = client.responses.create(
#             model="gpt-4.1-mini",
#             input=CLASSIFIER_PROMPT.format(q=state.user_query)
#         )
        
#         llm_label = response.output_text.strip()

#         # sanity check — prevent hallucinated labels
#         if llm_label in INTENT_RULES or llm_label == "unknown":
#             state.intent = llm_label
#             state.trace.append(f"Intent classified (LLM fallback): {llm_label}")
#         else:
#             state.intent = best_intent
#             state.trace.append(f"LLM returned invalid label ({llm_label}), kept rule result: {best_intent}")

#     except Exception as e:
#         state.intent = "unknown"
#         state.trace.append(f"LLM classification failed: {e}")

#     return state

from apps.agentic_hr.state.hr_state import HRState
from openai import OpenAI

client = OpenAI()

VALID_LABELS = {
    "employment_policy",
    "eligibility_exceptions",
    "benefits_compensation",
    "hr_procedures",
    "employee_handbook",
    "unknown",
}

CLASSIFIER_PROMPT = """
You are an HR AI assistant.

Your job is to classify the user's question into ONE category:

# - employment_policy
# - benefits_compensation
# - hr_procedures
# - employee_handbook
# - eligibility_exceptions
# - unknown

- employment_policy → rules about PTO, remote work, attendance, work hours, termination, leave rules
- benefits_compensation → salary, bonus, insurance, reimbursements, 401k, financial perks
- hr_procedures → how to do something (apply, submit, request, workflow, portal steps)
- employee_handbook → behavior, ethics, conduct, company culture, values, expectations
- eligibility_exceptions → who qualifies or does not qualify (interns, contractors, part-time, probation, exceptions)
- unknown → if none applies or unclear

Rules:
1. Return ONLY the label (no punctuation, no explanation)
2. If unsure, return: unknown

Question: "{q}"
"""


def classify_intent(state: HRState) -> HRState:
    query = state.user_query

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=CLASSIFIER_PROMPT.format(q=query),
        )

        label = response.output_text.strip()

        # Safety guard — prevent invalid categories
        if label not in VALID_LABELS:
            state.trace.append(f"LLM returned invalid label: {label}")
            label = "unknown"

        state.intent = label
        state.trace.append(f"Intent classified by LLM only: {label}")

    except Exception as e:
        state.intent = "unknown"
        state.trace.append(f"LLM classifier failed: {e}")

    return state