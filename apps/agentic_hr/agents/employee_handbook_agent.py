from apps.agentic_hr.state.hr_state import HRState
from apps.agentic_hr.rag.vectorstore import search_handbook
from openai import OpenAI

client = OpenAI()

ANSWER_PROMPT = """
You are an HR assistant.

Use ONLY the information in the handbook passages below.

If the information is missing, say:
"I don't have enough information in the handbook to answer that."

Answer clearly and concisely.

Handbook passages:
---
{context}
---

Question: {question}
"""


def employee_handbook_agent(state: HRState) -> HRState:
    state.trace.append("Employee handbook agent activated")

    results = search_handbook(state.user_query)

    if not results:
        state.answer = "I could not find anything related in the employee handbook."
        return state

    context = "\n\n".join(results)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=ANSWER_PROMPT.format(
                context=context,
                question=state.user_query,
            ),
        )

        state.answer = response.output_text.strip()
        state.sources = results
        state.trace.append("Answer generated using RAG + LLM")

    except Exception as e:
        state.answer = "I found handbook information, but couldn't generate a response."
        state.trace.append(f"LLM failed: {e}")

    return state
