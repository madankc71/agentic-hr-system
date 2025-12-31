from apps.agentic_hr.state.hr_state import HRState
from apps.agentic_hr.rag.vectorstore import search_benefits
from openai import OpenAI

client = OpenAI()

PROMPT = """
You are an HR assistant.

Answer using ONLY the benefit policy context.

If the answer is not clearly in the policy,
respond:

"I'm not fully sure — please verify with HR."

Question:
{q}

Policy context:
{ctx}
"""

def benefits_compensation_agent(state: HRState) -> HRState:
    query = state.user_query

    results = search_benefits(query)

    context = "\n\n".join(r["text"] for r in results)

    state.trace.append("Benefits search executed")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=PROMPT.format(q=query, ctx=context)
        )

        state.answer = response.output_text

    except Exception as e:
        state.answer = f"Benefits agent failed: {e}"

    return state
