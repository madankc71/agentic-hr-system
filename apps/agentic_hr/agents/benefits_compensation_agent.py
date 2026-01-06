from apps.agentic_hr.state.hr_state import HRState
from apps.agentic_hr.rag.vectorstore import search_benefits
from openai import OpenAI

client = OpenAI()

# PROMPT = """
# You are an HR assistant.

# Use ONLY the benefits policy excerpts below.

# If the answer is unclear, say:
# "I'm not fully sure — please verify with HR."

# Benefits excerpts:
# ---
# {ctx}
# ---

# Question: {q}

# Answer concisely.
# """

PROMPT = """
You are an HR assistant specializing in benefits.

Answer ONLY using the benefit policy text below.

Rules:
1. If the answer is NOT clearly there, say:
   "I’m not sure — please verify with HR."
2. Do NOT guess or invent details.
3. Keep the answer brief (2–3 sentences).

Benefit policy text:
---
{ctx}
---

Question:
{q}
"""

def benefits_compensation_agent(state: HRState) -> HRState:
    query = state.user_query
    state.trace.append("Benefits agent activated")

    try:
        results = search_benefits(query)

        if not results:
            state.answer = "I couldn't find benefits information related to that."
            state.trace.append("Benefits index returned no results")
            return state

        context = "\n\n".join(r["text"] for r in results)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=PROMPT.format(q=query, ctx=context)
        )

        state.answer = response.output_text.strip()
        state.sources = ["benefits_index"]
        state.trace.append("Benefits answered using RAG + LLM")

    except Exception as e:
        state.answer = "Benefits lookup failed — please try again later."
        state.trace.append(f"Benefits agent error: {e}")

    return state