import os
from openai import OpenAI

from apps.agentic_hr.state.hr_state import HRState
from apps.agentic_hr.rag.vectorstore import search_policies


client = OpenAI()


# PROMPT_TEMPLATE = """
# You are an HR policy assistant.

# Use ONLY the provided policy text to answer the question.
# If the answer is not there, say you don't know.

# User question:
# {question}

# Policy evidence:
# {evidence}

# Write a clear answer in 2–3 sentences. Avoid hallucinations.
# """

PROMPT_TEMPLATE = """
You are an HR policy assistant.

Answer ONLY using the policy text below.

Rules:
1. If the answer is NOT clearly stated, say:
   "I’m not sure — this policy is not clearly documented here."
2. Do NOT guess.
3. Do NOT add outside assumptions.
4. Keep the answer to 2–3 short sentences.

Policy text:
---
{evidence}
---

Question:
{question}
"""


def employment_policy_agent(state: HRState) -> HRState:
    state.trace.append("Employment policy agent running (RAG + LLM)...")

    results = search_policies(state.user_query)

    if not results:
        state.answer = "I checked the HR policy database but found nothing relevant."
        state.trace.append("No RAG results")
        return state

    evidence_text = "\n\n---\n\n".join(r["text"] for r in results)

    prompt = PROMPT_TEMPLATE.format(
        question=state.user_query,
        evidence=evidence_text
    )

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        answer = response.output_text

        state.answer = answer.strip()

        state.sources = [r["meta"]["filename"] for r in results]
        state.trace.append("Answer generated using RAG + LLM summarization")

        return state

    except Exception as e:
        state.answer = "Policy lookup failed due to an internal error."
        state.trace.append(f"LLM error: {str(e)}")
        return state