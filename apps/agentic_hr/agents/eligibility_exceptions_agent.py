from apps.agentic_hr.state.hr_state import HRState
from apps.agentic_hr.rag.vectorstore import search_eligibility
from openai import OpenAI

client = OpenAI()

ELIGIBILITY_PROMPT = """
You are an HR assistant specializing in eligibility rules and exceptions
(e.g., contractors, interns, part-time employees, probation periods).

Use ONLY the context below to answer the user's question.
If the answer is unclear or not covered, say you are not sure
and recommend contacting HR for confirmation.

Context:
{context}

User question:
{question}

Answer in 2–4 concise sentences. If rules differ by employment type
(full-time vs contractor vs intern), clearly call that out.
"""


def eligibility_exceptions_agent(state: HRState) -> HRState:
    query = state.user_query

    try:
        # 1) Retrieve relevant chunks from vector store
        results = search_eligibility(query, top_k=3)

        if not results:
            state.answer = (
                "I couldn't find any eligibility rules related to your question "
                "in the current knowledge base. Please contact HR directly to confirm."
            )
            state.sources.append("eligibility_index:empty")
            state.trace.append("Eligibility agent: no matches in index")
            return state

        context_text = "\n\n---\n\n".join(r["text"] for r in results)

        # 2) Ask the LLM to reason over those chunks
        prompt = ELIGIBILITY_PROMPT.format(
            context=context_text,
            question=query,
        )

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        answer = response.output_text.strip()

        # 3) Update state
        if not state.intent:
            state.intent = "eligibility_exceptions"

        state.answer = answer
        state.sources.append("eligibility_index")
        state.trace.append("Eligibility agent: answered with RAG+LLM")

        return state

    except Exception as e:
        # Defensive: never crash the graph
        state.trace.append(f"Eligibility agent error: {e}")
        state.answer = (
            "I ran into an issue while looking up eligibility rules. "
            "Please try again later or check with HR directly."
        )
        return state
