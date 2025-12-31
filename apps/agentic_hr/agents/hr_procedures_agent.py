from apps.agentic_hr.state.hr_state import HRState
from apps.agentic_hr.rag.vectorstore import search_procedures
from openai import OpenAI

client = OpenAI()

PROCEDURE_PROMPT = """
You are an HR assistant. Use the policy excerpts below to answer the question clearly.

If something is not included in the excerpts, say:
"I’m not fully sure — please contact HR."

Policy excerpts:
{context}

Question:
{question}

Answer in 4–6 sentences.
"""


def hr_procedures_agent(state: HRState) -> HRState:
    query = state.user_query

    try:
        results = search_procedures(query)

        if not results:
            state.answer = "I don’t see a documented procedure for that. Please contact HR."
            state.trace.append("Procedure search returned no results")
            return state

        context_text = "\n\n".join(r["text"] for r in results)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=PROCEDURE_PROMPT.format(
                context=context_text,
                question=query
            )
        )

        state.answer = response.output_text.strip()
        state.sources.append("procedure_index")
        state.trace.append("Procedure agent answered using RAG and LLM")

    except Exception as e:
        state.answer = "Something went wrong retrieving procedures."
        state.trace.append(f"Procedure agent error: {e}")

    return state
