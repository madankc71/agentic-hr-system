# Agentic HR Assistant (RAG + LLM + Evaluation)

This project explores how an HR question-answering system can be designed so that it answers **from policy documents**, instead of guessing or hallucinating.

The system combines:

- retrieval-augmented generation (RAG)
- an intent-driven agent workflow
- LLM reasoning layered on top of retrieved text
- simple, reproducible evaluation

It is intentionally built as something between an engineering prototype and a research playground.

---

## Why I Built This

Most “chatbots” say something even when they shouldn’t.

In HR, that is dangerous.

Here the idea was:

> The model should only answer when there is policy support - otherwise it should say it doesn’t know.

This project helped me explore:

- grounded responses vs hallucinations  
- routing questions to the right domain  
- evaluation as part of development (not an afterthought)

---

## System Design (High-Level)

The pipeline looks like this:

1. **User asks a question**
2. **Intent is classified**  
   I first used deterministic rules, and later added an **LLM classifier with safety checks**.
3. **Question is routed to the right agent**
4. The agent retrieves relevant policy text (FAISS + sentence-transformers)
5. The LLM summarizes only what was retrieved
6. If the answer isn’t present, it explicitly says so

The decision flow uses **LangGraph** to model agents and state transitions.

---

## Repository Structure

```
apps/agentic_hr/
 ├── api.py                # FastAPI entrypoint
 ├── graph.py              # LangGraph workflow
 ├── state/                # Shared conversation state
 ├── nodes/                # Intent classifier
 ├── agents/               # HR domain agents
 ├── rag/
 │   ├── loaders/          # load datasets
 │   ├── chunking/         # text splitting
 │   ├── indexes/          # FAISS indexes
 │   └── vectorstore.py
 └── evaluation/
     ├── golden.json       # golden dataset
     └── evaluate_intents.py
```

Datasets live in:

```
data/
```

(leave policies, benefits, procedures, eligibility rules, employee handbook, etc.)

---

## RAG Pipeline

Each domain (policies, benefits, eligibility, procedures, handbook):

- loads text  
- chunks it  
- embeds with sentence-transformers  
- builds a FAISS index  
- retrieves nearest passages per query  

Then the LLM answers **only using retrieved text**.

If nothing supports the answer, the assistant says:

> “I’m not sure - please verify with HR.”

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Build indexes (example):

```bash
python -m apps.agentic_hr.rag.build_policy_index
```

Start the API:

```bash
uvicorn apps.agentic_hr.api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## Evaluation

The project includes a small “golden set” and evaluation script.

It measures:

- whether intent routing is correct  
- whether answers remain grounded in policy text  

Run:

```bash
python -m apps.agentic_hr.evaluation.evaluate_intents
```

This helped me iterate and see failure modes instead of guessing.

---

## What I Learned / Research Value

Working on this taught me:

- RAG is powerful, but grounding must be measured explicitly  
- rule-based classifiers are simple but brittle  
- LLM-only classification handles subtle questions better  
- evaluation strongly influences architecture choices  

This provides a base to explore:

- stronger grounding metrics
- multi-document retrieval strategies
- conversational memory
- compliance-aware assistants

---

## Future Work

Some directions I’d like to continue:

- citations inline (like academic references)
- adversarial question testing
- ingesting PDFs / DOCX / web pages
- comparing classifiers (rules vs LLM vs hybrids)

---

Thanks for reading - I see this project as a bridge between practical engineering work and early-stage research.
