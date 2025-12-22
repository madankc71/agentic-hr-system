from langgraph.graph import StateGraph, END

from apps.agentic_hr.state.hr_state import HRState
from apps.agentic_hr.nodes.intent_classifier import classify_intent
from apps.agentic_hr.nodes.router import route_by_intent
from apps.agentic_hr.nodes.employment_policy_agent import employment_policy_agent


def build_hr_graph():
    """
    Builds and returns the LangGraph workflow for the HR agentic system.
    """

    graph = StateGraph(HRState)

    # Add nodes
    graph.add_node("classifier", classify_intent)
    graph.add_node("employment_policy_agent", employment_policy_agent)

    # Entry point
    graph.set_entry_point("classifier")

    # Conditional routing after classifier
    graph.add_conditional_edges(
        "classifier",
        route_by_intent,
        {
            "employment_policy_agent": "employment_policy_agent",
            # All other intents go to END for now
            "benefits_compensation_agent": END,
            "hr_procedures_agent": END,
            "employee_handbook_agent": END,
            "eligibility_exceptions_agent": END,
            "fallback_agent": END,
        },
    )

    # End after the agent runs
    graph.add_edge("employment_policy_agent", END)

    return graph.compile()