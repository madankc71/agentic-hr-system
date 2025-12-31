from langgraph.graph import StateGraph, END

from apps.agentic_hr.state.hr_state import HRState
from apps.agentic_hr.nodes.intent_classifier import classify_intent
from apps.agentic_hr.nodes.router import route_by_intent
from apps.agentic_hr.agents.employment_policy_agent import employment_policy_agent
from apps.agentic_hr.agents.benefits_compensation_agent import benefits_compensation_agent
from apps.agentic_hr.agents.employee_handbook_agent import employee_handbook_agent
from apps.agentic_hr.agents.hr_procedures_agent import hr_procedures_agent
from apps.agentic_hr.agents.eligibility_exceptions_agent import eligibility_exceptions_agent


def build_hr_graph():
    """
    Builds and returns the LangGraph workflow for the HR agentic system.
    """

    graph = StateGraph(HRState)

    # Add nodes
    graph.add_node("classifier", classify_intent)
    # graph.add_node("router", route_by_intent)
    graph.add_node("employment_policy_agent", employment_policy_agent)
    graph.add_node("benefits_compensation_agent", benefits_compensation_agent)
    graph.add_node("employee_handbook_agent", employee_handbook_agent)
    graph.add_node("hr_procedures_agent", hr_procedures_agent)
    graph.add_node("eligibility_exceptions_agent", eligibility_exceptions_agent)

    # Entry point
    graph.set_entry_point("classifier")

    # graph.add_edge("classifier", "router")

    # Conditional routing after classifier
    graph.add_conditional_edges(
        "classifier",
        route_by_intent,
        {
            "employment_policy_agent": "employment_policy_agent",
            "benefits_compensation_agent": "benefits_compensation_agent",
            "employee_handbook_agent": "employee_handbook_agent",
            "hr_procedures_agent": "hr_procedures_agent",
            "eligibility_exceptions_agent": "eligibility_exceptions_agent", 
            # All other intents go to END for now
            "fallback_agent": END,
        },
    )

    # End after the agent runs
    graph.add_edge("employment_policy_agent", END)
    graph.add_edge("benefits_compensation_agent", END)
    graph.add_edge("employee_handbook_agent", END)
    graph.add_edge("hr_procedures_agent", END)
    graph.add_edge("eligibility_exceptions_agent", END)

    return graph.compile()