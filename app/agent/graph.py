from langgraph.graph import END, START, StateGraph

from app.agent.nodes import AgentNodes
from app.agent.routers import clarification_router
from app.agent.state import AgentState


def build_agent_graph(
    *,
    nodes: AgentNodes,
):
    """
    Build and compile the Phase 8 LangGraph workflow.

    Ambiguous requests terminate after clarification without reaching
    the executor.

    Clear requests reach the placeholder executor, then the temporary
    response composer, before terminating.
    """

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "planner",
        nodes.planner_node,
    )

    workflow.add_node(
        "plan_validation",
        nodes.plan_validation_node,
    )

    workflow.add_node(
        "clarify",
        nodes.clarify_node,
    )

    workflow.add_node(
        "executor",
        nodes.executor_node,
    )

    workflow.add_node(
        "response_composer",
        nodes.response_composer_node,
    )

    workflow.add_edge(
        START,
        "planner",
    )

    workflow.add_edge(
        "planner",
        "plan_validation",
    )

    workflow.add_conditional_edges(
        "plan_validation",
        clarification_router,
        {
            "clarify": "clarify",
            "executor": "executor",
        },
    )

    workflow.add_edge(
        "clarify",
        END,
    )

    workflow.add_edge(
        "executor",
        "response_composer",
    )

    workflow.add_edge(
        "response_composer",
        END,
    )

    return workflow.compile()