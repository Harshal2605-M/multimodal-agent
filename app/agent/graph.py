from langgraph.graph import END, START, StateGraph

from app.agent.executor import Executor
from app.agent.nodes import (
    AgentNodes,
    create_executor_node,
)
from app.agent.planner import Planner
from app.agent.routers import (
    clarification_router,
    execution_router,
)
from app.agent.state import AgentState


def build_agent_graph(
    *,
    planner: Planner,
    executor: Executor,
):
    """
    Build and compile the LangGraph workflow.

    Ambiguous requests terminate after clarification without
    reaching the executor.

    Clear requests execute one plan step per executor-node call.
    Execution loops while plan steps remain, then routes to the
    response composer before terminating.
    """

    nodes = AgentNodes(
        planner=planner,
    )

    executor_node = create_executor_node(
        executor
    )

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
        executor_node,
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

    workflow.add_conditional_edges(
        "executor",
        execution_router,
        {
            "executor": "executor",
            "response_composer": "response_composer",
        },
    )

    workflow.add_edge(
        "response_composer",
        END,
    )

    return workflow.compile()