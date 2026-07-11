from typing import cast

from app.agent.graph import build_agent_graph
from app.agent.nodes import AgentNodes
from app.agent.planner import Planner


class UnusedPlanner:
    def create_plan(self, context):
        raise AssertionError(
            "Planner should not run while only building the graph."
        )


def test_build_agent_graph_returns_invocation_ready_graph() -> None:
    nodes = AgentNodes(
        planner=cast(
            Planner,
            UnusedPlanner(),
        )
    )

    graph = build_agent_graph(
        nodes=nodes,
    )

    assert callable(graph.invoke)