from typing import cast

from app.agent.executor import Executor
from app.agent.graph import build_agent_graph
from app.agent.planner import Planner
from app.agent.state import AgentState


class UnusedPlanner:
    def create_plan(self, context):
        raise AssertionError(
            "Planner should not be called while building the graph."
        )


class UnusedExecutor:
    def execute_current_step(
        self,
        state: AgentState,
    ) -> dict[str, object]:
        raise AssertionError(
            "Executor should not be called while building the graph."
        )


def test_build_agent_graph_returns_invocation_ready_graph() -> None:
    graph = build_agent_graph(
        planner=cast(
            Planner,
            UnusedPlanner(),
        ),
        executor=cast(
            Executor,
            UnusedExecutor(),
        ),
    )

    assert callable(graph.invoke)