from collections.abc import Iterable

from app.agent.schemas import ToolName
from app.tools.base import AgentTool


class ToolRegistry:
    """
    Explicit registry of agent-callable tool implementations.

    Only authoritative ToolName values may be registered and resolved.
    Duplicate registrations are rejected.
    """

    def __init__(
        self,
        *,
        tools: Iterable[AgentTool],
    ) -> None:
        self._tools: dict[ToolName, AgentTool] = {}

        for tool in tools:
            self.register(tool)

    def register(
        self,
        tool: AgentTool,
    ) -> None:
        tool_name = tool.name

        if not isinstance(tool_name, ToolName):
            raise TypeError(
                "Registered tool name must be a ToolName."
            )

        if tool_name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool_name.value}"
            )

        self._tools[tool_name] = tool

    def get(
        self,
        tool_name: ToolName,
    ) -> AgentTool:
        if not isinstance(tool_name, ToolName):
            raise TypeError(
                "Tool lookup requires a ToolName."
            )

        try:
            return self._tools[tool_name]
        except KeyError as error:
            raise KeyError(
                f"Tool is not registered: {tool_name.value}"
            ) from error

    def registered_names(self) -> tuple[ToolName, ...]:
        return tuple(self._tools)