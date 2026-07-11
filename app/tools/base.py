from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from app.agent.schemas import ToolName, ToolResult


class ToolInput(BaseModel):
    """
    Executor-resolved input passed to one agent-callable tool.

    Tools receive only application-resolved data. They do not inspect
    AgentState, resolve planner references, or select their own inputs.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    step_id: str = Field(
        min_length=1,
        max_length=100,
    )

    query: str = Field(
        default="",
        max_length=10_000,
    )

    texts: list[str] = Field(
        default_factory=list,
    )

    urls: list[str] = Field(
        default_factory=list,
    )


class AgentTool(ABC):
    """
    Base contract implemented by every agent-callable tool.
    """

    @property
    @abstractmethod
    def name(self) -> ToolName:
        """
        Return the authoritative ToolName handled by this tool.
        """

    @abstractmethod
    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        """
        Execute the tool independently and return a ToolResult.
        """