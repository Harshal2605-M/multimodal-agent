import pytest

from app.agent.schemas import (
    ToolName,
    ToolResult,
    ToolStatus,
)
from app.tools.base import AgentTool, ToolInput
from app.tools.registry import ToolRegistry


class FakeTool(AgentTool):
    def __init__(
        self,
        name: ToolName,
    ) -> None:
        self._name = name

    @property
    def name(self) -> ToolName:
        return self._name

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        return ToolResult(
            step_id=tool_input.step_id,
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output="fake output",
        )


class InvalidNameTool(AgentTool):
    @property
    def name(self):  # type: ignore[override]
        return "unknown_tool"

    def run(
        self,
        tool_input: ToolInput,
    ) -> ToolResult:
        raise AssertionError(
            "Invalid tool should never execute."
        )


def test_registry_returns_registered_tool() -> None:
    tool = FakeTool(
        ToolName.SUMMARIZE
    )

    registry = ToolRegistry(
        tools=[tool],
    )

    assert registry.get(
        ToolName.SUMMARIZE
    ) is tool


def test_registry_rejects_duplicate_registration() -> None:
    first_tool = FakeTool(
        ToolName.SUMMARIZE
    )

    second_tool = FakeTool(
        ToolName.SUMMARIZE
    )

    with pytest.raises(
        ValueError,
        match="Tool already registered",
    ):
        ToolRegistry(
            tools=[
                first_tool,
                second_tool,
            ],
        )


def test_registry_rejects_unregistered_allowed_tool() -> None:
    registry = ToolRegistry(
        tools=[],
    )

    with pytest.raises(
        KeyError,
        match="Tool is not registered",
    ):
        registry.get(
            ToolName.SUMMARIZE
        )


def test_registry_rejects_unknown_registration_name() -> None:
    with pytest.raises(
        TypeError,
        match="must be a ToolName",
    ):
        ToolRegistry(
            tools=[InvalidNameTool()],
        )


def test_registry_rejects_unknown_lookup_name() -> None:
    registry = ToolRegistry(
        tools=[],
    )

    with pytest.raises(
        TypeError,
        match="lookup requires a ToolName",
    ):
        registry.get(  # type: ignore[arg-type]
            "unknown_tool"
        )


def test_registered_names_returns_registered_allowlist_names() -> None:
    registry = ToolRegistry(
        tools=[
            FakeTool(
                ToolName.SUMMARIZE
            ),
            FakeTool(
                ToolName.SENTIMENT_ANALYSIS
            ),
        ],
    )

    assert registry.registered_names() == (
        ToolName.SUMMARIZE,
        ToolName.SENTIMENT_ANALYSIS,
    )


def test_registered_tool_output_follows_tool_result_contract() -> None:
    registry = ToolRegistry(
        tools=[
            FakeTool(
                ToolName.SUMMARIZE
            )
        ],
    )

    tool = registry.get(
        ToolName.SUMMARIZE
    )

    result = tool.run(
        ToolInput(
            step_id="step_1",
            query="Summarize this.",
            texts=["Document content."],
        )
    )

    assert isinstance(
        result,
        ToolResult,
    )

    assert result.step_id == "step_1"
    assert result.tool_name is ToolName.SUMMARIZE
    assert result.status is ToolStatus.SUCCESS
    assert result.output == "fake output"