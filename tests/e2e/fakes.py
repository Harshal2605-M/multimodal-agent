from app.agent.schemas import PlannerOutput
from app.llm.models import (
    LLMProviderName,
    LLMStructuredGenerationResult,
)
from app.models.input import NormalizedContext
from pydantic import BaseModel

from app.llm.base import BaseLLMProvider
from app.llm.models import (
    LLMGenerationResult,
    LLMProviderName,
    LLMStructuredGenerationResult,
)
from typing import cast

from app.agent.executor import Executor
from app.agent.planner import Planner
from app.config import Settings
from app.llm.service import LLMService
from app.services.agent_service import AgentService
from app.tools.base import AgentTool
from app.tools.registry import ToolRegistry


class DeterministicPlanner:
    """
    Planner test double returning one predefined validated plan.

    The real graph, executor, registry, and tools remain active.
    """

    def __init__(
        self,
        plan: PlannerOutput,
    ) -> None:
        self._plan = plan
        self.calls: list[NormalizedContext] = []

    def create_plan(
        self,
        context: NormalizedContext,
    ) -> LLMStructuredGenerationResult:
        self.calls.append(context)

        return LLMStructuredGenerationResult(
            output=self._plan,
            provider_used=LLMProviderName.GROQ,
        )
    
class DeterministicLLMProvider(BaseLLMProvider):
    """
    External LLM boundary replacement for API E2E tests.

    Responses are consumed in invocation order.
    """

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    @property
    def name(self) -> LLMProviderName:
        return LLMProviderName.GROQ

    def generate(
        self,
        prompt: str,
    ) -> LLMGenerationResult:
        self.calls.append(prompt)

        if not self._responses:
            raise AssertionError(
                "No deterministic LLM response remains."
            )

        return LLMGenerationResult(
            content=self._responses.pop(0),
            provider_used=self.name,
        )

    def generate_structured(
        self,
        prompt: str,
        output_model: type[BaseModel],
    ) -> LLMStructuredGenerationResult:
        raise AssertionError(
            "Structured generation is not expected because "
            "the planner is deterministic in API E2E tests."
        )
    

def build_e2e_agent_service(
    *,
    settings: Settings,
    plan: PlannerOutput,
    tools: list[AgentTool],
) -> tuple[
    AgentService,
    DeterministicPlanner,
]:
    planner = DeterministicPlanner(plan)

    registry = ToolRegistry(
        tools=tools,
    )

    executor = Executor(
        tool_registry=registry,
        max_execution_steps=settings.max_execution_steps,
    )

    service = AgentService(
        settings=settings,
        planner=cast(
            Planner,
            planner,
        ),
        executor=executor,
    )

    return service, planner