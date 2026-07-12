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
from types import SimpleNamespace
from app.extractors.audio import WhisperTranscriber
from collections.abc import Callable

PlanFactory = Callable[
    [NormalizedContext],
    PlannerOutput,
]
class DeterministicPlanner:
    """
    Deterministic planner test double.

    Supports:
    - fixed plans for query-only scenarios
    - context-aware plan factories for uploaded-source scenarios
    """

    def __init__(
        self,
        plan: PlannerOutput | PlanFactory,
    ) -> None:
        self._plan = plan
        self.calls: list[NormalizedContext] = []

    def create_plan(
        self,
        context: NormalizedContext,
    ) -> LLMStructuredGenerationResult:
        self.calls.append(context)

        if callable(self._plan):
            plan = self._plan(context)
        else:
            plan = self._plan

        return LLMStructuredGenerationResult(
            output=plan,
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
    

class FakeWhisperSegment:
    def __init__(
        self,
        text: str,
    ) -> None:
        self.text = text


class DeterministicWhisperTranscriber:
    """
    Deterministic faster-whisper boundary replacement.

    Returns predefined transcript segments and audio metadata while
    preserving the real extract_audio() implementation.
    """

    def __init__(
        self,
        *,
        transcript_segments: list[str],
        duration_seconds: float,
        language: str = "en",
        language_probability: float = 0.99,
    ) -> None:
        self._transcript_segments = transcript_segments
        self._duration_seconds = duration_seconds
        self._language = language
        self._language_probability = language_probability

        self.calls: list[
            tuple[str, dict[str, object]]
        ] = []

    def transcribe(
        self,
        audio: str,
        **kwargs,
    ):
        self.calls.append(
            (
                audio,
                dict(kwargs),
            )
        )

        segments = [
            FakeWhisperSegment(text)
            for text in self._transcript_segments
        ]

        info = SimpleNamespace(
            duration=self._duration_seconds,
            language=self._language,
            language_probability=(
                self._language_probability
            ),
        )

        return segments, info
    

def build_e2e_agent_service(
    *,
    settings: Settings,
    plan: PlannerOutput | PlanFactory,
    tools: list[AgentTool],
    audio_transcriber: WhisperTranscriber | None = None,
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
        audio_transcriber=audio_transcriber,
    )

    return service, planner