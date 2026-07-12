from functools import lru_cache

from fastapi import Request

from app.agent.executor import Executor
from app.agent.planner import Planner
from app.config import Settings, get_settings
from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.llm.service import LLMService
from app.services.agent_service import AgentService
from app.tools.code_explanation import CodeExplanationTool
from app.tools.compare_inputs import CompareInputsTool
from app.tools.conversational import ConversationalAnswerTool
from app.tools.registry import ToolRegistry
from app.tools.sentiment import SentimentAnalysisTool
from app.tools.summarize import SummarizeTool
from app.tools.youtube_transcript import YouTubeTranscriptTool


REQUEST_ID_STATE_KEY = "request_id"


def get_request_id(
    request: Request,
) -> str:
    """
    Return the request id assigned by request middleware.
    """

    request_id = getattr(
        request.state,
        REQUEST_ID_STATE_KEY,
        None,
    )

    if request_id is None:
        raise RuntimeError(
            "Request ID is unavailable."
        )

    return request_id


@lru_cache
def get_agent_service() -> AgentService:
    """
    Construct and cache the production application dependency graph.
    """

    settings: Settings = get_settings()

    primary_provider = GroqProvider(
        settings=settings,
    )

    fallback_provider = GeminiProvider(
        settings=settings,
    )

    llm_service = LLMService(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )

    planner = Planner(
        llm_service,
        max_plan_steps=settings.max_plan_steps,
    )

    tool_registry = ToolRegistry(
        tools=[
            SummarizeTool(
                llm_service=llm_service,
            ),
            SentimentAnalysisTool(
                llm_service=llm_service,
            ),
            ConversationalAnswerTool(
                llm_service=llm_service,
            ),
            CodeExplanationTool(
                llm_service=llm_service,
            ),
            YouTubeTranscriptTool(),
            CompareInputsTool(
                llm_service=llm_service,
            ),
        ]
    )

    executor = Executor(
        tool_registry=tool_registry,
        max_execution_steps=settings.max_execution_steps,
    )

    return AgentService(
        settings=settings,
        planner=planner,
        executor=executor,
    )