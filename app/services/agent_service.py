from fastapi import UploadFile

from app.agent.executor import Executor
from app.agent.graph import build_agent_graph
from app.agent.planner import Planner
from app.agent.state import create_initial_state
from app.config import Settings
from app.extractors.audio import WhisperTranscriber
from app.extractors.orchestrator import extract_files
from app.models.response import AgentResponse
from app.security.upload_gateway import process_uploads
from app.utils.context_builder import build_normalized_context


class AgentService:
    """
    Application-level orchestration boundary for one complete
    multimodal agent request.

    Owns preprocessing, context construction, workflow invocation,
    and request-scoped upload cleanup.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        planner: Planner,
        executor: Executor,
        audio_transcriber: WhisperTranscriber | None = None,
    ) -> None:
        self._settings = settings
        self._audio_transcriber = audio_transcriber

        self._graph = build_agent_graph(
            planner=planner,
            executor=executor,
        )

    async def run(
        self,
        *,
        request_id: str,
        query: str,
        uploads: list[UploadFile],
        clarification_answer: str | None = None,
    ) -> AgentResponse:
        """
        Execute one request through the complete application pipeline.
        """

        upload_batch = await process_uploads(
            uploads=uploads,
            settings=self._settings,
        )

        async with upload_batch:
            extraction_batch = extract_files(
                files=upload_batch.files,
                settings=self._settings,
                audio_transcriber=self._audio_transcriber,
            )

            context = build_normalized_context(
                query=query,
                files=upload_batch.files,
                extraction_batch=extraction_batch,
            )

            initial_state = create_initial_state(
                request_id=request_id,
                context=context,
                clarification_answer=clarification_answer,
            )

            result = self._graph.invoke(initial_state)

            final_response = result["final_response"]

            if final_response is None:
                raise RuntimeError(
                    "Agent workflow completed without a final response."
                )

            return final_response