from pathlib import Path
from typing import cast

import pytest

from app.agent.executor import Executor
from app.agent.planner import Planner
from app.config import Settings
from app.models.response import AgentResponse, ResponseStatus
from app.services.agent_service import AgentService


class FakeGraph:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)

        return {
            "final_response": AgentResponse(
                request_id=state["request_id"],
                status=ResponseStatus.COMPLETED,
                answer="Service completed.",
                final_answer="Service completed.",
            )
        }


class FakeUploadBatch:
    def __init__(self) -> None:
        self.files = []
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        self.exited = True


@pytest.mark.asyncio
async def test_agent_service_runs_pipeline_and_cleans_upload_batch(
    monkeypatch,
) -> None:
    upload_batch = FakeUploadBatch()
    fake_graph = FakeGraph()

    async def fake_process_uploads(
        *,
        uploads,
        settings,
    ):
        return upload_batch

    def fake_extract_files(
        *,
        files,
        settings,
        audio_transcriber,
    ):
        from app.extractors.models import ExtractionBatch

        return ExtractionBatch(
            contents=[],
        )

    monkeypatch.setattr(
        "app.services.agent_service.process_uploads",
        fake_process_uploads,
    )

    monkeypatch.setattr(
        "app.services.agent_service.extract_files",
        fake_extract_files,
    )

    service = AgentService(
        settings=Settings(
            app_env="testing",
        ),
        planner=cast(
            Planner,
            object(),
        ),
        executor=cast(
            Executor,
            object(),
        ),
    )

    service._graph = fake_graph

    response = await service.run(
        request_id="request_1",
        query="Answer this.",
        uploads=[],
    )

    assert upload_batch.entered is True
    assert upload_batch.exited is True

    assert len(fake_graph.calls) == 1

    state = fake_graph.calls[0]

    assert state["request_id"] == "request_1"
    assert state["context"].query == "Answer this."

    assert response.status is ResponseStatus.COMPLETED
    assert response.answer == "Service completed."