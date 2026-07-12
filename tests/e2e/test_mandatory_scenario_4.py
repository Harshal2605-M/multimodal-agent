import fitz

from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
)
from app.config import Settings
from app.llm.service import LLMService
from app.models.input import NormalizedContext, URLType
from app.tools.summarize import SummarizeTool
from app.tools.youtube_transcript import YouTubeTranscriptTool

from tests.e2e.fakes import (
    DeterministicLLMProvider,
    build_e2e_agent_service,
)


YOUTUBE_URL = "https://www.youtube.com/watch?v=abc123XYZ_0"
VIDEO_ID = "abc123XYZ_0"

PDF_TEXT = (
    "Please review the following engineering video and summarize it.\n"
    f"Video: {YOUTUBE_URL}\n"
)

TRANSCRIPT = (
    "The video explains deployment safety practices. "
    "Teams should run automated tests before release. "
    "API documentation should be updated before deployment. "
    "Production monitoring should be enabled after release."
)

EXPECTED_SUMMARY = (
    "The video recommends automated pre-release testing, updated API "
    "documentation, and production monitoring for safer deployments."
)


def build_valid_pdf_bytes() -> bytes:
    """
    Create a real text PDF containing one supported YouTube URL.
    """

    document = fitz.open()

    try:
        page = document.new_page()

        page.insert_text(
            (72, 72),
            PDF_TEXT,
        )

        return document.tobytes()

    finally:
        document.close()


def test_api_pdf_youtube_transcript_summary_scenario(
    e2e_client,
    override_agent_service,
) -> None:
    settings = Settings(
        app_env="testing",
    )

    transcript_calls: list[str] = []

    def deterministic_transcript_fetcher(
        video_id: str,
    ) -> str:
        transcript_calls.append(video_id)
        return TRANSCRIPT

    primary_provider = DeterministicLLMProvider(
        responses=[
            EXPECTED_SUMMARY,
        ]
    )

    fallback_provider = DeterministicLLMProvider(
        responses=[],
    )

    llm_service = LLMService(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )

    def build_youtube_summary_plan(
        context: NormalizedContext,
    ) -> PlannerOutput:
        assert len(context.extracted_inputs) == 1
        assert len(context.detected_urls) == 1

        detected_url = context.detected_urls[0]

        return PlannerOutput(
            goal=(
                "Retrieve the transcript for the YouTube URL "
                "found in the uploaded PDF and summarize it."
            ),
            constraints=[
                "Use the validated YouTube URL from the uploaded PDF.",
                "Summarize the retrieved transcript.",
            ],
            needs_clarification=False,
            clarification_question=None,
            steps=[
                PlanStep(
                    id="step_1",
                    tool=ToolName.YOUTUBE_TRANSCRIPT,
                    input_reference=InputReference(
                        type=InputReferenceType.SOURCE,
                        source_id=detected_url.source_id,
                    ),
                    depends_on=[],
                    reason=(
                        "Retrieve the transcript for the validated "
                        "YouTube URL."
                    ),
                ),
                PlanStep(
                    id="step_2",
                    tool=ToolName.SUMMARIZE,
                    input_reference=InputReference(
                        type=InputReferenceType.STEP_OUTPUT,
                        step_id="step_1",
                    ),
                    depends_on=["step_1"],
                    reason=(
                        "Summarize the retrieved YouTube transcript."
                    ),
                ),
            ],
        )

    service, planner = build_e2e_agent_service(
        settings=settings,
        plan=build_youtube_summary_plan,
        tools=[
            YouTubeTranscriptTool(
                transcript_fetcher=deterministic_transcript_fetcher,
            ),
            SummarizeTool(
                llm_service=llm_service,
            ),
        ],
    )

    override_agent_service(service)

    response = e2e_client.post(
        "/agent/run",
        data={
            "query": (
                "Summarize the YouTube video linked "
                "in this PDF."
            ),
        },
        files=[
            (
                "files",
                (
                    "video_notes.pdf",
                    build_valid_pdf_bytes(),
                    "application/pdf",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    body = response.json()

    print("\nBODY:")
    print(body)

    print("\nTRANSCRIPT CALLS:")
    print(transcript_calls)

    print("\nLLM CALLS:")
    print(primary_provider.calls)

    assert body["status"] == "completed"
    assert body["answer"] == EXPECTED_SUMMARY
    assert body["final_answer"] == EXPECTED_SUMMARY

    # Real PDF extraction and URL detection reached planner context.
    assert len(planner.calls) == 1

    context = planner.calls[0]

    assert len(context.extracted_inputs) == 1

    extracted_input = context.extracted_inputs[0]

    assert extracted_input.source_id
    assert extracted_input.source_id.startswith("source_")
    assert extracted_input.filename == "video_notes.pdf"

    assert YOUTUBE_URL in extracted_input.content

    assert len(context.detected_urls) == 1

    detected_url = context.detected_urls[0]

    assert str(detected_url.url) == YOUTUBE_URL
    assert detected_url.url_type is URLType.YOUTUBE
    assert detected_url.video_id == VIDEO_ID

    assert detected_url.source_id == extracted_input.source_id

    # Real YouTubeTranscriptTool reached deterministic API boundary.
    assert transcript_calls == [
        VIDEO_ID,
    ]

    # Real SummarizeTool received step_1 output.
    assert len(primary_provider.calls) == 1

    prompt = primary_provider.calls[0]

    assert TRANSCRIPT in prompt

    # Public response contains safe input metadata only.
    assert len(body["extracted_inputs"]) == 1

    response_input = body["extracted_inputs"][0]

    assert response_input["source_id"] == extracted_input.source_id
    assert response_input["filename"] == "video_notes.pdf"
    assert response_input["input_type"] == "pdf"

    assert "content" not in response_input

    # Verify two-step plan and successful execution.
    assert len(body["plan"]["steps"]) == 2

    assert (
        body["plan"]["steps"][0]["tool"]
        == "youtube_transcript"
    )

    assert (
        body["plan"]["steps"][0]["status"]
        == "success"
    )

    assert (
        body["plan"]["steps"][1]["tool"]
        == "summarize"
    )

    assert (
        body["plan"]["steps"][1]["status"]
        == "success"
    )

    # Frontend-ready execution metadata.
    assert body["metadata"]["total_plan_steps"] == 2
    assert body["metadata"]["executed_steps"] == 2
    assert body["metadata"]["successful_steps"] == 2
    assert body["metadata"]["failed_steps"] == 0
    assert body["metadata"]["skipped_steps"] == 0

    # Safe trace proves the intended workflow without chain-of-thought.
    plan_trace = body["plan_trace"]

    assert any(
        entry["stage"] == "planner"
        and entry["tool_name"] == "youtube_transcript"
        for entry in plan_trace
    )

    assert any(
        entry["stage"] == "planner"
        and entry["tool_name"] == "summarize"
        for entry in plan_trace
    )

    assert any(
        entry["stage"] == "workflow"
        and entry["message"] == "Execution completed."
        for entry in plan_trace
    )