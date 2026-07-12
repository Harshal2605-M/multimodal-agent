import fitz
import io
import wave

from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
)
from app.config import Settings
from app.llm.service import LLMService
from app.models.input import NormalizedContext
from app.tools.compare_inputs import CompareInputsTool

from tests.e2e.fakes import (
    DeterministicLLMProvider,
    DeterministicWhisperTranscriber,
    build_e2e_agent_service,
)


PDF_TEXT = (
    "The deployment guide recommends automated tests before release. "
    "API documentation should be updated before deployment. "
    "The team should monitor production after release."
)

AUDIO_TRANSCRIPT = (
    "The engineering meeting discussed deployment preparation. "
    "The team agreed to run automated tests before release. "
    "They also recommended monitoring production after deployment."
)

EXPECTED_COMPARISON = (
    "Both inputs recommend automated testing before release and "
    "production monitoring after deployment. The PDF additionally "
    "requires updating API documentation before deployment."
)


def build_valid_pdf_bytes() -> bytes:
    document = fitz.open()

    try:
        page = document.new_page()

        text_rect = fitz.Rect(
            72,
            72,
            page.rect.width - 72,
            page.rect.height - 72,
        )

        result = page.insert_textbox(
            text_rect,
            PDF_TEXT,
            fontsize=11,
        )

        assert result >= 0

        return document.tobytes()

    finally:
        document.close()


def build_valid_wav_bytes() -> bytes:
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)

        wav_file.writeframes(
            b"\x00\x00" * 1_600
        )

    return buffer.getvalue()


def test_api_audio_pdf_cross_input_comparison_scenario(
    e2e_client,
    override_agent_service,
) -> None:
    settings = Settings(
        app_env="testing",
    )

    whisper = DeterministicWhisperTranscriber(
        transcript_segments=[
            (
                "The engineering meeting discussed "
                "deployment preparation."
            ),
            (
                "The team agreed to run automated tests "
                "before release."
            ),
            (
                "They also recommended monitoring production "
                "after deployment."
            ),
        ],
        duration_seconds=35.0,
        language="en",
        language_probability=0.98,
    )

    primary_provider = DeterministicLLMProvider(
        responses=[
            EXPECTED_COMPARISON,
        ]
    )

    fallback_provider = DeterministicLLMProvider(
        responses=[],
    )

    llm_service = LLMService(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )

    def build_comparison_plan(
        context: NormalizedContext,
    ) -> PlannerOutput:
        assert len(context.extracted_inputs) == 2

        source_ids = [
            extracted_input.source_id
            for extracted_input in context.extracted_inputs
        ]

        return PlannerOutput(
            goal=(
                "Compare the uploaded audio transcript "
                "and PDF document."
            ),
            constraints=[
                "Use only the two uploaded inputs.",
                "Identify similarities and differences.",
            ],
            needs_clarification=False,
            clarification_question=None,
            steps=[
                PlanStep(
                    id="step_1",
                    tool=ToolName.COMPARE_INPUTS,
                    input_reference=InputReference(
                        type=InputReferenceType.SOURCES,
                        source_ids=source_ids,
                    ),
                    depends_on=[],
                    reason=(
                        "Compare the extracted content "
                        "from both uploaded sources."
                    ),
                )
            ],
        )

    service, planner = build_e2e_agent_service(
        settings=settings,
        plan=build_comparison_plan,
        tools=[
            CompareInputsTool(
                llm_service=llm_service,
            ),
        ],
        audio_transcriber=whisper,
    )

    override_agent_service(service)

    response = e2e_client.post(
        "/agent/run",
        data={
            "query": (
                "Compare these two inputs. Identify their "
                "similarities and differences."
            ),
        },
        files=[
            (
                "files",
                (
                    "deployment_guide.pdf",
                    build_valid_pdf_bytes(),
                    "application/pdf",
                ),
            ),
            (
                "files",
                (
                    "engineering_meeting.wav",
                    build_valid_wav_bytes(),
                    "audio/wav",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["answer"] == EXPECTED_COMPARISON
    assert body["final_answer"] == EXPECTED_COMPARISON

    # Real preprocessing reached both extraction paths.
    assert len(whisper.calls) == 1

    assert len(planner.calls) == 1

    context = planner.calls[0]

    assert len(context.extracted_inputs) == 2

    pdf_input = context.extracted_inputs[0]
    audio_input = context.extracted_inputs[1]

    normalized_pdf_content = " ".join(
        pdf_input.content.split()
    )

    assert pdf_input.source_id.startswith("source_")
    assert pdf_input.filename == "deployment_guide.pdf"

    assert (
        "automated tests before release"
        in normalized_pdf_content
    )

    assert (
        "API documentation should be updated"
        in normalized_pdf_content
    )

    assert (
        "monitor production after release"
        in normalized_pdf_content
    )

    assert audio_input.source_id.startswith("source_")
    assert audio_input.filename == "engineering_meeting.wav"
    assert audio_input.content == AUDIO_TRANSCRIPT

    assert pdf_input.source_id != audio_input.source_id

    # Real CompareInputsTool reached deterministic LLM boundary.
    assert len(primary_provider.calls) == 1

    prompt = primary_provider.calls[0]

    normalized_prompt = " ".join(
        prompt.split()
    )

    assert (
        "automated tests before release"
        in normalized_prompt
    )

    assert (
        "API documentation should be updated"
        in normalized_prompt
    )

    assert (
        "monitor production after release"
        in normalized_prompt
    )

    assert AUDIO_TRANSCRIPT in normalized_prompt