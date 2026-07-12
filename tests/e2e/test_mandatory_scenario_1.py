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
from app.tools.summarize import SummarizeTool

from tests.e2e.fakes import (
    DeterministicLLMProvider,
    DeterministicWhisperTranscriber,
    build_e2e_agent_service,
)
from app.models.input import NormalizedContext


TRANSCRIPT = (
    "The engineering team reviewed the deployment plan. "
    "They agreed to add automated tests before release. "
    "The API documentation must be updated. "
    "The team will monitor production after deployment."
)

EXPECTED_SUMMARY = """Transcript:
The engineering team reviewed the deployment plan. They agreed to add automated tests before release. The API documentation must be updated. The team will monitor production after deployment.

1-Line Summary:
The team reviewed deployment preparations, testing, documentation, and production monitoring.

3 Bullet Points:
- The engineering team reviewed the deployment plan.
- Automated tests and API documentation updates are required before release.
- The team will monitor production after deployment.

5-Sentence Summary:
The engineering team reviewed the deployment plan.
They agreed to add automated tests before release.
The API documentation must be updated.
The team plans to monitor production after deployment.
The discussion focused on preparing for a reliable release."""


def build_valid_wav_bytes() -> bytes:
    """
    Create a structurally valid short WAV file entirely in memory.

    The fake transcriber prevents actual audio decoding or Whisper
    inference, while the real upload gateway validates the WAV header.
    """

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)

        wav_file.writeframes(
            b"\x00\x00" * 1_600
        )

    return buffer.getvalue()


def test_api_audio_transcription_and_summary_scenario(
    e2e_client,
    override_agent_service,
) -> None:
    settings = Settings(
        app_env="testing",
    )

    whisper = DeterministicWhisperTranscriber(
        transcript_segments=[
            "The engineering team reviewed the deployment plan.",
            "They agreed to add automated tests before release.",
            "The API documentation must be updated.",
            "The team will monitor production after deployment.",
        ],
        duration_seconds=42.5,
        language="en",
        language_probability=0.99,
    )

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

    def build_audio_summary_plan(
        context: NormalizedContext,
    ) -> PlannerOutput:
        assert len(context.extracted_inputs) == 1

        source_id = context.extracted_inputs[0].source_id

        return PlannerOutput(
            goal=(
                "Produce the requested structured audio "
                "transcription and summary response."
            ),
            constraints=[
                "Use only the uploaded audio transcript.",
                (
                    "Return the transcript, one-line summary, "
                    "exactly three bullet points, and a "
                    "five-sentence summary."
                ),
            ],
            needs_clarification=False,
            clarification_question=None,
            steps=[
                PlanStep(
                    id="step_1",
                    tool=ToolName.SUMMARIZE,
                    input_reference=InputReference(
                        type=InputReferenceType.SOURCE,
                        source_id=source_id,
                    ),
                    depends_on=[],
                    reason=(
                        "Summarize the extracted audio transcript."
                    ),
                )
            ],
        )

    service, planner = build_e2e_agent_service(
        settings=settings,
        plan=build_audio_summary_plan,
        tools=[
            SummarizeTool(
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
                "Transcribe this audio and provide the duration, "
                "a 1-line summary, 3 bullet points, and a "
                "5-sentence summary."
            ),
        },
        files=[
            (
                "files",
                (
                    "meeting.wav",
                    build_valid_wav_bytes(),
                    "audio/wav",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["answer"] == EXPECTED_SUMMARY
    assert body["final_answer"] == EXPECTED_SUMMARY

    assert len(whisper.calls) == 1

    audio_path, transcribe_kwargs = whisper.calls[0]

    assert audio_path.endswith(".wav")

    assert transcribe_kwargs == {
        "beam_size": 1,
        "vad_filter": True,
    }

    assert len(planner.calls) == 1

    context = planner.calls[0]

    assert context.query == (
        "Transcribe this audio and provide the duration, "
        "a 1-line summary, 3 bullet points, and a "
        "5-sentence summary."
    )
    assert len(context.extracted_inputs) == 1

    extracted_input = context.extracted_inputs[0]

    assert extracted_input.source_id
    assert extracted_input.source_id.startswith("source_")
    assert extracted_input.filename == "meeting.wav"
    assert extracted_input.content == TRANSCRIPT

    assert extracted_input.metadata.duration_seconds == 42.5
    assert extracted_input.metadata.language == "en"

    assert len(primary_provider.calls) == 1

    prompt = primary_provider.calls[0]

    assert TRANSCRIPT in prompt

    assert len(body["extracted_inputs"]) == 1

    response_input = body["extracted_inputs"][0]

    assert (
        response_input["source_id"]
        == extracted_input.source_id
    )

    assert response_input["filename"] == "meeting.wav"
    assert response_input["input_type"] == "audio"

    assert response_input["metadata"]["duration_seconds"] == 42.5
    assert response_input["metadata"]["language"] == "en"

    assert "content" not in response_input

    assert body["plan"]["steps"][0]["tool"] == "summarize"
    assert body["plan"]["steps"][0]["status"] == "success"

    assert body["metadata"]["total_plan_steps"] == 1
    assert body["metadata"]["executed_steps"] == 1
    assert body["metadata"]["successful_steps"] == 1
    assert body["metadata"]["failed_steps"] == 0
    assert body["metadata"]["skipped_steps"] == 0

    final_answer = body["final_answer"]

    assert "Transcript:" in final_answer
    assert TRANSCRIPT in final_answer

    assert "1-Line Summary:" in final_answer

    assert "3 Bullet Points:" in final_answer

    bullet_section = (
        final_answer
        .split("3 Bullet Points:", 1)[1]
        .split("5-Sentence Summary:", 1)[0]
    )

    bullets = [
        line
        for line in bullet_section.splitlines()
        if line.strip().startswith("- ")
    ]

    assert len(bullets) == 3

    assert "5-Sentence Summary:" in final_answer

    five_sentence_section = (
        final_answer
        .split("5-Sentence Summary:", 1)[1]
        .strip()
    )

    sentences = [
        sentence.strip()
        for sentence in five_sentence_section.split(".")
        if sentence.strip()
    ]

    assert len(sentences) == 5

    assert (
        body["extracted_inputs"][0]
        ["metadata"]
        ["duration_seconds"]
        == 42.5
    )