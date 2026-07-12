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
from app.models.input import NormalizedContext
from app.tools.conversational import ConversationalAnswerTool

from tests.e2e.fakes import (
    DeterministicLLMProvider,
    build_e2e_agent_service,
)


PDF_TEXT = (
    "Project Release Meeting\n"
    "The team reviewed the upcoming production release.\n"
    "Alice will add automated integration tests before Friday.\n"
    "Bob will update the API documentation before deployment.\n"
    "The DevOps team will configure production monitoring alerts.\n"
    "The release is scheduled for next Monday."
)


EXPECTED_ACTION_ITEMS = (
    "- Alice: Add automated integration tests before Friday.\n"
    "- Bob: Update the API documentation before deployment.\n"
    "- DevOps team: Configure production monitoring alerts."
)


def build_valid_pdf_bytes() -> bytes:
    """
    Create a real one-page text PDF entirely in memory.

    The real PDF extractor must process this document through
    direct text extraction.
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


def test_api_pdf_action_items_only_scenario(
    e2e_client,
    override_agent_service,
) -> None:
    settings = Settings(
        app_env="testing",
    )

    primary_provider = DeterministicLLMProvider(
        responses=[
            EXPECTED_ACTION_ITEMS,
        ]
    )

    fallback_provider = DeterministicLLMProvider(
        responses=[],
    )

    llm_service = LLMService(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )

    def build_action_items_plan(
        context: NormalizedContext,
    ) -> PlannerOutput:
        assert len(context.extracted_inputs) == 1

        source_id = context.extracted_inputs[0].source_id

        return PlannerOutput(
            goal="Extract only the action items from the uploaded PDF.",
            constraints=[
                "Use only the uploaded PDF content.",
                "Return only action items.",
                "Do not include summaries or unrelated information.",
            ],
            needs_clarification=False,
            clarification_question=None,
            steps=[
                PlanStep(
                    id="step_1",
                    tool=ToolName.CONVERSATIONAL_ANSWER,
                    input_reference=InputReference(
                        type=InputReferenceType.SOURCE,
                        source_id=source_id,
                    ),
                    depends_on=[],
                    reason=(
                        "Answer the user's action-item request "
                        "using the extracted PDF content."
                    ),
                )
            ],
        )

    service, planner = build_e2e_agent_service(
        settings=settings,
        plan=build_action_items_plan,
        tools=[
            ConversationalAnswerTool(
                llm_service=llm_service,
            ),
        ],
    )

    override_agent_service(service)

    response = e2e_client.post(
        "/agent/run",
        data={
            "query": "What are the action items?",
        },
        files=[
            (
                "files",
                (
                    "release_meeting.pdf",
                    build_valid_pdf_bytes(),
                    "application/pdf",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["answer"] == EXPECTED_ACTION_ITEMS
    assert body["final_answer"] == EXPECTED_ACTION_ITEMS

    # Real PDF preprocessing reached the normalized context.
    assert len(planner.calls) == 1

    context = planner.calls[0]

    assert context.query == "What are the action items?"
    assert len(context.extracted_inputs) == 1

    extracted_input = context.extracted_inputs[0]

    assert extracted_input.source_id
    assert extracted_input.source_id.startswith("source_")
    assert extracted_input.filename == "release_meeting.pdf"

    assert "Alice will add automated integration tests" in (
        extracted_input.content
    )

    assert "Bob will update the API documentation" in (
        extracted_input.content
    )

    assert "production monitoring alerts" in (
        extracted_input.content
    )

    assert (
        extracted_input.metadata.extraction_method
        == "direct_text"
    )

    # Real ConversationalAnswerTool reached the deterministic LLM.
    assert len(primary_provider.calls) == 1

    prompt = primary_provider.calls[0]

    assert "What are the action items?" in prompt

    assert extracted_input.content in prompt

    # Exact output contract: action items only.
    assert body["answer"].splitlines() == [
        "- Alice: Add automated integration tests before Friday.",
        "- Bob: Update the API documentation before deployment.",
        "- DevOps team: Configure production monitoring alerts.",
    ]

    assert "summary" not in body["answer"].lower()
    assert "release is scheduled" not in body["answer"].lower()

    # Public extracted-input projection is safe.
    assert len(body["extracted_inputs"]) == 1

    response_input = body["extracted_inputs"][0]

    assert response_input["source_id"] == extracted_input.source_id
    assert response_input["filename"] == "release_meeting.pdf"
    assert response_input["input_type"] == "pdf"

    assert "content" not in response_input

    # Frontend-ready plan and metadata.
    assert (
        body["plan"]["steps"][0]["tool"]
        == "conversational_answer"
    )

    assert body["plan"]["steps"][0]["status"] == "success"

    assert body["metadata"]["total_plan_steps"] == 1
    assert body["metadata"]["executed_steps"] == 1
    assert body["metadata"]["successful_steps"] == 1
    assert body["metadata"]["failed_steps"] == 0
    assert body["metadata"]["skipped_steps"] == 0