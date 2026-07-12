import io

from PIL import Image

from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
)
from app.config import Settings
from app.extractors.models import ExtractionMethod
from app.extractors.ocr import OCRResult
from app.llm.service import LLMService
from app.models.input import NormalizedContext
from app.tools.code_explanation import CodeExplanationTool

from tests.e2e.fakes import (
    DeterministicLLMProvider,
    build_e2e_agent_service,
)


OCR_CODE = (
    "def find_item(items, target):\n"
    "    for index in range(len(items)):\n"
    "        if items[index] == target:\n"
    "            return index\n"
    "    return -1"
)


EXPECTED_EXPLANATION = """Language:
Python

Explanation:
The function performs a linear search through the input list and returns the index of the first element equal to the target. If the target is not found, it returns -1.

Bugs/Issues:
No correctness bug exists for ordinary list inputs. Using enumerate(items) would make the loop more idiomatic and avoid repeated indexing.

Time Complexity:
O(n)

Space Complexity:
O(1)"""


def build_valid_png_bytes() -> bytes:
    """
    Create a structurally valid PNG entirely in memory.

    Real image decoding and extraction remain active. Only the
    external Tesseract OCR boundary is replaced deterministically.
    """

    buffer = io.BytesIO()

    image = Image.new(
        "RGB",
        (640, 320),
        "white",
    )

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def test_api_image_code_explanation_scenario(
    e2e_client,
    override_agent_service,
    monkeypatch,
) -> None:
    settings = Settings(
        app_env="testing",
    )

    ocr_calls = []

    def deterministic_ocr(image):
        ocr_calls.append(image)

        return OCRResult(
            text=OCR_CODE,
            confidence=97.5,
        )

    monkeypatch.setattr(
        "app.extractors.image.extract_text_with_ocr",
        deterministic_ocr,
    )

    primary_provider = DeterministicLLMProvider(
        responses=[
            EXPECTED_EXPLANATION,
        ]
    )

    fallback_provider = DeterministicLLMProvider(
        responses=[],
    )

    llm_service = LLMService(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )

    def build_code_explanation_plan(
        context: NormalizedContext,
    ) -> PlannerOutput:
        assert len(context.extracted_inputs) == 1

        source_id = context.extracted_inputs[0].source_id

        return PlannerOutput(
            goal=(
                "Explain the code extracted from the uploaded image."
            ),
            constraints=[
                "Use only the code extracted from the uploaded image.",
                "Identify the programming language.",
                "Explain what the code does.",
                "Identify bugs or issues.",
                "Provide time complexity.",
            ],
            needs_clarification=False,
            clarification_question=None,
            steps=[
                PlanStep(
                    id="step_1",
                    tool=ToolName.CODE_EXPLANATION,
                    input_reference=InputReference(
                        type=InputReferenceType.SOURCE,
                        source_id=source_id,
                    ),
                    depends_on=[],
                    reason=(
                        "Analyze the OCR-extracted code according "
                        "to the user's request."
                    ),
                )
            ],
        )

    service, planner = build_e2e_agent_service(
        settings=settings,
        plan=build_code_explanation_plan,
        tools=[
            CodeExplanationTool(
                llm_service=llm_service,
            ),
        ],
    )

    override_agent_service(service)

    response = e2e_client.post(
        "/agent/run",
        data={
            "query": (
                "Explain this code. Identify the programming language, "
                "explain what it does, identify bugs or issues, "
                "and provide the time complexity."
            ),
        },
        files=[
            (
                "files",
                (
                    "code.png",
                    build_valid_png_bytes(),
                    "image/png",
                ),
            ),
        ],
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["answer"] == EXPECTED_EXPLANATION
    assert body["final_answer"] == EXPECTED_EXPLANATION

    # Real image extractor reached the deterministic OCR boundary.
    assert len(ocr_calls) == 1

    # Planner received real normalized OCR content.
    assert len(planner.calls) == 1

    context = planner.calls[0]

    assert len(context.extracted_inputs) == 1

    extracted_input = context.extracted_inputs[0]

    assert extracted_input.source_id
    assert extracted_input.source_id.startswith("source_")
    assert extracted_input.filename == "code.png"
    assert "def find_item(items, target):" in extracted_input.content

    assert "for index in range(len(items)):" in (
        extracted_input.content
    )

    assert "if items[index] == target:" in (
        extracted_input.content
    )

    assert "return index" in extracted_input.content
    assert "return -1" in extracted_input.content

    assert (
        extracted_input.metadata.extraction_method
        == ExtractionMethod.OCR.value
    )

    # Real CodeExplanationTool reached deterministic LLM boundary.
    assert len(primary_provider.calls) == 1

    prompt = primary_provider.calls[0]

    assert "def find_item(items, target):" in prompt

    assert "for index in range(len(items)):" in prompt

    assert "if items[index] == target:" in prompt

    assert "return index" in prompt
    assert "return -1" in prompt
    assert "Identify the programming language" in prompt
    assert "identify bugs or issues" in prompt
    assert "time complexity" in prompt

    # Exact Scenario 3 output contract.
    final_answer = body["final_answer"]

    assert "Language:" in final_answer
    assert "Python" in final_answer

    assert "Explanation:" in final_answer

    assert "Bugs/Issues:" in final_answer

    assert "Time Complexity:" in final_answer
    assert "O(n)" in final_answer

    # Public response must not expose OCR content.
    assert len(body["extracted_inputs"]) == 1

    response_input = body["extracted_inputs"][0]

    assert response_input["source_id"] == extracted_input.source_id
    assert response_input["filename"] == "code.png"
    assert response_input["input_type"] == "image"

    assert "content" not in response_input

    # Frontend-ready plan and execution metadata.
    assert (
        body["plan"]["steps"][0]["tool"]
        == "code_explanation"
    )

    assert body["plan"]["steps"][0]["status"] == "success"

    assert body["metadata"]["total_plan_steps"] == 1
    assert body["metadata"]["executed_steps"] == 1
    assert body["metadata"]["successful_steps"] == 1
    assert body["metadata"]["failed_steps"] == 0
    assert body["metadata"]["skipped_steps"] == 0