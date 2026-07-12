from app.agent.schemas import (
    InputReference,
    InputReferenceType,
    PlannerOutput,
    PlanStep,
    ToolName,
)
from app.config import Settings
from app.llm.service import LLMService
from app.tools.conversational import ConversationalAnswerTool

from tests.e2e.fakes import (
    DeterministicLLMProvider,
    build_e2e_agent_service,
)


def test_real_api_pipeline_executes_deterministically(
    e2e_client,
    override_agent_service,
) -> None:
    settings = Settings(
        app_env="testing",
    )

    primary_provider = DeterministicLLMProvider(
        responses=[
            "Deterministic conversational answer.",
        ]
    )

    fallback_provider = DeterministicLLMProvider(
        responses=[],
    )

    llm_service = LLMService(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )

    plan = PlannerOutput(
        goal="Answer the user.",
        constraints=[],
        needs_clarification=False,
        clarification_question=None,
        steps=[
            PlanStep(
                id="step_1",
                tool=ToolName.CONVERSATIONAL_ANSWER,
                input_reference=InputReference(
                    type=InputReferenceType.QUERY_CONTEXT,
                ),
                depends_on=[],
                reason="Answer the user request.",
            )
        ],
    )

    service, planner = build_e2e_agent_service(
        settings=settings,
        plan=plan,
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
            "query": "Explain deterministic testing.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert (
        body["final_answer"]
        == "Deterministic conversational answer."
    )

    assert len(planner.calls) == 1
    assert (
        planner.calls[0].query
        == "Explain deterministic testing."
    )

    assert len(primary_provider.calls) == 1

    assert body["metadata"]["total_plan_steps"] == 1
    assert body["metadata"]["executed_steps"] == 1
    assert body["metadata"]["successful_steps"] == 1