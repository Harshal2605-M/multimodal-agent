from app.agent.plan_validation import (
    PlanValidationError,
    validate_plan,
)
from app.agent.prompts import (
    build_planner_prompt,
    build_planner_repair_prompt,
)
from app.agent.schemas import PlannerOutput
from app.llm.models import LLMStructuredGenerationResult
from app.llm.service import LLMService
from app.models.input import NormalizedContext


class Planner:
    """
    Application-level structured planner.

    The planner generates a structured PlannerOutput, validates it
    against the current NormalizedContext, and performs at most one
    repair generation when semantic validation fails.
    """

    def __init__(
        self,
        llm_service: LLMService,
        *,
        max_plan_steps: int,
    ) -> None:
        if max_plan_steps < 1:
            raise ValueError(
                "max_plan_steps must be at least 1."
            )

        self._llm_service = llm_service
        self._max_plan_steps = max_plan_steps

    def create_plan(
        self,
        context: NormalizedContext,
    ) -> LLMStructuredGenerationResult:
        """
        Generate and semantically validate a planner result.

        Exactly one repair generation is attempted when the initial
        PlannerOutput fails semantic validation.

        If the repaired output is also semantically invalid, the
        second PlanValidationError propagates to the caller.
        """

        initial_result = self._generate(
            build_planner_prompt(context)
        )

        try:
            self._validate(
                plan=initial_result.output,
                context=context,
            )
        except PlanValidationError as initial_error:
            repair_prompt = build_planner_repair_prompt(
                context=context,
                invalid_plan=initial_result.output,
                validation_errors=initial_error.errors,
            )

            repaired_result = self._generate(
                repair_prompt
            )

            self._validate(
                plan=repaired_result.output,
                context=context,
            )

            return repaired_result

        return initial_result

    def _generate(
        self,
        prompt: str,
    ) -> LLMStructuredGenerationResult:
        """
        Delegate structured planner generation to LLMService.
        """

        return self._llm_service.generate_structured(
            prompt=prompt,
            output_model=PlannerOutput,
        )

    def _validate(
        self,
        *,
        plan: PlannerOutput,
        context: NormalizedContext,
    ) -> None:
        """
        Validate one PlannerOutput using the configured plan limit.
        """

        validate_plan(
            plan=plan,
            context=context,
            max_plan_steps=self._max_plan_steps,
        )