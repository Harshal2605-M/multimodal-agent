from app.agent.schemas import (
    InputReferenceType,
    PlannerOutput,
)
from app.models.input import NormalizedContext


class PlanValidationError(ValueError):
    """
    Raised when a schema-valid PlannerOutput is semantically invalid
    against the current NormalizedContext or complete plan.
    """

    def __init__(
        self,
        errors: list[str],
    ) -> None:
        if not errors:
            raise ValueError(
                "PlanValidationError requires at least one validation error."
            )

        self.errors = list(errors)

        super().__init__(
            "Semantic plan validation failed: "
            + "; ".join(self.errors)
        )


def validate_plan(
    plan: PlannerOutput,
    context: NormalizedContext,
    *,
    max_plan_steps: int,
) -> None:
    """
    Validate a schema-valid PlannerOutput against the current context.

    All discoverable semantic errors are collected before raising so
    a later repair attempt can receive useful validation feedback.

    Pydantic-owned structural validation is intentionally not
    duplicated here.
    """

    if max_plan_steps < 1:
        raise ValueError(
            "max_plan_steps must be at least 1."
        )

    errors: list[str] = []

    if len(plan.steps) > max_plan_steps:
        errors.append(
            f"Plan contains {len(plan.steps)} steps but the maximum "
            f"allowed is {max_plan_steps}."
        )

    source_ids = {
        extracted_input.source_id
        for extracted_input in context.extracted_inputs
    }

    step_ids = [
        step.id
        for step in plan.steps
    ]

    known_step_ids = set(step_ids)

    if len(known_step_ids) != len(step_ids):
        errors.append(
            "Plan step IDs must be unique."
        )

    step_positions = {
        step.id: position
        for position, step in enumerate(plan.steps)
    }

    for position, step in enumerate(plan.steps):
        reference = step.input_reference

        if reference.type is InputReferenceType.SOURCE:
            if reference.source_id not in source_ids:
                errors.append(
                    f"Step '{step.id}' references unknown source_id "
                    f"'{reference.source_id}'."
                )

        elif reference.type is InputReferenceType.SOURCES:
            assert reference.source_ids is not None

            unknown_source_ids = [
                source_id
                for source_id in reference.source_ids
                if source_id not in source_ids
            ]

            if unknown_source_ids:
                errors.append(
                    f"Step '{step.id}' references unknown source_ids: "
                    f"{unknown_source_ids}."
                )

        elif reference.type is InputReferenceType.ALL_SOURCES:
            if not context.extracted_inputs:
                errors.append(
                    f"Step '{step.id}' uses ALL_SOURCES but the "
                    "context contains no extracted inputs."
                )

        elif reference.type is InputReferenceType.STEP_OUTPUT:
            assert reference.step_id is not None

            referenced_step_id = reference.step_id

            if referenced_step_id not in known_step_ids:
                errors.append(
                    f"Step '{step.id}' references unknown step_id "
                    f"'{referenced_step_id}'."
                )

            elif step_positions[referenced_step_id] >= position:
                errors.append(
                    f"Step '{step.id}' references step_id "
                    f"'{referenced_step_id}' that is not an earlier step."
                )

            if referenced_step_id not in step.depends_on:
                errors.append(
                    f"Step '{step.id}' uses STEP_OUTPUT from "
                    f"'{referenced_step_id}' but does not declare it "
                    "in depends_on."
                )

        elif reference.type is InputReferenceType.DETECTED_URLS:
            if not context.detected_urls:
                errors.append(
                    f"Step '{step.id}' uses DETECTED_URLS but the "
                    "context contains no detected URLs."
                )

        for dependency_id in step.depends_on:
            if dependency_id not in known_step_ids:
                errors.append(
                    f"Step '{step.id}' depends on unknown step_id "
                    f"'{dependency_id}'."
                )

            elif step_positions[dependency_id] >= position:
                errors.append(
                    f"Step '{step.id}' depends on step_id "
                    f"'{dependency_id}' that is not an earlier step."
                )

    if errors:
        raise PlanValidationError(errors)