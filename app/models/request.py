from pydantic import BaseModel, ConfigDict, Field


class AgentRunRequest(BaseModel):
    """
    Temporary Phase 2 request contract for POST /agent/run.

    File uploads and clarification continuation will be integrated
    in later phases.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        default="",
        max_length=10_000,
    )