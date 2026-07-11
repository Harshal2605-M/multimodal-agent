from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class LLMProviderName(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"


class LLMGenerationResult(BaseModel):
    """
    Result of successful plain-text generation.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    content: str

    provider_used: LLMProviderName


StructuredOutputT = TypeVar(
    "StructuredOutputT",
    bound=BaseModel,
)


class LLMStructuredGenerationResult(
    BaseModel,
    Generic[StructuredOutputT],
):
    """
    Result of successful structured generation.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    output: StructuredOutputT

    provider_used: LLMProviderName