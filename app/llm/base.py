from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.llm.models import (
    LLMGenerationResult,
    LLMStructuredGenerationResult,
)


class BaseLLMProvider(ABC):
    """
    Framework-independent provider contract.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> LLMGenerationResult:
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        output_model: type[BaseModel],
    ) -> LLMStructuredGenerationResult:
        raise NotImplementedError