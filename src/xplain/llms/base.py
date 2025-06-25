import typing as t
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        output_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> str:
        ...

class LLMFactory(ABC):
    @abstractmethod
    def create_llm(self, model_name: str, **kwargs) -> BaseLLM:
        ...