import typing as t
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):

    @abstractmethod
    def generate_embeddings(
        self,
        text: str,
        output_dimension: t.Optional[int] = None,
    ) -> t.List[float]:
        ...
        
class EmbedderFactory(ABC):
    @abstractmethod
    def create_embedder(self, model_name: str, **kwargs) -> BaseEmbedder:
        ...