import typing as t
import os

from google import genai
from google.genai import types
from xplain.embeddings.base import BaseEmbedder
from xplain.embeddings.base import EmbedderFactory

class GoogleAIStudioEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        """
        Initializes the VertexAIEmbedder.

        Args:
            model_name: The name of the Vertex AI text embedding model to use.
        """
        key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if not key:
            raise ValueError("'GOOGLE_AI_STUDIO_API_KEY' environment variable is not set.")
        self.client = genai.Client(api_key=key)
        self.model = model_name
    
    def generate_embeddings(
            self,
            text: str,
            output_dimension: t.Optional[int] = 768,
            task: str = "SEMANTIC_SIMILARITY"
        ) -> t.List[float]:
        """
        Generates embeddings for the given text using the underlying embedding model.

        Args:
            text: The input text to generate embeddings for.
            output_dimension: The desired dimensionality of the output embeddings. Defaults to 768.
            task: The specific task for which the embeddings are being generated. This can influence 
                how the model generates the embeddings. Defaults to "SEMANTIC_SIMILARITY".

        Returns:
            A list of floats representing the embedding vector for the input text.
        """

        result = self.client.models.embed_content(   
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task,
                                                output_dimensionality=output_dimension))


        return result.embeddings[0].values
    
class GoogleAIStudioEmbedderFactory(EmbedderFactory):
    def create_embedder(**kwargs) -> BaseEmbedder:
        return GoogleAIStudioEmbedder(
            model_name=kwargs.get('MODEL_NAME')
        )