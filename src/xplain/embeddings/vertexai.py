import typing as t

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.language_models import TextEmbeddingInput
from xplain.embeddings.base import BaseEmbedder
from xplain.embeddings.base import EmbedderFactory

class VertexAIEmbedder(BaseEmbedder):
    def __init__(self, project_id: str, location: str, model_name: str):
        """
        Initializes the VertexAIEmbedder.

        Args:
            project_id: Your Google Cloud project ID.
            location: The location of your Vertex AI resources (e.g., 'us-central1').
            model_name: The name of the Vertex AI text embedding model to use.
        """
        vertexai.init(project=project_id, location=location)
        self.model = TextEmbeddingModel.from_pretrained(model_name)
    
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
        input = TextEmbeddingInput(text, task)
        kwargs = dict(output_dimensionality=output_dimension)
        embeddings = self.model.get_embeddings([input], **kwargs)

        return embeddings[0].values
    
class VertexAIEmbedderFactory(EmbedderFactory):
    def create_embedder(**kwargs) -> BaseEmbedder:
        return VertexAIEmbedder(
            project_id=kwargs.get('PROJECT_ID'),
            location=kwargs.get('LOCATION'),
            model_name=kwargs.get('MODEL_NAME')
        )