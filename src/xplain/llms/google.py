import typing as t

from google import genai
from google.genai import types
from xplain.llms.base import BaseLLM
from xplain.llms.base import LLMFactory

class GoogleAIStudioLLM(BaseLLM):
    def __init__(self, model_name: str, output_tokens: int, 
                 temperature: float, top_k: int, top_p: float):
        """
        Initializes the VertexAIGemini LLM.

        Args:
            project_id: Your Google Cloud project ID.
            location: The location of your Vertex AI resources (e.g., 'us-central1').
            model_name: The name of the Vertex AI Gemini model to use.
            output_tokens: The maximum number of tokens to generate in the response.
            temperature: Controls the randomness of the generated text.
            top_k: The model samples from the `top_k` most likely tokens at each step.
            top_p: The model samples from the smallest set of 
                tokens whose cumulative probability exceeds `top_p`.
        """
        self.client = genai.Client(api_key='API_KEY')
        self.model = model_name
        self.output_tokens = output_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
    
    def generate_text(
        self,
        prompt: str,
        output_tokens = 65535,   
        top_k: int = 40,
        top_p: float = 0.95,
        temperature: float = 0,
    ) -> str:
        """
        Generates text using the Vertex AI Gemini model.

        Args:
            prompt: The input prompt to generate text from.
            output_tokens: The maximum number of tokens to generate
                in the response. Defaults to 65535.
            temperature: Controls the randomness of the generated text
                (default is very low).
            top_k: The model samples from the `top_k` most likely
                tokens at each step. Defaults to 40.
            top_p: The model samples from the smallest set of
                tokens whose cumulative probability exceeds `top_p`.

        Returns:
            The generated text.
        """
        response = self.client.models.generate_content(
                    model=self.model,
                    config=types.GenerateContentConfig(temperature=temperature,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        max_output_tokens=output_tokens),
                    contents=[prompt]
                )

        return response.candidates[0].content.parts[0].text
    
class GoogleAIStudioLLMFactory(LLMFactory):
    def create_llm(**kwargs) -> BaseLLM:
        return GoogleAIStudioLLM(
            model_name=kwargs.get('MODEL_NAME'),
            output_tokens = kwargs.get("output_tokens", 65535),   
            top_k = kwargs.get("top_k", 40),
            top_p = kwargs.get("top_p", 0.95),
            temperature = kwargs.get("temperature", 0)
        )